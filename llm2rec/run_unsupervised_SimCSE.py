import logging
from dataclasses import dataclass, field
import hashlib
import os

import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger

import transformers
from transformers import (
    MODEL_FOR_MASKED_LM_MAPPING,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    set_seed,
)
from transformers.trainer_utils import seed_worker

from peft import LoraConfig, get_peft_model

from llm2vec import LLM2Vec
from dataset_utils import load_dataset
from llm2vec.loss.utils import load_loss

from tqdm import tqdm

transformers.logging.set_verbosity_error()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = get_logger(__name__, log_level="INFO")
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
_ORIGINAL_TORCH_LOAD = torch.load


def _torch_load_trusted_checkpoint(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _ORIGINAL_TORCH_LOAD(*args, **kwargs)


def initialize_peft(
    model,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_modules: Optional[List[str]] = None,
):
    if lora_modules is None and model.config.__class__.__name__ in [
        "LlamaConfig",
        "MistralConfig",
        "GemmaConfig",
        "Qwen2Config",
    ]:
        lora_modules = [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    elif lora_modules is None:
        raise ValueError("lora_modules must be specified for this model.")

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=None,
    )

    model = get_peft_model(model, config)
    print(f"Model's Lora trainable parameters:")
    model.print_trainable_parameters()
    return model


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The base model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    peft_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": ("The PEFT model checkpoint to add on top of base model.")},
    )
    bidirectional: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether to enable bidirectional attention in the model. If set to False, the model will use unidirectional attention."
            )
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    attn_implementation: Optional[str] = field(
        default="sdpa",
        metadata={
            "help": ("The attention implementation to use in the model."),
            "choices": ["eager", "sdpa", "flash_attention_2"],
        },
    )
    pooling_mode: Optional[str] = field(
        default="mean",
        metadata={
            "help": ("The pooling mode to use in the model."),
            "choices": ["mean", "weighted_mean", "eos_token"],
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use. Options: E5"},
    )
    dataset_file_path: Optional[str] = field(
        default=None, metadata={"help": "The input training data file or folder."}
    )
    # TODO: implement this
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )


@dataclass
class CustomArguments:
    """
    Custom arguments for the script
    """

    simcse_dropout: float = field(
        default=0.1, metadata={"help": "The SimCSE dropout rate for the model"}
    )

    lora_dropout: float = field(
        default=0.05, metadata={"help": "The dropout rate for lora"}
    )

    lora_r: int = field(default=8, metadata={"help": "The r value for lora"})

    stop_after_n_steps: int = field(
        default=10000, metadata={"help": "Stop training after n steps"}
    )

    experiment_id: Optional[str] = field(
        default=None, metadata={"help": "The experiment id"}
    )

    loss_class: Optional[str] = field(
        default="HardNegativeNLLLoss",
        metadata={
            "help": "The loss class to use for training. Options: HardNegativeNLLLoss"
        },
    )

    loss_scale: float = field(
        default=50.0, metadata={"help": "The loss scale for the loss function"}
    )

    cf_distill_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a precomputed local CF distillation .pt file."},
    )

    cf_distill_lambda: float = field(
        default=0.0,
        metadata={"help": "Weight of the optional local CF distillation loss."},
    )

    cf_distill_student_tau: float = field(
        default=0.05,
        metadata={"help": "Temperature for student local-neighborhood similarities."},
    )

    cf_distill_top_k: Optional[int] = field(
        default=None,
        metadata={"help": "Use at most this many precomputed CF neighbors per anchor."},
    )

    cf_distill_anchor_subsample: Optional[int] = field(
        default=None,
        metadata={"help": "Randomly use at most this many anchors from each batch for CF KD."},
    )

    cf_distill_weight_mode: str = field(
        default="reliability",
        metadata={"help": "CF KD weighting mode: none or reliability."},
    )


@dataclass
class DefaultCollator:
    model: LLM2Vec

    def __init__(self, model: LLM2Vec, include_item_ids: bool = False) -> None:
        self.model = model
        self.include_item_ids = include_item_ids

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = features
        num_texts = len(batch[0].texts)
        texts = [[] for _ in range(num_texts)]
        labels = []

        for example in batch:
            for idx, text in enumerate(example.texts):
                # TODO: Add prepare_for_tokenization here similar to supervised training and see if it impacts performance
                texts[idx].append(text)
            labels.append(example.label)
        labels = torch.tensor(labels)

        sentence_features = []
        for idx in range(num_texts):
            tokenized = self.model.tokenize(texts[idx])
            sentence_features.append(tokenized)

        # Preserve the original tuple exactly unless the optional KD path is enabled.
        if not self.include_item_ids:
            return sentence_features, labels

        item_ids = torch.tensor([int(example.guid) for example in batch], dtype=torch.long)
        return {"features": sentence_features, "labels": labels, "item_ids": item_ids}


class StopTrainingCallback(TrainerCallback):
    def __init__(self, stop_after_n_steps: int):
        self.stop_after_n_steps = stop_after_n_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.stop_after_n_steps:
            control.should_training_stop = True


class SimCSETrainer(Trainer):
    def __init__(
        self,
        *args,
        loss_function=None,
        cf_distill_path: Optional[str] = None,
        cf_distill_lambda: float = 0.0,
        cf_distill_student_tau: float = 0.05,
        cf_distill_top_k: Optional[int] = None,
        cf_distill_anchor_subsample: Optional[int] = None,
        cf_distill_weight_mode: str = "reliability",
        cf_item_texts: Optional[List[str]] = None,
        cf_item_titles: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.loss_function = loss_function
        self.cf_distill_lambda = cf_distill_lambda
        self.cf_distill_student_tau = cf_distill_student_tau
        self.cf_distill_top_k = cf_distill_top_k
        self.cf_distill_anchor_subsample = cf_distill_anchor_subsample
        self.cf_distill_weight_mode = cf_distill_weight_mode
        self.cf_item_texts = cf_item_texts
        self.cf_item_titles = cf_item_titles
        self.cf_neighbor_ids_cpu = None
        self.cf_teacher_probs_cpu = None
        self.cf_reliability_cpu = None

        self.cf_distill_enabled = bool(cf_distill_path) and cf_distill_lambda > 0
        if not self.cf_distill_enabled:
            if cf_distill_lambda > 0:
                logger.warning(
                    "CF distillation is disabled because cf_distill_path was not provided."
                )
            return

        if cf_distill_weight_mode not in {"none", "reliability"}:
            raise ValueError(
                "cf_distill_weight_mode must be one of: none, reliability."
            )
        if cf_distill_student_tau <= 0:
            raise ValueError("cf_distill_student_tau must be positive.")
        if cf_distill_top_k is not None and cf_distill_top_k <= 0:
            raise ValueError("cf_distill_top_k must be positive when provided.")
        if (
            cf_distill_anchor_subsample is not None
            and cf_distill_anchor_subsample <= 0
        ):
            raise ValueError(
                "cf_distill_anchor_subsample must be positive when provided."
            )
        if not cf_item_texts:
            raise ValueError("CF distillation requires item texts from ItemTitleData.")

        cf_data = torch.load(cf_distill_path, map_location="cpu")
        required_keys = {"neighbor_ids", "teacher_probs", "reliability", "meta"}
        missing_keys = required_keys.difference(cf_data)
        if missing_keys:
            raise ValueError(
                f"Invalid CF distillation file; missing keys: {sorted(missing_keys)}"
            )

        neighbor_ids = cf_data["neighbor_ids"].long().contiguous().cpu()
        teacher_probs = cf_data["teacher_probs"].float().contiguous().cpu()
        reliability = cf_data["reliability"].float().contiguous().cpu()
        if neighbor_ids.ndim != 2 or teacher_probs.shape != neighbor_ids.shape:
            raise ValueError(
                "neighbor_ids and teacher_probs must both have shape [num_items, top_k]."
            )
        if reliability.shape != (neighbor_ids.shape[0],):
            raise ValueError("reliability must have shape [num_items].")
        if len(cf_item_texts) != neighbor_ids.shape[0]:
            raise ValueError(
                "CF distillation item count does not match ItemTitleData. "
                f"Expected {len(cf_item_texts)}, got {neighbor_ids.shape[0]}."
            )
        expected_titles_checksum = cf_data["meta"].get("item_titles_sha256")
        if expected_titles_checksum is not None:
            if not cf_item_titles:
                raise ValueError("CF distillation title checksum requires item titles.")
            actual_titles_checksum = hashlib.sha256(
                "\n".join(cf_item_titles).encode("utf-8")
            ).hexdigest()
            if actual_titles_checksum != expected_titles_checksum:
                raise ValueError(
                    "CF distillation titles do not match ItemTitleData order; refusing to guess an item-id mapping."
                )
        if neighbor_ids.numel() and (
            neighbor_ids.min().item() < 0
            or neighbor_ids.max().item() >= neighbor_ids.shape[0]
        ):
            raise ValueError("CF distillation neighbor_ids contain invalid item ids.")
        if not torch.isfinite(teacher_probs).all() or not torch.isfinite(reliability).all():
            raise ValueError("CF distillation probabilities and reliability must be finite.")

        self.cf_neighbor_ids_cpu = neighbor_ids
        self.cf_teacher_probs_cpu = teacher_probs
        self.cf_reliability_cpu = reliability.clamp(0.0, 1.0)
        logger.info(
            "Enabled local CF distillation with %d items and top-%d neighbors.",
            neighbor_ids.shape[0],
            neighbor_ids.shape[1],
        )

    @staticmethod
    def _move_features_to_device(
        features: Dict[str, torch.Tensor], device: torch.device
    ) -> Dict[str, torch.Tensor]:
        return {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in features.items()
        }

    def _compute_cf_kd_loss(
        self, q_reps: torch.Tensor, item_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        anchor_ids_cpu = item_ids.detach().cpu().long()
        if self.cf_distill_anchor_subsample is not None:
            anchor_count = min(self.cf_distill_anchor_subsample, anchor_ids_cpu.numel())
            selected = torch.randperm(anchor_ids_cpu.numel())[:anchor_count]
            anchor_ids_cpu = anchor_ids_cpu[selected]
            q_reps = q_reps[selected.to(q_reps.device)]

        neighbor_ids = self.cf_neighbor_ids_cpu[anchor_ids_cpu]
        teacher_probs = self.cf_teacher_probs_cpu[anchor_ids_cpu]
        if self.cf_distill_top_k is not None:
            top_k = min(self.cf_distill_top_k, neighbor_ids.shape[1])
            neighbor_ids = neighbor_ids[:, :top_k]
            teacher_probs = teacher_probs[:, :top_k]

        # Re-normalize in case cf_distill_top_k truncates the precomputed distribution.
        teacher_probs = teacher_probs / teacher_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        flat_neighbor_ids = neighbor_ids.reshape(-1)
        unique_neighbor_ids, inverse = torch.unique(flat_neighbor_ids, return_inverse=True)
        neighbor_texts = [self.cf_item_texts[item_id] for item_id in unique_neighbor_ids.tolist()]
        neighbor_features = self.model.tokenize(neighbor_texts)
        neighbor_features = self._move_features_to_device(neighbor_features, q_reps.device)
        unique_neighbor_reps = self.model(neighbor_features)
        neighbor_reps = unique_neighbor_reps[inverse.to(q_reps.device)].reshape(
            neighbor_ids.shape[0], neighbor_ids.shape[1], -1
        )

        anchor_reps = F.normalize(q_reps.float(), dim=-1)
        neighbor_reps = F.normalize(neighbor_reps.float(), dim=-1)
        student_scores = torch.einsum("bd,bkd->bk", anchor_reps, neighbor_reps)
        student_log_probs = F.log_softmax(
            student_scores / self.cf_distill_student_tau, dim=-1
        )
        kd_per_item = -(teacher_probs.to(student_log_probs.device) * student_log_probs).sum(dim=-1)

        if self.cf_distill_weight_mode == "none":
            weights = torch.ones_like(kd_per_item)
        else:
            weights = self.cf_reliability_cpu[anchor_ids_cpu].to(kd_per_item.device)
        cf_kd_loss = (weights * kd_per_item).sum() / weights.sum().clamp_min(1e-6)
        return cf_kd_loss, weights.mean()

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        if self.cf_distill_enabled:
            features = inputs["features"]
            labels = inputs["labels"]
            item_ids = inputs["item_ids"]
        else:
            features, labels = inputs
            item_ids = None
        q_reps = self.model(features[0])
        d_reps = self.model(features[1])

        d_reps_neg = None
        if len(features) > 2:
            d_reps_neg = self.model(features[2])

        simcse_loss = self.loss_function(q_reps, d_reps, d_reps_neg)
        loss = simcse_loss

        if self.cf_distill_enabled:
            cf_kd_loss, cf_weight_mean = self._compute_cf_kd_loss(q_reps, item_ids)
            loss = simcse_loss + self.cf_distill_lambda * cf_kd_loss
            self.log(
                {
                    "simcse_loss": simcse_loss.detach().float().item(),
                    "cf_kd_loss": cf_kd_loss.detach().float().item(),
                    "total_loss": loss.detach().float().item(),
                    "cf_weight_mean": cf_weight_mean.detach().float().item(),
                    "cf_distill_lambda": self.cf_distill_lambda,
                }
            )

        if return_outputs:
            output = torch.cat(
                [model(row)["sentence_embedding"][:, None] for row in features], dim=1
            )
            return loss, output

        return loss

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        self.model.save(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, CustomArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, custom_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        (
            model_args,
            data_args,
            training_args,
            custom_args,
        ) = parser.parse_args_into_dataclasses()
    if training_args.ddp_find_unused_parameters:
        kwargs = [
            DistributedDataParallelKwargs(
                dim=0,
                broadcast_buffers=True,
                bucket_cap_mb=25,
                find_unused_parameters=True,
                check_reduction=False,
                gradient_as_bucket_view=False,
            )
        ]
    else:
        kwargs = []
    accelerator = Accelerator(kwargs_handlers=kwargs)

    set_seed(training_args.seed)

    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    train_dataset = load_dataset(
        data_args.dataset_name,
        split="train",
        file_path=data_args.dataset_file_path,
    )

    train_examples = [
        train_dataset[i]
        for i in tqdm(
            range(len(train_dataset)),
            desc="Loading train examples...",
            disable=not accelerator.is_main_process,
        )
    ]

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    model = LLM2Vec.from_pretrained(
        base_model_name_or_path=model_args.model_name_or_path,
        enable_bidirectional=model_args.bidirectional,
        peft_model_name_or_path=model_args.peft_model_name_or_path,
        merge_peft=True,
        pooling_mode=model_args.pooling_mode,
        max_length=model_args.max_seq_length,
        torch_dtype=torch_dtype,
        attn_implementation=model_args.attn_implementation,
        attention_dropout=custom_args.simcse_dropout,
    )

    # model organization is LLM2VecModel.model -> HF Model, we have to apply PEFT to the inner model
    if custom_args.lora_r is not None:
        model.model = initialize_peft(
            model.model,
            lora_r=custom_args.lora_r,
            lora_alpha=2 * custom_args.lora_r,
            lora_dropout=custom_args.lora_dropout,
        )
    else:
        for param in model.model.parameters():
            param.requires_grad = True

    tokenizer = model.tokenizer
    if not hasattr(model, "_keys_to_ignore_on_save"):
        model._keys_to_ignore_on_save = None

    train_loss = load_loss(custom_args.loss_class, scale=custom_args.loss_scale)

    cf_distill_enabled = bool(custom_args.cf_distill_path) and custom_args.cf_distill_lambda > 0
    data_collator = DefaultCollator(model, include_item_ids=cf_distill_enabled)

    print(training_args)
    
    trainer = SimCSETrainer(
        model=model,
        args=training_args,
        train_dataset=train_examples,
        data_collator=data_collator,
        tokenizer=tokenizer,
        loss_function=train_loss,
        cf_distill_path=custom_args.cf_distill_path,
        cf_distill_lambda=custom_args.cf_distill_lambda,
        cf_distill_student_tau=custom_args.cf_distill_student_tau,
        cf_distill_top_k=custom_args.cf_distill_top_k,
        cf_distill_anchor_subsample=custom_args.cf_distill_anchor_subsample,
        cf_distill_weight_mode=custom_args.cf_distill_weight_mode,
        cf_item_texts=getattr(train_dataset, "item_texts", None),
        cf_item_titles=getattr(train_dataset, "item_titles", None),
    )

    if custom_args.stop_after_n_steps is not None:
        trainer.add_callback(StopTrainingCallback(custom_args.stop_after_n_steps))

    if training_args.resume_from_checkpoint is not None:
        torch.load = _torch_load_trusted_checkpoint
    try:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    finally:
        torch.load = _ORIGINAL_TORCH_LOAD


if __name__ == "__main__":
    main()
