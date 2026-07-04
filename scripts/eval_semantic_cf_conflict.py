import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from seqrec.models import SASRec
from seqrec.recdata import NormalRecData
from seqrec.utils import get_config, init_device, init_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pair-path", required=True)
    parser.add_argument("--baseline-embedding", required=True)
    parser.add_argument("--baseline-ckpt", required=True)
    parser.add_argument("--method-embedding", required=True)
    parser.add_argument("--method-ckpt", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--rand-seed", type=int, default=2024)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--loss-type", default="ce")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--top-neg", type=int, default=1)
    return parser.parse_args()


def build_config(args, embedding_path):
    config = get_config(
        model_name="SASRec",
        config_file=None,
        config_dict={
            "dataset": args.dataset,
            "embedding": embedding_path,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "loss_type": args.loss_type,
            "rand_seed": args.rand_seed,
            "epochs": args.epochs,
            "patience": args.patience,
            "eval_interval": args.eval_interval,
            "save": True,
            "run_id": "diagnostic",
            "exp_type": "srec",
        },
    )
    config["device"], config["use_ddp"] = init_device()
    init_seed(config["rand_seed"], config["reproducibility"])
    data = NormalRecData(config).load_data()
    config["select_pool"] = data[3]
    config["item_num"] = data[4]
    config["eos_token"] = data[4] + 1
    return config, data[2]


def load_model(config, embedding_path, ckpt_path):
    pretrained_item_embeddings = torch.tensor(
        np.load(embedding_path), dtype=torch.float32
    ).to(config["device"])
    model = SASRec(config, pretrained_item_embeddings)
    state_dict = torch.load(ckpt_path, map_location=config["device"], weights_only=False)
    model.load_state_dict(state_dict)
    model.to(config["device"])
    model.eval()
    return model


def score_batch(model, batch, item_ids):
    with torch.no_grad():
        batch = {
            key: value.to(model.config["device"]) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        state_hidden = model.get_representation(batch).view(-1, model.config["hidden_size"])
        item_emb = model.get_all_embeddings(state_hidden.device)
        gathered = item_emb[item_ids]
        return (state_hidden.unsqueeze(1) * gathered).sum(dim=-1)


def evaluate_method(model, dataset, pair_data, batch_size, top_neg):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    neg_items = pair_data["neg_items"]
    neg_weights = pair_data["neg_weights"]
    total_triples = 0
    total_correct = 0
    margin_sum = 0.0
    covered_users = set()
    covered_pos_items = set()

    user_offset = 0
    for batch in dataloader:
        labels = batch["labels"]
        all_neg_ids = neg_items[labels]
        all_neg_weights = neg_weights[labels]
        valid_mask = (all_neg_ids >= 0) & (all_neg_weights > 0)
        if top_neg == 1:
            has_valid = valid_mask.any(dim=1)
            chosen_neg_ids = torch.full_like(labels, -1)
            if has_valid.any():
                first_valid = valid_mask.float().argmax(dim=1)
                chosen_neg_ids[has_valid] = all_neg_ids[torch.arange(labels.size(0))[has_valid], first_valid[has_valid]]
            row_mask = chosen_neg_ids >= 0
            if not row_mask.any():
                user_offset += labels.size(0)
                continue
            pos_scores = score_batch(model, batch, labels.to(model.config["device"]).unsqueeze(1))[row_mask, 0]
            neg_scores = score_batch(
                model,
                {key: value[row_mask] if torch.is_tensor(value) and value.size(0) == row_mask.size(0) else value for key, value in batch.items()},
                chosen_neg_ids[row_mask].to(model.config["device"]).unsqueeze(1),
            )[:, 0]
            total_correct += int((pos_scores > neg_scores).sum().item())
            margin_sum += float((pos_scores - neg_scores).sum().item())
            total_triples += int(row_mask.sum().item())
            for local_idx in torch.nonzero(row_mask, as_tuple=False).squeeze(-1).tolist():
                covered_users.add(user_offset + local_idx)
                covered_pos_items.add(int(labels[local_idx].item()))
            user_offset += labels.size(0)
        else:
            raise NotImplementedError("top_neg > 1 is not implemented")

    return {
        "pairwise_acc": float(total_correct / total_triples) if total_triples > 0 else 0.0,
        "avg_margin": float(margin_sum / total_triples) if total_triples > 0 else 0.0,
        "num_triples": total_triples,
        "num_users": len(dataset),
        "coverage_users": float(len(covered_users) / len(dataset)) if len(dataset) > 0 else 0.0,
        "coverage_pos_items": float(len(covered_pos_items) / len({int(dataset[i]['labels']) for i in range(len(dataset))})) if len(dataset) > 0 else 0.0,
    }


def main():
    args = parse_args()
    pair_data = torch.load(args.pair_path, map_location="cpu", weights_only=False)

    baseline_config, test_dataset = build_config(args, args.baseline_embedding)
    method_config, _ = build_config(args, args.method_embedding)

    baseline_model = load_model(baseline_config, args.baseline_embedding, args.baseline_ckpt)
    method_model = load_model(method_config, args.method_embedding, args.method_ckpt)

    baseline_stats = evaluate_method(baseline_model, test_dataset, pair_data, args.batch_size, args.top_neg)
    method_stats = evaluate_method(method_model, test_dataset, pair_data, args.batch_size, args.top_neg)

    payload = {
        "dataset": args.dataset,
        "pair_path": args.pair_path,
        "baseline_embedding": args.baseline_embedding,
        "baseline_ckpt": args.baseline_ckpt,
        "method_embedding": args.method_embedding,
        "method_ckpt": args.method_ckpt,
        "baseline_pairwise_acc": baseline_stats["pairwise_acc"],
        "method_pairwise_acc": method_stats["pairwise_acc"],
        "delta_pairwise_acc": method_stats["pairwise_acc"] - baseline_stats["pairwise_acc"],
        "baseline_avg_margin": baseline_stats["avg_margin"],
        "method_avg_margin": method_stats["avg_margin"],
        "delta_avg_margin": method_stats["avg_margin"] - baseline_stats["avg_margin"],
        "num_triples": method_stats["num_triples"],
        "num_users": method_stats["num_users"],
        "coverage_users": method_stats["coverage_users"],
        "coverage_pos_items": method_stats["coverage_pos_items"],
        "baseline_stats": baseline_stats,
        "method_stats": method_stats,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
