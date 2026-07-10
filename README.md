# LLM2Rec: Large Language Models Are Powerful Embedding Models for Sequential Recommendation

## Introduction
This is the code implementation for our paper on KDD'25 "LLM2Rec: Large Language Models Are Powerful Embedding Models for Sequential Recommendation".

## Environments
To execute the code correctly, the following python packages are required:

- `torch >= 2.6.0`  
- `transformers >= 4.44.2`  
- `llm2vec == 0.2.3`  
- `flash-attn >= 2.7.4`

## Datasets
The zipped datasets used in this paper can be downloaded from this [link](https://drive.google.com/file/d/1GIXWaaaNuUkUtuFy5JTN0OwAQiLGb2z4/view?usp=sharing). Please unzip the dataset files under directory `./data` .

## Training

LLM2Rec follows a two-stage training pipeline:

1. **Collaborative Supervised Fine-Tuning (CSFT)**  
   Fine-tunes a pre-trained LLM to capture collaborative filtering (CF) signals using user interaction sequences as training data.

2. **Item-level Embedding Modeling (IEM)**  
   Converts the CF-aware LLM into an embedding generator.

### Run training

We provide example shell scripts for training:

```bash
# Stage 1: Collaborative Supervised Fine-Tuning
bash run_LLM2Rec_CSFT.sh

# Stage 2: Item-level Embedding Modeling
bash run_LLM2Rec_IEM.sh
```

Please change the necessary configs of your own device (e.g. path of the saved pre-trained LLMs) before executing.

### Experimental: Reliability-Calibrated Local CF Distillation

The optional IEM2-only CF distillation pipeline keeps CSFT, MNTP, and downstream recommenders unchanged. It requires a SASRec teacher item embedding stored as a tensor whose rows exactly match `data/AmazonMix-6/5-core/info/item_titles.txt` line order.

```bash
TEACHER_ITEM_EMB_PATH=/path/to/sasrec_teacher_item_emb.pt \
PYTHON_BIN=/path/to/venv/bin/python \
bash run_LLM2Rec_IEM_CF_Distill.sh
```

The script precomputes top-16 teacher neighborhoods, then runs the uncalibrated and reliability-calibrated configurations on one GPU. Set `SEQUENCE_PATH=""` to omit support weighting; reliability then uses teacher confidence only. Use `llm2rec/train_simcse_cfkd_debug_config.json` for a five-step smoke test first. The baseline `llm2rec/train_simcse_config.json` leaves distillation disabled (`cf_distill_lambda = 0`) and retains its original SimCSE behavior.

## Evaluation

We integrate the evaluation process, including embedding extraction and training downstream sequential recommenders, into one script, which can be easily executed by
```bash
bash script_extract_and_evaluate.sh
```

You can change the paths of the saved checkpoints to evaluate in the config part of the script_extract_and_evaluate.sh script.


## Citation
If you find our repo useful, please consider citing:
```bibtex
@inproceedings{he2025llm2rec,
  title={LLM2Rec: Large Language Models Are Powerful Embedding Models for Sequential Recommendation},
  author={He, Yingzhi and Liu, Xiaohao and Zhang, An and Ma, Yunshan and Chua, Tat-Seng},
  booktitle={Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2},
  pages={896--907},
  year={2025}
}
```

## Acknowledgements

The code implementation is based on previous repos, including [llm2vec](https://github.com/McGill-NLP/llm2vec), [recbole](https://github.com/RUCAIBox/RecBole), and [DecodingMatters](https://github.com/SAI990323/DecodingMatters).
