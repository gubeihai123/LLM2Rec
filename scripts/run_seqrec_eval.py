import argparse
import json
import os
import shutil

from seqrec.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--embedding", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--model", default="SASRec")
    parser.add_argument("--rand-seed", type=int, default=2024)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--loss-type", default="ce")
    parser.add_argument("--run-id", default="quick_eval")
    parser.add_argument("--exp-type", default="srec")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--saved-ckpt", default="")
    return parser.parse_args()


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def main():
    args = parse_args()
    config = {
        "dataset": args.dataset,
        "embedding": args.embedding,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "loss_type": args.loss_type,
        "rand_seed": args.rand_seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "eval_interval": args.eval_interval,
        "save": args.save,
        "run_id": args.run_id,
        "exp_type": args.exp_type,
    }
    runner = Runner(model_name=args.model, config_dict=config)
    result, resolved_config = runner.run()
    saved_model_ckpt = ""
    if args.save:
        saved_model_ckpt = runner.trainer.saved_model_ckpt
        if args.saved_ckpt:
            os.makedirs(os.path.dirname(args.saved_ckpt), exist_ok=True)
            shutil.copy2(saved_model_ckpt, args.saved_ckpt)
            saved_model_ckpt = args.saved_ckpt

    payload = {
        "model": args.model,
        "config": to_jsonable(dict(resolved_config)),
        "result": to_jsonable(dict(result)),
        "saved_model_ckpt": saved_model_ckpt,
    }
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
