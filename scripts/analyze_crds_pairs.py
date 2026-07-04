import argparse
import json
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-path", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--method", default="")
    return parser.parse_args()


def summarize_tensor(values: torch.Tensor, prefix: str):
    if values.numel() == 0:
        return {
            f"avg_{prefix}": 0.0,
            f"std_{prefix}": 0.0,
            f"p25_{prefix}": 0.0,
            f"p50_{prefix}": 0.0,
            f"p75_{prefix}": 0.0,
        }
    values = values.float()
    return {
        f"avg_{prefix}": float(values.mean().item()),
        f"std_{prefix}": float(values.std(unbiased=False).item()),
        f"p25_{prefix}": float(torch.quantile(values, 0.25).item()),
        f"p50_{prefix}": float(torch.quantile(values, 0.50).item()),
        f"p75_{prefix}": float(torch.quantile(values, 0.75).item()),
    }


def main():
    args = parse_args()
    pair_data = torch.load(args.pair_path, map_location="cpu", weights_only=False)
    neg_items = pair_data["neg_items"]
    neg_weights = pair_data["neg_weights"]
    pair_semantic = pair_data.get("pair_semantic_similarity", pair_data.get("neg_semantic_similarity"))
    pair_cf = pair_data.get("pair_cf_distance", pair_data.get("neg_cf_distance"))
    pair_reliability = pair_data.get("pair_reliability")
    if pair_reliability is None:
        reliability = pair_data["reliability"]
        pair_reliability = torch.minimum(
            reliability.unsqueeze(1).expand_as(neg_items),
            reliability[neg_items.clamp_min(0)],
        )
        pair_reliability = pair_reliability * (neg_items >= 0)

    valid_mask = (neg_items >= 0) & (neg_weights > 0)
    num_items = int(neg_items.size(0))
    per_item_counts = valid_mask.sum(dim=1)
    num_items_with_valid_neg = int((per_item_counts > 0).sum().item())
    coverage = float(num_items_with_valid_neg / num_items) if num_items > 0 else 0.0

    payload = {
        "dataset": args.dataset or pair_data.get("meta", {}).get("dataset", ""),
        "method": args.method,
        "pair_path": args.pair_path,
        "num_items": num_items,
        "num_items_with_valid_neg": num_items_with_valid_neg,
        "coverage": coverage,
        "avg_num_neg_per_item": float(per_item_counts.float().mean().item()),
        "median_num_neg_per_item": float(per_item_counts.float().median().item()),
        "meta": pair_data.get("meta", {}),
    }
    payload.update(summarize_tensor(pair_semantic[valid_mask], "semantic_similarity"))
    payload.update(summarize_tensor(pair_cf[valid_mask], "cf_distance"))
    payload.update(summarize_tensor(pair_reliability[valid_mask], "reliability"))
    payload.update(summarize_tensor(neg_weights[valid_mask], "neg_weight"))

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
