import argparse
import ast
from pathlib import Path

from modules.crds_utils import build_crds_pairs


def parse_args():
    parser = argparse.ArgumentParser(description="Build CRDS hard negatives for LLM2Rec SimCSE/IEM.")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--item_sem_emb", type=str, default=None)
    parser.add_argument("--item_cf_emb", type=str, default=None)
    parser.add_argument("--user_cf_emb", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--raw_train_path", type=str, default=None)
    parser.add_argument("--sequence_path", type=str, default=None)
    parser.add_argument("--item_info_path", type=str, default=None)
    parser.add_argument("--user_id_map_path", type=str, default=None)
    parser.add_argument("--semantic_topk", type=int, default=None)
    parser.add_argument("--num_hard_neg", type=int, default=None)
    parser.add_argument("--K_user", type=int, default=None)
    parser.add_argument("--M_item", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--lambda_user", type=float, default=None)
    parser.add_argument("--tau_n", type=float, default=None)
    parser.add_argument("--delta_r", type=float, default=None)
    parser.add_argument("--gamma_s", type=float, default=None)
    parser.add_argument("--gamma_c", type=float, default=None)
    parser.add_argument("--quantile", type=float, default=None)
    parser.add_argument("--global_hard_filter_quantile", type=float, default=None)
    parser.add_argument("--chunk_size", type=int, default=None)
    parser.add_argument("--kmeans_iters", type=int, default=None)
    parser.add_argument("--kmeans_seed", type=int, default=None)
    parser.add_argument("--kmeans_device", type=str, default=None)
    parser.add_argument("--crds_no_reliability", action="store_true")
    parser.add_argument("--crds_no_transition", action="store_true")
    parser.add_argument("--crds_no_cooccurrence", action="store_true")
    return parser.parse_args()


def load_config(path: str):
    if not path:
        return {}
    config = {}
    with Path(path).open("r") as file:
        for raw_line in file:
            line = raw_line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            config[key.strip()] = parse_scalar(value.strip())
    return config


def parse_scalar(value: str):
    if value in {"", "null", "None", "~"}:
        return None
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def main():
    args = parse_args()
    config = load_config(args.config)

    merged = dict(config)
    for key, value in vars(args).items():
        if key == "config":
            continue
        if key in {"crds_no_reliability", "crds_no_transition", "crds_no_cooccurrence"}:
            if value:
                merged[key] = value
            continue
        if value is not None:
            merged[key] = value

    dataset = merged.get("dataset") or merged.get("crds_dataset")
    item_sem_emb = (
        merged.get("item_sem_emb")
        or merged.get("crds_item_sem_embedding")
        or merged.get("embedding")
    )
    item_cf_emb = merged.get("item_cf_emb") or merged.get("crds_item_cf_embedding")
    user_cf_emb = merged.get("user_cf_emb") or merged.get("crds_user_cf_embedding")
    output = merged.get("output") or merged.get("crds_pairs_path") or "crds_pairs.pt"

    if not dataset:
        raise ValueError("Need `dataset` or `crds_dataset`.")
    if not item_sem_emb:
        raise ValueError("Need `item_sem_emb`, `crds_item_sem_embedding`, or `embedding`.")
    if not item_cf_emb:
        raise ValueError("Need `item_cf_emb` or `crds_item_cf_embedding`.")
    if not user_cf_emb:
        raise ValueError("Need `user_cf_emb` or `crds_user_cf_embedding`.")

    pair_mode = "no_cooccurrence" if merged.get("crds_no_cooccurrence", False) else "crds"
    use_reliability = not merged.get("crds_no_reliability", False)
    use_transition = not merged.get("crds_no_transition", False)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    build_crds_pairs(
        dataset=dataset,
        item_sem_path=item_sem_emb,
        item_cf_path=item_cf_emb,
        user_cf_path=user_cf_emb,
        output_path=output_path,
        data_root=merged.get("data_root") or merged.get("crds_data_root", "data"),
        raw_train_path=merged.get("raw_train_path") or merged.get("crds_raw_train_path"),
        sequence_path=merged.get("sequence_path") or merged.get("crds_sequence_path"),
        item_info_path=merged.get("item_info_path") or merged.get("crds_item_info_path"),
        user_id_map_path=merged.get("user_id_map_path") or merged.get("crds_user_id_map_path"),
        semantic_topk=merged.get("semantic_topk", 200),
        num_hard_neg=merged.get("num_hard_neg", 20),
        k_user=merged.get("K_user", 64),
        m_item=merged.get("M_item", 64),
        alpha=merged.get("alpha", 10.0),
        beta=merged.get("beta", 10.0),
        lambda_user=merged.get("lambda_user", 0.5),
        tau_n=merged.get("tau_n", 20.0),
        delta_r=merged.get("delta_r", 0.3),
        gamma_s=merged.get("gamma_s", 10.0),
        gamma_c=merged.get("gamma_c", 10.0),
        quantile=merged.get("crds_quantile", merged.get("quantile", 0.75)),
        global_hard_filter_quantile=merged.get(
            "crds_global_hard_filter_quantile",
            merged.get("global_hard_filter_quantile", 0.75),
        ),
        use_transition=use_transition,
        use_reliability=use_reliability,
        pair_mode=pair_mode,
        chunk_size=merged.get("crds_build_chunk_size", merged.get("chunk_size", 1024)),
        kmeans_iters=merged.get("crds_kmeans_iters", merged.get("kmeans_iters", 25)),
        kmeans_seed=merged.get("crds_kmeans_seed", merged.get("kmeans_seed", 2024)),
        kmeans_device=merged.get("crds_kmeans_device", merged.get("kmeans_device", "auto")),
    )
    print(f"saved CRDS pairs to {output_path}")


if __name__ == "__main__":
    main()
