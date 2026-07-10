#!/usr/bin/env python3
"""Precompute local CF teacher neighborhoods for optional IEM SimCSE distillation."""

import argparse
import hashlib
import math
import os

import torch
from torch.nn import functional as F


def parse_bool(value: str) -> bool:
    normalized = value.lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("Expected one of: true, false, 1, 0, yes, no.")


def read_item_titles(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle]


def item_titles_checksum(item_titles: list[str]) -> str:
    return hashlib.sha256("\n".join(item_titles).encode("utf-8")).hexdigest()


def count_sequence_support(
    sequence_path: str, num_items: int, support_cap: int
) -> torch.Tensor:
    if support_cap <= 0:
        raise ValueError("support_cap must be positive.")

    counts = torch.zeros(num_items, dtype=torch.float32)
    with open(sequence_path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.split()
            if not fields:
                continue
            # Sequential recommendation files in this repository use: user_id item_id ...
            for raw_item_id in fields[1:]:
                try:
                    item_id = int(raw_item_id)
                except ValueError as error:
                    raise ValueError(
                        f"Invalid item id {raw_item_id!r} on line {line_number} of {sequence_path}."
                    ) from error
                if item_id < 0 or item_id >= num_items:
                    raise ValueError(
                        f"Item id {item_id} on line {line_number} is outside [0, {num_items})."
                    )
                counts[item_id] += 1

    return (torch.log1p(counts) / math.log1p(support_cap)).clamp(max=1.0)


def precompute_neighbors(
    teacher_embeddings: torch.Tensor,
    top_k: int,
    teacher_tau: float,
    chunk_size: int,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if teacher_embeddings.ndim != 2:
        raise ValueError("teacher_item_emb_path must contain a rank-2 tensor [num_items, dim].")
    num_items = teacher_embeddings.shape[0]
    if num_items < 2:
        raise ValueError("At least two items are required to compute non-self neighbors.")
    if top_k <= 0 or top_k >= num_items:
        raise ValueError("top_k must be in [1, num_items - 1].")
    if teacher_tau <= 0:
        raise ValueError("teacher_tau must be positive.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    compute_device = torch.device(device)
    if compute_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested for precomputation but is not available.")
    teacher_embeddings = F.normalize(teacher_embeddings.float(), dim=-1).to(compute_device)
    neighbor_ids = torch.empty((num_items, top_k), dtype=torch.long)
    teacher_scores = torch.empty((num_items, top_k), dtype=torch.float32)

    for start in range(0, num_items, chunk_size):
        stop = min(start + chunk_size, num_items)
        similarities = teacher_embeddings[start:stop] @ teacher_embeddings.T
        local_rows = torch.arange(stop - start, device=compute_device)
        similarities[local_rows, start + local_rows] = -torch.inf
        scores, ids = torch.topk(similarities, k=top_k, dim=-1)
        neighbor_ids[start:stop] = ids.cpu()
        teacher_scores[start:stop] = scores.float().cpu()

    teacher_probs = F.softmax(teacher_scores / teacher_tau, dim=-1)
    return neighbor_ids, teacher_probs, teacher_scores


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute top-K teacher CF neighborhoods for IEM SimCSE distillation."
    )
    parser.add_argument("--teacher_item_emb_path", required=True)
    parser.add_argument("--item_titles_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--top_k", type=int, required=True)
    parser.add_argument("--teacher_tau", type=float, default=0.05)
    parser.add_argument("--sequence_path", default=None)
    parser.add_argument("--support_cap", type=int, default=100)
    parser.add_argument("--use_reliability", type=parse_bool, default=True)
    parser.add_argument("--chunk_size", type=int, default=4096)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for chunked similarity computation, for example cpu or cuda.",
    )
    args = parser.parse_args()

    teacher_embeddings = torch.load(args.teacher_item_emb_path, map_location="cpu")
    if not isinstance(teacher_embeddings, torch.Tensor):
        raise ValueError("teacher_item_emb_path must contain a torch.Tensor, not a checkpoint dictionary.")
    item_titles = read_item_titles(args.item_titles_path)
    if teacher_embeddings.shape[0] != len(item_titles):
        raise ValueError(
            "Teacher embedding rows must align exactly with item_titles.txt lines: "
            f"got {teacher_embeddings.shape[0]} embeddings and {len(item_titles)} titles."
        )

    neighbor_ids, teacher_probs, teacher_scores = precompute_neighbors(
        teacher_embeddings, args.top_k, args.teacher_tau, args.chunk_size, args.device
    )
    if args.use_reliability:
        if args.sequence_path:
            support_weight = count_sequence_support(
                args.sequence_path, len(item_titles), args.support_cap
            )
        else:
            support_weight = torch.ones(len(item_titles), dtype=torch.float32)
        if args.top_k == 1:
            teacher_confidence = torch.ones(len(item_titles), dtype=torch.float32)
        else:
            entropy = -(teacher_probs * teacher_probs.clamp_min(1e-12).log()).sum(dim=-1)
            teacher_confidence = (1.0 - entropy / math.log(args.top_k)).clamp(0.0, 1.0)
        # TODO: incorporate transition lift and temporal stability in later revisions.
        reliability = support_weight * teacher_confidence
    else:
        reliability = torch.ones(len(item_titles), dtype=torch.float32)

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(
        {
            "neighbor_ids": neighbor_ids,
            "teacher_probs": teacher_probs,
            "teacher_scores": teacher_scores,
            "reliability": reliability,
            "meta": {
                "top_k": args.top_k,
                "teacher_tau": args.teacher_tau,
                "num_items": len(item_titles),
                "use_reliability": args.use_reliability,
                "source": "precompute_cf_distill.py",
                "item_titles_sha256": item_titles_checksum(item_titles),
            },
        },
        args.output_path,
    )
    print(
        f"Saved CF distillation data for {len(item_titles)} items with top-{args.top_k} "
        f"neighbors to {args.output_path}."
    )


if __name__ == "__main__":
    main()
