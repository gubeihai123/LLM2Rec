from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


DATASET_SOURCE_DICT = {
    "AmazonMix-6": {
        "raw_train_path": "AmazonMix-6/5-core/train/AmazonMix-6.csv",
        "sequence_path": None,
        "item_info_path": "AmazonMix-6/5-core/info/AmazonMix-6.txt",
        "item_titles_path": "AmazonMix-6/5-core/info/item_titles.txt",
    },
    "Mix6": {
        "raw_train_path": "AmazonMix-6/5-core/train/AmazonMix-6.csv",
        "sequence_path": None,
        "item_info_path": "AmazonMix-6/5-core/info/AmazonMix-6.txt",
        "item_titles_path": "AmazonMix-6/5-core/info/item_titles.txt",
    },
    "Goodreads": {
        "raw_train_path": None,
        "sequence_path": "Goodreads/clean/data.txt",
        "item_info_path": None,
        "item_titles_path": None,
    },
    "Games_5core": {
        "raw_train_path": "Video_Games/5-core/train/Video_Games_5_1996-9-2023-10.csv",
        "sequence_path": "Video_Games/5-core/downstream/data.txt",
        "item_info_path": None,
        "item_titles_path": None,
    },
    "Movies_5core": {
        "raw_train_path": "Movies_and_TV/5-core/train/Movies_and_TV_5_2019-9-2023-10.csv",
        "sequence_path": "Movies_and_TV/5-core/downstream/data.txt",
        "item_info_path": None,
        "item_titles_path": None,
    },
    "Arts_5core": {
        "raw_train_path": "Arts_Crafts_and_Sewing/5-core/train/Arts_Crafts_and_Sewing_5_2014-9-2023-10.csv",
        "sequence_path": "Arts_Crafts_and_Sewing/5-core/downstream/data.txt",
        "item_info_path": None,
        "item_titles_path": None,
    },
    "Sports_5core": {
        "raw_train_path": "Sports_and_Outdoors/5-core/train/Sports_and_Outdoors_5_2014-9-2023-10.csv",
        "sequence_path": "Sports_and_Outdoors/5-core/downstream/data.txt",
        "item_info_path": None,
        "item_titles_path": None,
    },
    "Baby_5core": {
        "raw_train_path": "Baby_Products/5-core/train/Baby_Products_5_1996-9-2023-10.csv",
        "sequence_path": "Baby_Products/5-core/downstream/data.txt",
        "item_info_path": None,
        "item_titles_path": None,
    },
}


def load_tensor_file(path: Union[str, Path]) -> torch.Tensor:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".npy":
        tensor = torch.from_numpy(np.load(file_path))
    elif suffix in {".pt", ".pth"}:
        obj = torch.load(file_path, map_location="cpu")
        if isinstance(obj, torch.Tensor):
            tensor = obj
        elif isinstance(obj, np.ndarray):
            tensor = torch.from_numpy(obj)
        elif isinstance(obj, dict):
            tensor = _extract_tensor_from_mapping(obj)
        else:
            raise TypeError(f"Unsupported tensor payload type: {type(obj)!r}")
    else:
        raise ValueError(f"Unsupported tensor file: {file_path}")
    return tensor.detach().cpu().float()


def _extract_tensor_from_mapping(mapping: Dict[str, Any]) -> torch.Tensor:
    candidate_keys = (
        "tensor",
        "data",
        "weight",
        "embedding",
        "embeddings",
        "item_embedding",
        "item_embeddings",
        "user_embedding",
        "user_embeddings",
    )
    for key in candidate_keys:
        value = mapping.get(key)
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value)
    raise KeyError(f"Cannot find tensor payload in keys: {sorted(mapping.keys())}")


def resolve_dataset_paths(
    dataset: Optional[str],
    data_root: Union[str, Path] = "data",
) -> Dict[str, Optional[Path]]:
    if not dataset:
        return {}
    if dataset not in DATASET_SOURCE_DICT:
        raise KeyError(f"Unsupported dataset: {dataset}")
    root = Path(data_root)
    spec = DATASET_SOURCE_DICT[dataset]
    resolved = {}
    for key, relative_path in spec.items():
        resolved[key] = root / relative_path if relative_path else None
    return resolved


def infer_item_num(
    dataset: Optional[str],
    data_root: Union[str, Path],
    raw_train_path: Optional[Union[str, Path]],
    sequence_path: Optional[Union[str, Path]],
    item_info_path: Optional[Union[str, Path]],
    sem_rows: int,
) -> int:
    dataset_paths = resolve_dataset_paths(dataset, data_root=data_root) if dataset else {}
    resolved_item_info = Path(item_info_path) if item_info_path else dataset_paths.get("item_info_path")
    if resolved_item_info and resolved_item_info.exists():
        return count_items_from_info(resolved_item_info)

    resolved_sequence = Path(sequence_path) if sequence_path else dataset_paths.get("sequence_path")
    if resolved_sequence and resolved_sequence.exists():
        seq_min, seq_max = read_sequence_item_stats(resolved_sequence)
        return seq_max + 1 if seq_min == 0 else seq_max

    resolved_raw_train = Path(raw_train_path) if raw_train_path else dataset_paths.get("raw_train_path")
    if resolved_raw_train and resolved_raw_train.exists():
        raw_min, raw_max = read_raw_item_stats(resolved_raw_train)
        return raw_max + 1 if raw_min == 0 else raw_max

    return sem_rows


def count_items_from_info(path: Union[str, Path]) -> int:
    count = 0
    with Path(path).open("r") as file:
        for count, _ in enumerate(file, 1):
            pass
    return count


def read_raw_item_stats(path: Union[str, Path]) -> Tuple[int, int]:
    frame = pd.read_csv(path, usecols=["item_id"])
    return int(frame["item_id"].min()), int(frame["item_id"].max())


def read_sequence_item_stats(path: Union[str, Path]) -> Tuple[int, int]:
    min_item_id = None
    max_item_id = None
    with Path(path).open("r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            values = list(map(int, line.split()))
            line_min = min(values)
            line_max = max(values)
            min_item_id = line_min if min_item_id is None else min(min_item_id, line_min)
            max_item_id = line_max if max_item_id is None else max(max_item_id, line_max)
    if min_item_id is None or max_item_id is None:
        raise ValueError(f"No sequence items found in {path}")
    return min_item_id, max_item_id


def infer_item_index_base(
    item_min: int,
    item_max: int,
    item_num: int,
) -> int:
    if item_min == 0 and item_max <= item_num - 1:
        return 0
    if item_min >= 1 and item_max <= item_num:
        return 1
    if item_max == item_num - 1:
        return 0
    if item_max == item_num:
        return 1
    raise ValueError(
        f"Cannot infer item index base from min={item_min}, max={item_max}, item_num={item_num}"
    )


def align_item_tensor_for_pairs(
    tensor: torch.Tensor,
    item_num: int,
    item_index_base: int,
) -> torch.Tensor:
    output = tensor.detach().cpu().float()
    if output.dim() != 2:
        raise ValueError(f"Expected a 2D item tensor, got shape {tuple(output.shape)}")
    if output.size(0) == item_num:
        return output
    if output.size(0) == item_num + 1 and item_index_base == 1:
        return output[1:].clone()
    raise ValueError(
        f"Unexpected item tensor row count {output.size(0)} for item_num={item_num} "
        f"and item_index_base={item_index_base}"
    )


def stable_global_quantile(
    values: torch.Tensor,
    quantile: float,
    sample_size: int = 2_000_000,
    seed: int = 2024,
) -> torch.Tensor:
    flat_cpu = values.detach().reshape(-1).float().cpu()
    if flat_cpu.numel() == 0:
        raise ValueError("Cannot compute quantile for an empty tensor.")
    try:
        return torch.quantile(flat_cpu, quantile)
    except RuntimeError as exc:
        logger.warning(
            "Global quantile on %s values failed with %s; falling back to sampled estimate (%s values).",
            flat_cpu.numel(),
            exc,
            min(sample_size, flat_cpu.numel()),
        )
        if flat_cpu.numel() > sample_size:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            indices = torch.randint(
                flat_cpu.numel(),
                (sample_size,),
                generator=generator,
            )
            flat_cpu = flat_cpu.index_select(0, indices)
        return torch.quantile(flat_cpu, quantile)


def build_crds_pairs(
    dataset: Optional[str],
    item_sem_path: Union[str, Path],
    item_cf_path: Union[str, Path],
    user_cf_path: Union[str, Path],
    output_path: Union[str, Path],
    data_root: Union[str, Path] = "data",
    raw_train_path: Optional[Union[str, Path]] = None,
    sequence_path: Optional[Union[str, Path]] = None,
    item_info_path: Optional[Union[str, Path]] = None,
    user_id_map_path: Optional[Union[str, Path]] = None,
    semantic_topk: int = 200,
    num_hard_neg: int = 20,
    k_user: int = 64,
    m_item: int = 64,
    alpha: float = 10.0,
    beta: float = 10.0,
    lambda_user: float = 0.5,
    tau_n: float = 20.0,
    delta_r: float = 0.3,
    gamma_s: float = 10.0,
    gamma_c: float = 10.0,
    quantile: float = 0.75,
    use_transition: bool = True,
    use_reliability: bool = True,
    pair_mode: str = "crds",
    global_hard_filter_quantile: Optional[float] = 0.75,
    chunk_size: int = 1024,
    kmeans_iters: int = 25,
    kmeans_seed: int = 2024,
    kmeans_device: str = "cpu",
) -> Dict[str, Any]:
    raw_sem_emb = load_tensor_file(item_sem_path)
    raw_item_cf_emb = load_tensor_file(item_cf_path)
    user_cf_emb = load_tensor_file(user_cf_path)

    item_num = infer_item_num(
        dataset=dataset,
        data_root=data_root,
        raw_train_path=raw_train_path,
        sequence_path=sequence_path,
        item_info_path=item_info_path,
        sem_rows=raw_sem_emb.size(0),
    )

    dataset_paths = resolve_dataset_paths(dataset, data_root=data_root) if dataset else {}
    resolved_raw_train = Path(raw_train_path) if raw_train_path else dataset_paths.get("raw_train_path")
    resolved_sequence = Path(sequence_path) if sequence_path else dataset_paths.get("sequence_path")

    if resolved_raw_train and resolved_raw_train.exists():
        item_min, item_max = read_raw_item_stats(resolved_raw_train)
    elif resolved_sequence and resolved_sequence.exists():
        item_min, item_max = read_sequence_item_stats(resolved_sequence)
    else:
        item_min, item_max = 0, item_num - 1

    item_index_base = infer_item_index_base(item_min, item_max, item_num)
    sem_emb = align_item_tensor_for_pairs(raw_sem_emb, item_num, item_index_base)
    item_cf_emb = align_item_tensor_for_pairs(raw_item_cf_emb, item_num, item_index_base)

    user_communities = run_kmeans(
        user_cf_emb,
        n_clusters=k_user,
        n_iters=kmeans_iters,
        seed=kmeans_seed,
        device=kmeans_device,
    )
    item_communities = None
    if use_transition:
        item_communities = run_kmeans(
            item_cf_emb,
            n_clusters=m_item,
            n_iters=kmeans_iters,
            seed=kmeans_seed,
            device=kmeans_device,
        )

    stats = collect_interaction_statistics(
        dataset=dataset,
        item_num=item_num,
        item_index_base=item_index_base,
        user_communities=user_communities,
        item_communities=item_communities,
        data_root=data_root,
        raw_train_path=resolved_raw_train,
        sequence_path=resolved_sequence,
        user_id_map_path=user_id_map_path,
        expected_num_users=user_cf_emb.size(0),
        build_item_user_sets=(pair_mode == "no_cooccurrence"),
    )

    reliability = compute_item_reliability(stats["item_counts"], tau_n=tau_n)
    if not use_reliability:
        reliability = torch.ones_like(reliability)

    user_dist = build_smoothed_distribution(stats["user_comm_counts"], alpha=alpha)
    if use_transition and stats["trans_comm_counts"] is not None:
        trans_dist = build_smoothed_distribution(stats["trans_comm_counts"], alpha=beta)
    else:
        trans_dist = None

    semantic_indices, semantic_similarities = semantic_topk_search(
        sem_emb,
        topk=semantic_topk,
        chunk_size=chunk_size,
        device=kmeans_device,
    )
    candidate_pair_reliability = torch.minimum(
        reliability.unsqueeze(1).expand(-1, semantic_indices.size(1)),
        reliability[semantic_indices],
    )
    candidate_cf_distance = None
    if pair_mode != "no_cooccurrence":
        candidate_cf_distance = torch.zeros_like(semantic_similarities)
        for item_idx in range(item_num):
            candidate_idx = semantic_indices[item_idx]
            d_user = js_divergence(
                user_dist[item_idx].unsqueeze(0),
                user_dist[candidate_idx],
            ).squeeze(0)
            if trans_dist is not None:
                d_trans = js_divergence(
                    trans_dist[item_idx].unsqueeze(0),
                    trans_dist[candidate_idx],
                ).squeeze(0)
                candidate_cf_distance[item_idx] = lambda_user * d_user + (1.0 - lambda_user) * d_trans
            else:
                candidate_cf_distance[item_idx] = d_user

    global_semantic_threshold = None
    global_cf_threshold = None
    if global_hard_filter_quantile is not None:
        global_hard_filter_quantile = float(global_hard_filter_quantile)
        if not 0.0 < global_hard_filter_quantile < 1.0:
            raise ValueError("global_hard_filter_quantile must be in (0, 1).")
        global_semantic_threshold = stable_global_quantile(
            semantic_similarities,
            global_hard_filter_quantile,
            seed=kmeans_seed,
        )
        if candidate_cf_distance is not None:
            valid_cf_mask = torch.ones_like(candidate_cf_distance, dtype=torch.bool)
            if use_reliability:
                valid_cf_mask &= candidate_pair_reliability >= delta_r
            valid_cf_values = candidate_cf_distance[valid_cf_mask]
            if valid_cf_values.numel() > 0:
                global_cf_threshold = stable_global_quantile(
                    valid_cf_values,
                    global_hard_filter_quantile,
                    seed=kmeans_seed,
                )

    neg_items = torch.full((item_num, num_hard_neg), -1, dtype=torch.long)
    neg_weights = torch.zeros((item_num, num_hard_neg), dtype=torch.float32)
    neg_semantic_similarity = torch.zeros((item_num, num_hard_neg), dtype=torch.float32)
    neg_cf_distance = torch.zeros((item_num, num_hard_neg), dtype=torch.float32)
    neg_pair_reliability = torch.zeros((item_num, num_hard_neg), dtype=torch.float32)
    item_user_sets = stats.get("item_user_sets")

    for item_idx in range(item_num):
        candidate_idx = semantic_indices[item_idx]
        semantic_similarity = semantic_similarities[item_idx]
        pair_reliability = candidate_pair_reliability[item_idx]
        cf_distance = (
            torch.zeros_like(semantic_similarity)
            if candidate_cf_distance is None
            else candidate_cf_distance[item_idx]
        )
        candidate_mask = torch.ones_like(candidate_idx, dtype=torch.bool)

        if pair_mode == "no_cooccurrence":
            candidate_mask &= filter_no_cooccurrence_candidates(item_idx, candidate_idx, item_user_sets)

        if use_reliability:
            candidate_mask &= pair_reliability >= delta_r

        if global_semantic_threshold is not None:
            candidate_mask &= semantic_similarity >= global_semantic_threshold

        if pair_mode != "no_cooccurrence" and global_cf_threshold is not None:
            candidate_mask &= cf_distance >= global_cf_threshold

        candidate_idx = candidate_idx[candidate_mask]
        semantic_similarity = semantic_similarity[candidate_mask]
        pair_reliability = pair_reliability[candidate_mask]
        cf_distance = cf_distance[candidate_mask]

        if candidate_idx.numel() == 0:
            continue

        if pair_mode == "no_cooccurrence":
            hard_score = normalize_values(semantic_similarity) * pair_reliability
        else:
            hard_score = (
                normalize_values(semantic_similarity)
                * normalize_values(cf_distance)
                * pair_reliability
            )

        top_count = min(num_hard_neg, candidate_idx.numel())
        _, top_positions = torch.topk(hard_score, k=top_count, largest=True)
        chosen_idx = candidate_idx[top_positions]
        chosen_semantic_similarity = semantic_similarity[top_positions]
        chosen_cf_distance = cf_distance[top_positions]
        chosen_pair_reliability = pair_reliability[top_positions]

        delta_s_value = torch.quantile(semantic_similarity, quantile)
        semantic_weight = torch.sigmoid(gamma_s * (chosen_semantic_similarity - delta_s_value))

        if pair_mode == "no_cooccurrence":
            chosen_weight = chosen_pair_reliability * semantic_weight
        else:
            delta_c_value = torch.quantile(cf_distance, quantile)
            cf_weight = torch.sigmoid(gamma_c * (chosen_cf_distance - delta_c_value))
            chosen_weight = chosen_pair_reliability * semantic_weight * cf_weight

        positive_mask = chosen_weight > 0
        if not positive_mask.any():
            continue
        chosen_idx = chosen_idx[positive_mask]
        chosen_weight = chosen_weight[positive_mask]
        chosen_semantic_similarity = chosen_semantic_similarity[positive_mask]
        chosen_cf_distance = chosen_cf_distance[positive_mask]
        top_count = chosen_idx.numel()

        neg_items[item_idx, :top_count] = chosen_idx.long()
        neg_weights[item_idx, :top_count] = chosen_weight.float()
        neg_semantic_similarity[item_idx, :top_count] = chosen_semantic_similarity.float()
        neg_cf_distance[item_idx, :top_count] = chosen_cf_distance.float()
        neg_pair_reliability[item_idx, :top_count] = chosen_pair_reliability.float()

    output = {
        "neg_items": neg_items,
        "neg_weights": neg_weights,
        "reliability": reliability.float(),
        "neg_semantic_similarity": neg_semantic_similarity,
        "neg_cf_distance": neg_cf_distance,
        "pair_semantic_similarity": neg_semantic_similarity.clone(),
        "pair_cf_distance": neg_cf_distance.clone(),
        "pair_reliability": neg_pair_reliability,
        "meta": {
            "dataset": dataset,
            "item_num": item_num,
            "item_index_base": item_index_base,
            "semantic_topk": semantic_topk,
            "num_hard_neg": num_hard_neg,
            "k_user": k_user,
            "m_item": m_item,
            "alpha": alpha,
            "beta": beta,
            "lambda_user": lambda_user,
            "tau_n": tau_n,
            "delta_r": delta_r,
            "gamma_s": gamma_s,
            "gamma_c": gamma_c,
            "quantile": quantile,
            "use_transition": use_transition,
            "use_reliability": use_reliability,
            "pair_mode": pair_mode,
            "global_hard_filter_quantile": global_hard_filter_quantile,
            "global_semantic_threshold": (
                float(global_semantic_threshold.item())
                if global_semantic_threshold is not None
                else None
            ),
            "global_cf_threshold": (
                float(global_cf_threshold.item())
                if global_cf_threshold is not None
                else None
            ),
            "valid_pair_count": int((neg_weights > 0).sum().item()),
        },
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    return output


def collect_interaction_statistics(
    dataset: Optional[str],
    item_num: int,
    item_index_base: int,
    user_communities: torch.Tensor,
    item_communities: Optional[torch.Tensor],
    data_root: Union[str, Path] = "data",
    raw_train_path: Optional[Union[str, Path]] = None,
    sequence_path: Optional[Union[str, Path]] = None,
    user_id_map_path: Optional[Union[str, Path]] = None,
    expected_num_users: Optional[int] = None,
    build_item_user_sets: bool = False,
) -> Dict[str, Any]:
    dataset_paths = resolve_dataset_paths(dataset, data_root=data_root) if dataset else {}
    resolved_raw_train = Path(raw_train_path) if raw_train_path else dataset_paths.get("raw_train_path")
    if resolved_raw_train and resolved_raw_train.exists():
        try:
            return collect_raw_train_statistics(
                raw_train_path=resolved_raw_train,
                item_num=item_num,
                item_index_base=item_index_base,
                user_communities=user_communities,
                item_communities=item_communities,
                expected_num_users=expected_num_users,
                user_id_map_path=user_id_map_path,
                build_item_user_sets=build_item_user_sets,
            )
        except ValueError as error:
            logger.warning(
                "Falling back from raw-train statistics to sequence statistics for %s: %s",
                resolved_raw_train,
                error,
            )

    resolved_sequence = Path(sequence_path) if sequence_path else dataset_paths.get("sequence_path")
    if not resolved_sequence or not resolved_sequence.exists():
        raise FileNotFoundError("Need either raw_train_path or sequence_path to build CRDS pairs.")

    return collect_sequence_statistics(
        sequence_path=resolved_sequence,
        item_num=item_num,
        item_index_base=item_index_base,
        user_communities=user_communities,
        item_communities=item_communities,
        expected_num_users=expected_num_users,
        build_item_user_sets=build_item_user_sets,
    )


def collect_raw_train_statistics(
    raw_train_path: Union[str, Path],
    item_num: int,
    item_index_base: int,
    user_communities: torch.Tensor,
    item_communities: Optional[torch.Tensor],
    expected_num_users: Optional[int] = None,
    user_id_map_path: Optional[Union[str, Path]] = None,
    build_item_user_sets: bool = False,
) -> Dict[str, Any]:
    frame = pd.read_csv(raw_train_path, usecols=["user_id", "history_item_id", "item_id"])
    if frame.empty:
        raise ValueError(f"No interaction rows found in {raw_train_path}")

    user_map = load_user_id_map(user_id_map_path)
    if user_map is None:
        user_map = {}
        for user_id in frame["user_id"]:
            if user_id not in user_map:
                user_map[user_id] = len(user_map)

    if expected_num_users is not None and len(user_map) != expected_num_users:
        raise ValueError(
            f"user_cf_emb rows ({expected_num_users}) do not match raw-train users ({len(user_map)})."
        )

    num_user_clusters = int(user_communities.max().item()) + 1
    num_item_clusters = int(item_communities.max().item()) + 1 if item_communities is not None else 0
    item_counts = torch.zeros(item_num, dtype=torch.float32)
    user_comm_counts = torch.zeros(item_num, num_user_clusters, dtype=torch.float32)
    trans_comm_counts = (
        torch.zeros(item_num, num_item_clusters, dtype=torch.float32)
        if item_communities is not None
        else None
    )
    item_user_sets = [set() for _ in range(item_num)] if build_item_user_sets else None

    for row in frame.itertuples(index=False):
        user_idx = user_map.get(row.user_id)
        if user_idx is None or user_idx >= user_communities.numel():
            continue

        item_idx = int(row.item_id) - item_index_base
        if not 0 <= item_idx < item_num:
            continue
        item_counts[item_idx] += 1
        user_comm = int(user_communities[user_idx].item())
        user_comm_counts[item_idx, user_comm] += 1
        if item_user_sets is not None:
            item_user_sets[item_idx].add(user_idx)

        if trans_comm_counts is not None:
            history_items = parse_item_list(row.history_item_id)
            if history_items:
                prev_idx = int(history_items[-1]) - item_index_base
                if 0 <= prev_idx < item_num:
                    dst_cluster = int(item_communities[item_idx].item())
                    trans_comm_counts[prev_idx, dst_cluster] += 1

    return {
        "item_counts": item_counts,
        "user_comm_counts": user_comm_counts,
        "trans_comm_counts": trans_comm_counts,
        "item_user_sets": item_user_sets,
    }


def collect_sequence_statistics(
    sequence_path: Union[str, Path],
    item_num: int,
    item_index_base: int,
    user_communities: torch.Tensor,
    item_communities: Optional[torch.Tensor],
    expected_num_users: Optional[int] = None,
    build_item_user_sets: bool = False,
) -> Dict[str, Any]:
    if not Path(sequence_path).exists():
        raise FileNotFoundError(f"Missing sequence file: {sequence_path}")

    sequences = []
    with Path(sequence_path).open("r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            sequences.append(list(map(int, line.split())))

    if expected_num_users is not None and len(sequences) != expected_num_users:
        raise ValueError(
            f"user_cf_emb rows ({expected_num_users}) do not match sequence users ({len(sequences)})."
        )

    num_user_clusters = int(user_communities.max().item()) + 1
    num_item_clusters = int(item_communities.max().item()) + 1 if item_communities is not None else 0
    item_counts = torch.zeros(item_num, dtype=torch.float32)
    user_comm_counts = torch.zeros(item_num, num_user_clusters, dtype=torch.float32)
    trans_comm_counts = (
        torch.zeros(item_num, num_item_clusters, dtype=torch.float32)
        if item_communities is not None
        else None
    )
    item_user_sets = [set() for _ in range(item_num)] if build_item_user_sets else None

    for user_idx, sequence in enumerate(sequences):
        user_comm = int(user_communities[user_idx].item())
        normalized_sequence = [item_id - item_index_base for item_id in sequence]
        for item_idx in normalized_sequence:
            if not 0 <= item_idx < item_num:
                continue
            item_counts[item_idx] += 1
            user_comm_counts[item_idx, user_comm] += 1
            if item_user_sets is not None:
                item_user_sets[item_idx].add(user_idx)
        if trans_comm_counts is not None:
            for prev_idx, next_idx in zip(normalized_sequence[:-1], normalized_sequence[1:]):
                if not (0 <= prev_idx < item_num and 0 <= next_idx < item_num):
                    continue
                dst_cluster = int(item_communities[next_idx].item())
                trans_comm_counts[prev_idx, dst_cluster] += 1

    return {
        "item_counts": item_counts,
        "user_comm_counts": user_comm_counts,
        "trans_comm_counts": trans_comm_counts,
        "item_user_sets": item_user_sets,
    }


def load_user_id_map(path: Optional[Union[str, Path]]) -> Optional[Dict[str, int]]:
    if not path:
        return None
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Missing user id map: {file_path}")
    if file_path.suffix.lower() == ".json":
        import json

        payload = json.loads(file_path.read_text())
    else:
        payload = torch.load(file_path, map_location="cpu")
    if isinstance(payload, dict):
        return {str(key): int(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return {str(user_id): idx for idx, user_id in enumerate(payload)}
    raise TypeError(f"Unsupported user id map payload: {type(payload)!r}")


def parse_item_list(raw_value: Any) -> List[int]:
    if isinstance(raw_value, list):
        return [int(value) for value in raw_value]
    if pd.isna(raw_value):
        return []
    text = str(raw_value).strip()
    if not text:
        return []
    parsed = ast.literal_eval(text)
    if isinstance(parsed, (list, tuple)):
        return [int(value) for value in parsed]
    return [int(parsed)]


def run_kmeans(
    embeddings: torch.Tensor,
    n_clusters: int,
    n_iters: int = 25,
    seed: int = 2024,
    device: str = "cpu",
    chunk_size: int = 4096,
) -> torch.Tensor:
    points = embeddings.detach().float()
    n_points = points.size(0)
    if n_points == 0:
        raise ValueError("Cannot run k-means on an empty tensor.")
    n_clusters = max(1, min(int(n_clusters), n_points))
    target_device = resolve_device(device)
    work_points = points.to(target_device)
    generator = torch.Generator(device=target_device)
    generator.manual_seed(seed)
    initial_perm = torch.randperm(n_points, generator=generator, device=target_device)
    centers = work_points[initial_perm[:n_clusters]].clone()

    for _ in range(n_iters):
        labels = assign_clusters(work_points, centers, chunk_size=chunk_size)
        new_centers = torch.zeros_like(centers)
        counts = torch.zeros(n_clusters, device=target_device)
        new_centers.index_add_(0, labels, work_points)
        counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float32))

        empty_mask = counts == 0
        counts = counts.clamp_min(1.0).unsqueeze(1)
        new_centers = new_centers / counts
        if empty_mask.any():
            refill_perm = torch.randperm(n_points, generator=generator, device=target_device)
            new_centers[empty_mask] = work_points[refill_perm[: int(empty_mask.sum().item())]]

        if torch.allclose(new_centers, centers, atol=1e-4, rtol=1e-4):
            centers = new_centers
            break
        centers = new_centers

    return assign_clusters(work_points, centers, chunk_size=chunk_size).cpu()


def assign_clusters(points: torch.Tensor, centers: torch.Tensor, chunk_size: int = 4096) -> torch.Tensor:
    point_norm = (points ** 2).sum(dim=1, keepdim=True)
    center_norm = (centers ** 2).sum(dim=1).unsqueeze(0)
    labels = []
    for start in range(0, points.size(0), chunk_size):
        end = min(start + chunk_size, points.size(0))
        chunk = points[start:end]
        distances = point_norm[start:end] + center_norm - 2.0 * (chunk @ centers.t())
        labels.append(distances.argmin(dim=1))
    return torch.cat(labels, dim=0)


def semantic_topk_search(
    item_sem_emb: torch.Tensor,
    topk: int,
    chunk_size: int = 1024,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    target_device = resolve_device(device)
    normalized = F.normalize(item_sem_emb.float(), dim=-1).to(target_device)
    item_num = normalized.size(0)
    topk = max(1, min(topk, item_num - 1))
    all_indices = torch.zeros((item_num, topk), dtype=torch.long)
    all_scores = torch.zeros((item_num, topk), dtype=torch.float32)
    for start in range(0, item_num, chunk_size):
        end = min(start + chunk_size, item_num)
        similarities = normalized[start:end] @ normalized.t()
        row_indices = torch.arange(end - start, device=target_device)
        similarities[row_indices, torch.arange(start, end, device=target_device)] = -1e9
        scores, indices = torch.topk(similarities, k=topk, dim=-1, largest=True)
        all_indices[start:end] = indices.cpu()
        all_scores[start:end] = scores.cpu()
    return all_indices, all_scores


def normalize_values(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values
    min_value = values.min()
    max_value = values.max()
    if torch.isclose(max_value, min_value):
        return torch.ones_like(values)
    return (values - min_value) / (max_value - min_value)


def compute_item_reliability(item_counts: torch.Tensor, tau_n: float) -> torch.Tensor:
    return 1.0 - torch.exp(-item_counts.float() / tau_n)


def build_smoothed_distribution(counts: torch.Tensor, alpha: float) -> torch.Tensor:
    total_counts = counts.sum(dim=1, keepdim=True)
    global_prior = counts.sum(dim=0)
    if global_prior.sum() == 0:
        global_prior = torch.full_like(global_prior, 1.0 / global_prior.numel())
    else:
        global_prior = global_prior / global_prior.sum()
    return (counts + alpha * global_prior.unsqueeze(0)) / (total_counts + alpha)


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    if p.dim() == 1:
        p = p.unsqueeze(0)
    if q.dim() == 1:
        q = q.unsqueeze(0)
    if p.size(0) == 1 and q.size(0) > 1:
        p = p.expand_as(q)
    m = 0.5 * (p + q)
    log_p = torch.log(p.clamp_min(eps))
    log_q = torch.log(q.clamp_min(eps))
    log_m = torch.log(m.clamp_min(eps))
    kl_pm = (p * (log_p - log_m)).sum(dim=-1)
    kl_qm = (q * (log_q - log_m)).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)


def filter_no_cooccurrence_candidates(
    item_idx: int,
    candidate_indices: torch.Tensor,
    item_user_sets: Optional[List[set]],
) -> torch.Tensor:
    if item_user_sets is None:
        return torch.ones_like(candidate_indices, dtype=torch.bool)
    anchor_users = item_user_sets[item_idx]
    mask = []
    for candidate_idx in candidate_indices.tolist():
        candidate_users = item_user_sets[candidate_idx]
        mask.append(anchor_users.isdisjoint(candidate_users))
    return torch.tensor(mask, dtype=torch.bool)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def select_crds_negatives(
    pair_data: Dict[str, Any],
    anchor_item_ids: torch.Tensor,
    num_neg_per_anchor: int,
    neg_sampling: str = "random_top",
    random_top_pool_size: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    all_neg_ids = pair_data["neg_items"][anchor_item_ids]
    all_neg_weights = pair_data["neg_weights"][anchor_item_ids]
    reliability = pair_data["reliability"]
    semantic_stats = pair_data.get("pair_semantic_similarity", pair_data.get("neg_semantic_similarity"))
    cf_stats = pair_data.get("pair_cf_distance", pair_data.get("neg_cf_distance"))
    pair_reliability_stats = pair_data.get("pair_reliability")

    num_anchors, stored_neg_count = all_neg_ids.shape
    use_count = min(num_neg_per_anchor, stored_neg_count)
    selected_neg_ids = torch.full((num_anchors, use_count), -1, dtype=torch.long)
    selected_neg_weights = torch.zeros((num_anchors, use_count), dtype=torch.float32)
    selected_pair_reliability = torch.zeros((num_anchors, use_count), dtype=torch.float32)
    selected_semantic_stats = torch.zeros((num_anchors, use_count), dtype=torch.float32)
    selected_cf_stats = torch.zeros((num_anchors, use_count), dtype=torch.float32)

    for row_idx in range(num_anchors):
        valid_positions = torch.nonzero(
            (all_neg_ids[row_idx] >= 0) & (all_neg_weights[row_idx] > 0),
            as_tuple=False,
        ).squeeze(-1)
        if valid_positions.numel() == 0:
            continue

        if neg_sampling not in {"top", "random_top"}:
            raise ValueError(f"Unsupported CRDS negative sampling mode: {neg_sampling}")

        if neg_sampling == "top" or valid_positions.numel() <= use_count:
            chosen_positions = valid_positions[:use_count]
        else:
            pool_size = random_top_pool_size if random_top_pool_size is not None else valid_positions.numel()
            pool_size = max(use_count, min(int(pool_size), valid_positions.numel()))
            pool_positions = valid_positions[:pool_size]
            sampled_offsets = torch.randperm(pool_positions.numel())[:use_count]
            chosen_positions = torch.sort(pool_positions[sampled_offsets]).values
        chosen_count = chosen_positions.numel()
        chosen_neg_ids = all_neg_ids[row_idx, chosen_positions]
        chosen_neg_weights = all_neg_weights[row_idx, chosen_positions]

        selected_neg_ids[row_idx, :chosen_count] = chosen_neg_ids
        selected_neg_weights[row_idx, :chosen_count] = chosen_neg_weights

        if pair_reliability_stats is not None:
            selected_pair_reliability[row_idx, :chosen_count] = pair_reliability_stats[
                anchor_item_ids[row_idx], chosen_positions
            ]
        else:
            anchor_reliability = reliability[anchor_item_ids[row_idx]]
            selected_pair_reliability[row_idx, :chosen_count] = torch.minimum(
                anchor_reliability.expand(chosen_count),
                reliability[chosen_neg_ids],
            )
        if semantic_stats is not None:
            selected_semantic_stats[row_idx, :chosen_count] = semantic_stats[anchor_item_ids[row_idx], chosen_positions]
        if cf_stats is not None:
            selected_cf_stats[row_idx, :chosen_count] = cf_stats[anchor_item_ids[row_idx], chosen_positions]

    valid_mask = (selected_neg_ids >= 0) & (selected_neg_weights > 0)
    flat_valid_neg_ids = selected_neg_ids[valid_mask]
    if flat_valid_neg_ids.numel() > 0:
        unique_neg_ids, inverse_indices = torch.unique(
            flat_valid_neg_ids,
            sorted=True,
            return_inverse=True,
        )
        neg_inverse_indices = torch.full_like(selected_neg_ids, -1)
        neg_inverse_indices[valid_mask] = inverse_indices
    else:
        unique_neg_ids = torch.empty(0, dtype=torch.long)
        neg_inverse_indices = torch.full_like(selected_neg_ids, -1)

    return {
        "neg_ids": selected_neg_ids,
        "neg_weights": selected_neg_weights,
        "pair_reliability": selected_pair_reliability,
        "neg_semantic_similarity": selected_semantic_stats,
        "neg_cf_distance": selected_cf_stats,
        "valid_mask": valid_mask,
        "unique_neg_ids": unique_neg_ids,
        "neg_inverse_indices": neg_inverse_indices,
    }


def crds_hard_negative_loss(
    h1: torch.Tensor,
    h2: torch.Tensor,
    h_neg: torch.Tensor,
    neg_weights: torch.Tensor,
    tau: float,
    valid_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if valid_mask is None:
        valid_mask = neg_weights > 0
    valid_rows = valid_mask.any(dim=1)
    if not valid_rows.any():
        return h1.new_zeros(()), valid_rows

    h1 = F.normalize(h1, dim=-1)
    h2 = F.normalize(h2, dim=-1)
    h_neg = F.normalize(h_neg, dim=-1)

    pos_logits = (h1 * h2).sum(dim=-1, keepdim=True) / tau
    neg_logits = torch.einsum("bd,bkd->bk", h1, h_neg) / tau
    neg_logits = neg_logits.masked_fill(~valid_mask, -1e9)
    neg_logits = neg_logits + torch.where(
        valid_mask,
        torch.log(neg_weights.clamp_min(1e-8)),
        torch.zeros_like(neg_logits),
    )

    logits = torch.cat([pos_logits, neg_logits], dim=1)
    labels = torch.zeros(h1.size(0), dtype=torch.long, device=h1.device)
    loss = F.cross_entropy(logits[valid_rows], labels[valid_rows])
    return loss, valid_rows
