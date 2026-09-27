from __future__ import annotations

import json
import math
from collections.abc import Hashable, Iterable, Mapping
from statistics import fmean
from typing import TypeAlias

UserId: TypeAlias = Hashable
ItemId: TypeAlias = Hashable

Recommendations: TypeAlias = Mapping[UserId, Iterable[ItemId]]
GroundTruth: TypeAlias = Mapping[UserId, Iterable[ItemId]]

def _validate_k(k: int) -> None:
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be an integer")

def _unique_top_k(items: Iterable[ItemId], k: int) -> list[ItemId]:
    result: list[ItemId] = []
    seen: set[ItemId] = set()

    for item in items:
        if item in seen:
            continue

        result.append(item)
        seen.add(item)

        if len(result) == k:
            break

    return result

def _prepare_ground_truth(ground_truth: GroundTruth) -> dict[UserId, set[ItemId]]:
    prepared: dict[UserId, set[ItemId]] = {}

    for user_id, items in ground_truth.items():
        relevant_items = set(items)
        if relevant_items:
            prepared[user_id] = relevant_items

    return prepared

def recall_at_k(recommendations: Recommendations, ground_truth: GroundTruth, k: int) -> float:
    _validate_k(k)
    targets = _prepare_ground_truth(ground_truth)

    if not targets:
        return 0.0

    user_recalls: list[float] = []
    for user_id, relevant_items in targets.items():
        top_k = _unique_top_k(
            recommendations.get(user_id, []),
            k
        )

        hits = len(set(top_k) & relevant_items)
        user_recalls.append(hits / len(relevant_items))

    return float(fmean(user_recalls))

def hit_rate_at_k(recommendations: Recommendations, ground_truth: GroundTruth, k: int) -> float:
    _validate_k(k)
    targets = _prepare_ground_truth(ground_truth)
    
    if not targets:
        return 0.0

    user_hits: list[float] = []
    for user_id, relevant_items in targets.items():
        top_k = _unique_top_k(
            recommendations.get(user_id, []),
            k
        )

        hit_cnt = bool(set(top_k) & relevant_items)
        user_hits.append(float(hit_cnt))

    return float(fmean(user_hits))


def ndcg_at_k(recommendations: Recommendations, ground_truth: GroundTruth, k: int) -> float:
    _validate_k(k)
    targets = _prepare_ground_truth(ground_truth)
        
    if not targets:
        return 0.0
    
    user_ncdgs: list[float] = []
    for user_id, relevant_items in targets.items():
        top_k = _unique_top_k(
            recommendations.get(user_id, []),
            k
        )

        dcg = 0.0
        for rank, item_id in enumerate(top_k, start=1):
            if item_id in relevant_items:
                dcg += 1.0 / math.log2(rank + 1)

            ideal_length = min(len(relevant_items), k)

            idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_length + 1))

            user_ncdgs.append(dcg / idcg if idcg > 0 else 0.0)

    return float(fmean(user_ncdgs))

def coverage_at_k(recommendations: Recommendations, catalog_size: int, k: int) -> float:
    _validate_k(k)
    if catalog_size <= 0:
        raise ValueError("catalog_size must be greater than zero.")

    recommended_items: set[ItemId] = set()

    for items in recommendations.values():
        recommended_items.update(_unique_top_k(items, k))

    return len(recommended_items) / catalog_size

def evaluate_ranking(recommendations: Recommendations, ground_truth:GroundTruth, catalog_size: int) -> dict[str, float | int]:
    prepared_targets = _prepare_ground_truth(ground_truth)

    metrics: dict[str, float | int] = {
        "evaluated_users": len(prepared_targets),
        "recall@20": recall_at_k(recommendations, ground_truth, k=20),
        "recall@50": recall_at_k(recommendations, ground_truth, k=50),
        "hitrate@20": hit_rate_at_k(recommendations, ground_truth, k=20),
        "ndcg@10": ndcg_at_k(recommendations, ground_truth, k=10),
        "coverage@50": coverage_at_k(recommendations, catalog_size=catalog_size, k=50)
    }

    return metrics

def print_metrics(
    metrics: Mapping[str, float | int],
) -> None:

    print("\nEvaluation Metrics")

    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{name:<18} {value:.6f}")
        else:
            print(f"{name:<18} {value:,}")


def save_metrics(
    metrics: Mapping[str, float | int | str],
    output_path: str,
) -> None:

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            dict(metrics),
            file,
            ensure_ascii=True,
            indent=2,
        )