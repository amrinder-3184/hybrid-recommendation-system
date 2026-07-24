
import numpy as np


def precision_at_k(actual: list[list[str]], predicted: list[list[str]], k: int = 10) -> float:
    precisions = []
    for a, p in zip(actual, predicted):
        p_k = p[:k]
        if not p_k:
            precisions.append(0.0)
            continue
        num_hits = len(set(a) & set(p_k))
        precisions.append(num_hits / k)
    return float(np.mean(precisions)) if precisions else 0.0

def recall_at_k(actual: list[list[str]], predicted: list[list[str]], k: int = 10) -> float:
    recalls = []
    for a, p in zip(actual, predicted):
        p_k = p[:k]
        if not a:
            recalls.append(0.0)
            continue
        num_hits = len(set(a) & set(p_k))
        recalls.append(num_hits / len(a))
    return float(np.mean(recalls)) if recalls else 0.0

def map_at_k(actual: list[list[str]], predicted: list[list[str]], k: int = 10) -> float:
    """Computes Mean Average Precision at K"""
    maps = []
    for a, p in zip(actual, predicted):
        if not a:
            maps.append(0.0)
            continue
        p_k = p[:k]
        hits = 0
        sum_precisions = 0.0
        for i, item in enumerate(p_k):
            if item in a:
                hits += 1
                sum_precisions += hits / (i + 1.0)
        maps.append(sum_precisions / min(len(a), k))
    return float(np.mean(maps)) if maps else 0.0

def ndcg_at_k(actual: list[list[str]], predicted: list[list[str]], k: int = 10) -> float:
    """Computes Normalized Discounted Cumulative Gain at K"""
    ndcgs = []
    for a, p in zip(actual, predicted):
        if not a:
            ndcgs.append(0.0)
            continue
        p_k = p[:k]
        dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(p_k) if item in a)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(a), k)))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(ndcgs)) if ndcgs else 0.0

class Evaluator:
    def __init__(self, k_values: list[int] = [10]):
        self.k_values = k_values

    def evaluate(self, actual: list[list[str]], predicted: list[list[str]]) -> dict[str, float]:
        metrics = {}
        for k in self.k_values:
            metrics[f"Precision@{k}"] = precision_at_k(actual, predicted, k)
            metrics[f"Recall@{k}"] = recall_at_k(actual, predicted, k)
            metrics[f"MAP@{k}"] = map_at_k(actual, predicted, k)
            metrics[f"NDCG@{k}"] = ndcg_at_k(actual, predicted, k)
        return metrics
