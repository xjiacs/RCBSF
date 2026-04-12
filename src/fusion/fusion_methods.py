from typing import Dict, List, Tuple

def score_map_to_float(q: Dict[str, str], weights=(0.4, 0.2, 0.2, 0.2)) -> float:
    m = {"A": 1.0, "B": 0.7, "C": 0.4}
    return sum(m.get(q.get(k, "C"), 0.4) * w for k, w in zip(["Q1", "Q2", "Q3", "Q4"], weights))

def normalize(xs: List[float]) -> List[float]:
    s = sum(xs) or 1.0
    return [x / s for x in xs]

def fuse_weighted_sum(
    outer_q: Dict[str, Dict[str, str]],
    inner_q: Dict[str, Dict[str, str]],
    weights=(0.4, 0.2, 0.2, 0.2),
) -> Dict[str, float]:
    out = {}
    for k in set(list(outer_q.keys()) + list(inner_q.keys())):
        so = score_map_to_float(outer_q.get(k, {}), weights)
        si = score_map_to_float(inner_q.get(k, {}), weights)
        out[k] = 0.6 * so + 0.4 * si
    return out

def fuse_poe(
    outer_q: Dict[str, Dict[str, str]],
    inner_q: Dict[str, Dict[str, str]],
    weights=(0.4, 0.2, 0.2, 0.2),
) -> Dict[str, float]:
    out = {}
    for k in set(list(outer_q.keys()) + list(inner_q.keys())):
        so = max(1e-3, score_map_to_float(outer_q.get(k, {}), weights))
        si = max(1e-3, score_map_to_float(inner_q.get(k, {}), weights))
        out[k] = so * si
    mx = max(out.values() or [1.0])
    for k in out:
        out[k] /= mx
    return out

def fuse_topk_with_budget(scores: Dict[str, float], k: int, budget_tokens: int) -> List[str]:
    return sorted(scores, key=lambda x: scores[x], reverse=True)[:k]

def fuse_moe(
    outers: Dict[str, Dict[str, str]],
    inners: Dict[str, Dict[str, str]],
    weights=(0.4, 0.2, 0.2, 0.2),
) -> Dict[str, float]:
    out = {}
    for k in set(list(outers.keys()) + list(inners.keys())):
        so = score_map_to_float(outers.get(k, {}), weights)
        si = score_map_to_float(inners.get(k, {}), weights)
        out[k] = 0.5 * so + 0.5 * si
    return out

def fuse_signals(
    method: str,
    categories: List[str],
    outer_q: Dict[str, Dict[str, str]],
    inner_q: Dict[str, Dict[str, str]],
    q_weights=(0.4, 0.2, 0.2, 0.2),
    contract_budget: int = 1800,
    audit_budget: int = 900,
) -> Tuple[Dict[str, float], List[str], str]:
    if method == "poe":
        scores = fuse_poe(outer_q, inner_q, q_weights)
    elif method == "topk_with_budget":
        scores = fuse_weighted_sum(outer_q, inner_q, q_weights)
    elif method == "moe":
        scores = fuse_moe(outer_q, inner_q, q_weights)
    else:
        scores = fuse_weighted_sum(outer_q, inner_q, q_weights)

    budget_tokens = (contract_budget or 1800) + (audit_budget or 900)
    per = 240
    est_k = max(8, min(len(categories), budget_tokens // per))
    topk = fuse_topk_with_budget(scores, est_k, budget_tokens)
    hint = (
        f"[Budget Constraint] Contract ≤ {contract_budget}, "
        f"Audit ≤ {audit_budget} tokens; "
        f"Top-{len(topk)}: {topk}."
    )
    return scores, topk, hint
