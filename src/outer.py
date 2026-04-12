
from __future__ import annotations
import os
import json
import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from rcbsf.llm import LocalQwen, GenConfig
from rcbsf.utils import safe_json_loads


EVIDENCE_PROMPT = """You are a strictly logical contract risk auditor. Please review the contract text below and extract risk points.
You must analyze from 5 dimensions for each risk:
1. **category**: Specific risk classification. (At least ten words or more, describe the risk category as detailed as possible)
2. **location**: Where this risk appears (e.g., "Section 1.2", "Preamble", or "Whole Document").
3. **evidence**: Original text quote supporting this risk. If missing (e.g., missing clause), state "Missing clause".
4. **issue**: specific description of what is wrong (the defect, ambiguity, or unfairness).
5. **suggestion**: Actionable advice on how to modify/add clause text.

Requirements:
- Even for short texts or definitions, analyze strictly (e.g., are definitions vague? is the scope clear?).
- **Output must be a JSON object** containing a list "risk_categories".
- Each item in the list must have keys: "category", "location", "evidence", "issue", "suggestion".
- If the text is very short, try to identify at least 8-15 potential risks/missing elements.

Output Structure:
{
  "risk_categories": [
    {
      "category": "...",
      "location": "...",
      "evidence": "...",
      "issue": "...",
      "suggestion": "..."
    }
  ]
}

Contract text:
<<<CONTRACT>>>
{CONTRACT}
<<<END>>>
Please only output JSON."""

@dataclass
class OuterDecisionV2:
    categories: List[str]
    details: Dict[str, Any]
    weights: List[float]
    hint_text: str

def _dedup_keep_order(items: List[Dict[str, Any]], max_cats: int) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for it in items:
        cat = str(it.get("category", "")).strip()
        # Ensure we have the 5 dimensions, fill defaults if missing
        it["location"] = str(it.get("location", "Global"))
        it["evidence"] = str(it.get("evidence", "Context implied"))
        it["issue"] = str(it.get("issue", "Potential risk detected"))
        it["suggestion"] = str(it.get("suggestion", "Review and revise"))
        
        if not cat: 
            continue
        if cat not in seen:
            seen.add(cat)
            out.append(it)
        if len(out) >= max_cats:
            break
    return out

def extract_categories_v2(model: LocalQwen, contract_text: str, max_cats: int = 12) -> List[Dict[str, Any]]:
    msgs = [
        {"role": "system", "content": "You are a risk auditor. Output JSON with fields: category, location, evidence, issue, suggestion."},
        {"role": "user", "content": EVIDENCE_PROMPT.replace("{CONTRACT}", (contract_text or "")[:16000])}
    ]
    raw = model.generate(msgs, gen_config=GenConfig(max_new_tokens=1500, temperature=0.3, do_sample=True))
    data = safe_json_loads(raw) or safe_json_loads(raw.strip("```json").strip("```")) or {}
    
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("risk_categories") or []
    else:
        items = []
        
    if not isinstance(items, list):
        items = []
        
    return _dedup_keep_order(items, max_cats=max_cats)


_Q_TEMPLATE_FALLBACK = """The following is a section of contract clauses (or a risk point). Please select option A/B/C for each of the 4 dimensions based only on this content, and output your judgment result in JSON format.
【Start of contract clauses】
{{CLAUSE_TEXT}}
【End of contract clauses】
Please output:
{"Q1":"A|B|C","Q2":"A|B|C","Q3":"A|B|C","Q4":"A|B|C"}"""

def _load_q_prompt_template() -> str:
    candidates = [
        os.getenv("RISK_Q_PROMPT_TEMPLATE") or "",
        "/public/home/chengtongtong/xsj/process/out_inner/rcbsf_llm_leader_1_gpt/prompts/risk_category_prompt_template.txt",
        "./risk_category_prompt_template.txt",
        "/mnt/data/risk_category_prompt_template.txt",
    ]
    for p in candidates:
        if p and os.path.exists(p):
            try:
                return open(p, "r", encoding="utf-8").read()
            except Exception:
                pass
    return _Q_TEMPLATE_FALLBACK

def _grade_with_template(model: LocalQwen, clause_text: str, q_template: str) -> Dict[str, str]:
    prompt = q_template.replace("{{CLAUSE_TEXT}}", clause_text[:8000])
    msgs = [
        {"role": "system", "content": "You must only output JSON with keys Q1~Q4 and values A/B/C."},
        {"role": "user", "content": prompt}
    ]
    raw = model.generate(msgs, gen_config=GenConfig(max_new_tokens=600, temperature=0.1, do_sample=False))
    d = safe_json_loads(raw) or safe_json_loads(raw.strip("```json").strip("```")) or {}
    q = {
        "Q1": str(d.get("Q1", "B")).strip().upper(),
        "Q2": str(d.get("Q2", "B")).strip().upper(),
        "Q3": str(d.get("Q3", "B")).strip().upper(),
        "Q4": str(d.get("Q4", "B")).strip().upper(),
    }
    for k in ("Q1", "Q2", "Q3", "Q4"):
        if q[k] not in ("A", "B", "C"):
            q[k] = "B"
    return q

def _abc_to_num(x: str) -> float:
    m = {"A": 1.0, "B": 0.6, "C": 0.25}
    return m.get(str(x).upper(), 0.5)

def _aggregate_score(q: Dict[str, str], q_weights: Tuple[float, float, float, float]) -> float:
    w1, w2, w3, w4 = q_weights
    return (_abc_to_num(q["Q1"])*w1 + _abc_to_num(q["Q2"])*w2 +
            _abc_to_num(q["Q3"])*w3 + _abc_to_num(q["Q4"])*w4)

def _softmax(xs: List[float], temp: float = 1.0) -> List[float]:
    if not xs:
        return []
    m = max(xs)
    exps = [math.exp((x - m)/max(1e-6, temp)) for x in xs]
    s = sum(exps) or 1.0
    return [v/s for v in exps]


def build_detailed_hint(items: List[Dict[str, Any]], contract_budget: Optional[int], audit_budget: Optional[int]) -> str:

    lines = ["【Outer Layer Analysis - Detailed Risk Report】"]
    for idx, it in enumerate(items):
        lines.append(f"Risk #{idx+1} [{it.get('category')}]")
        lines.append(f"  - Location: {it.get('location')}")
        lines.append(f"  - Evidence: {it.get('evidence')}")
        lines.append(f"  - Issue: {it.get('issue')}")
        lines.append(f"  - Suggestion: {it.get('suggestion')}")
    
    lines.append("\n【Resource Constraints】")
    if contract_budget: lines.append(f"- Target Tokens: {contract_budget}")
    lines.append("General Principle: Use the suggestions above to rewrite the contract executable.")
    return "\n".join(lines)

def decide_with_budget(model: LocalQwen, contract_text: str,
                          q_weights: Tuple[float, float, float, float] = (0.4, 0.2, 0.2, 0.2),
                          softmax_temp: float = 1.0,
                          contract_budget: Optional[int] = None, audit_budget: Optional[int] = None) -> OuterDecisionV2:
    items = extract_categories_v2(model, contract_text)
    cats = [it["category"] for it in items]

    q_template = _load_q_prompt_template()
    scored = []
    for it in items:
        clause_context = f"Evidence: {it.get('evidence')}\nIssue: {it.get('issue')}\nOriginal Context: {(contract_text or '')[:800]}"
        q = _grade_with_template(model, clause_context, q_template)

        scored.append({
            "category": it["category"],
            "location": it["location"],
            "evidence": it["evidence"],
            "issue": it["issue"],
            "suggestion": it["suggestion"],
            "q": q
        })

    scores = [_aggregate_score(s["q"], q_weights) for s in scored]
    weights = _softmax(scores, temp=max(1e-6, softmax_temp))

    hint_text = build_detailed_hint(items, contract_budget, audit_budget)

    return OuterDecisionV2(
        categories=cats,
        details={"risk_categories": scored, "evidence": items}, # 'scored' now contains full 5-dim info
        weights=weights,
        hint_text=hint_text
    )