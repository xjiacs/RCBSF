from __future__ import annotations
import os
import json
from typing import List, Dict, Any
from dataclasses import dataclass

from .llm import LocalQwen, GenConfig
from .utils import safe_json_loads

RISK_EDITOR_PROMPT = """You are a strictly logical contract risk auditor. Please review the contract text below and extract risk points.
You must analyze from 5 dimensions for each risk:
1. **category**: Specific risk classification. At least ten words, as detailed as possible.
2. **location**: Where this risk appears.
3. **evidence**: Original text quote supporting this risk.
4. **issue**: Specific description of the defect, ambiguity, or unfairness.
5. **suggestion**: Actionable advice on how to modify or add clause text.

Requirements:
- Output must be a JSON object containing a list named "risk_reviews".
- Each item must include: "category", "location", "evidence", "issue", "suggestion".
- If the text is very short, identify at least 8–15 potential risks or missing elements.
- Always analyze strictly, even for short texts or definitions.

Output Structure:
{
  "risk_reviews": [
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
Only output JSON.
"""

_Q_TEMPLATE_FALLBACK = """The following is a section of contract clauses..."""

def _load_q_prompt_template() -> str:
    candidates = [
        os.getenv("RISK_Q_PROMPT_TEMPLATE") or "",
        "/public/home/chengtongtong/xsj/process/out_inner/rcbse_llm_leader_1_gpt/prompts/risk_category_prompt_template.txt",
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

def _grade_with_template(
    model: LocalQwen,
    clause_text: str,
    q_template: str
) -> Dict[str, str]:
    prompt = q_template.replace("{{CLAUSE_TEXT}}", clause_text[:8000])
    msgs = [
        {
            "role": "system",
            "content": "Only output JSON with keys Q1, Q2, Q3, Q4 and values A, B, or C."
        },
        {"role": "user", "content": prompt},
    ]
    raw = model.generate(
        msgs,
        gen_config=GenConfig(
            max_new_tokens=600,
            temperature=0.1,
            do_sample=False,
        ),
    )
    d = (
        safe_json_loads(raw)
        or safe_json_loads(raw.strip("```json").strip("```"))
        or {}
    )
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

@dataclass
class RiskEditorAgent:
    model: LocalQwen

    def audit(
        self,
        contract_text: str,
        categories: List[str],
    ) -> Dict[str, Any]:
        msgs = [
            {
                "role": "system",
                "content": "Only output JSON with fields: category, location, evidence, issue, suggestion.",
            },
            {
                "role": "user",
                "content": RISK_EDITOR_PROMPT.replace(
                    "{CONTRACT}", (contract_text or "")[:16000]
                ),
            },
        ]
        raw = self.model.generate(
            msgs,
            gen_config=GenConfig(
                max_new_tokens=1500,
                temperature=0.2,
                do_sample=False,
            ),
        )
        ans = (
            safe_json_loads(raw)
            or safe_json_loads(raw.strip("```json").strip("```"))
            or {}
        )
        reviews = ans.get("risk_reviews") if isinstance(ans, dict) else None
        if not isinstance(reviews, list):
            reviews = []

        q_template = _load_q_prompt_template()
        norm_reviews = []
        qmap: Dict[str, Dict[str, Any]] = {}

        for item in reviews:
            cat = str(item.get("category", "")).strip()
            if not cat:
                continue

            clause_context = (
                f"Clause: {item.get('evidence', '')}\n"
                f"Issue: {item.get('issue', '')}"
            )
            if len(clause_context) < 10:
                clause_context = (contract_text or "")[:2000]

            q = _grade_with_template(
                self.model,
                clause_context,
                q_template,
            )

            entry = {
                "category": cat,
                "location": item.get("location", "Unknown"),
                "evidence": item.get("evidence", ""),
                "issue": item.get("issue", ""),
                "suggestion": item.get("suggestion", ""),
                "q": q,
            }
            norm_reviews.append(entry)

            qmap[cat] = {
                "Q1": q["Q1"],
                "Q2": q["Q2"],
                "Q3": q["Q3"],
                "Q4": q["Q4"],
                "location": entry["location"],
                "evidence": entry["evidence"],
                "issue": entry["issue"],
                "suggestion": entry["suggestion"],
            }

        return {
            "risk_reviews": norm_reviews,
            "qmap": qmap,
            "__raw__": raw,
        }

    def to_drafter_prompt(
        self,
        categories: List[str],
        qmap: Dict[str, Dict[str, Any]],
    ) -> str:
        lines = [
            "Please revise according to the following audited risk points "
            "(detailed five-dimension analysis):"
        ]

        found_any = False
        for c in categories:
            if c in qmap:
                found_any = True
                data = qmap[c]
                block = f"""
### Risk Category: {c}
- Location: {data.get('location', 'Global')}
- Evidence: "{data.get('evidence', 'See text')}"
- Issue: {data.get('issue', 'General risk')}
- Suggestion: {data.get('suggestion', 'Revise to mitigate risk')}
- Urgency Score: Q={{Q1:{data['Q1']}, Q2:{data['Q2']}, Q3:{data['Q3']}, Q4:{data['Q4']}}}
"""
                lines.append(block)

        if not found_any:
            lines.append(
                "No specific inner-layer risks were identified. "
                "Please review broadly based on outer constraints."
            )

        lines.append(
            "Note: Strictly implement the suggestion for each risk above."
        )
        return "\n".join(lines)
