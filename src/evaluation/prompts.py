
from typing import List, Dict

def build_original_risk_messages(contract_text: str) -> List[Dict[str, str]]:
    """
    Ask the model to produce 8–12 risk SEED items from an ORIGINAL contract.
    Each item must include a short 'category' and a concrete 'basis' excerpt/rationale.
    Strict JSON.
    """
    system = (
        "You are a senior legal risk officer. Read the ORIGINAL contract and return 8-12 high-signal risk  items. "
        "Exclude items about contract validity, parties' identities, dates/signatures/seals/IDs'."
    )
    user = f"""
Return STRICTLY a single JSON object:
{{
  "seed_items": [
    {{"id": 0, "category": "<8-15 word risk category>", "basis": "<Detailed specific reasons or detailed clause excerpts>"}}
    # ... 7–11 more
  ]
}}

Rules:
- 8–12 items
- English only
- 'basis' should cite the risk signal or clause rationale; Keep detailed.
- Do NOT include validity/identity/date/signature/seal/ID/termination-ambiguity items.
- Categories should be specific.

ORIGINAL CONTRACT:
<<<CONTRACT_START>>>
{contract_text}
<<<CONTRACT_END>>>
""".strip()
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
