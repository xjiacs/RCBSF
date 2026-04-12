import os, json, sys, re, math
from typing import Dict, Any, List, Optional
from .llm import GenConfig, BaseModel
from .utils import read_text, safe_json_loads


def judge_contract(model: BaseModel, contract_text: str, debug_input: bool = False, debug_output: bool = False) -> Dict[str, Any]:
    prompt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", "judge_prompt.txt")
    if not os.path.exists(prompt_path):
        print(f"Error: Review prompt file not found: {prompt_path}", file=sys.stderr)
        return {"__raw__": "Prompt file not found"}

    prompt = read_text(prompt_path)
    prompt = prompt.format(contract_text=contract_text)
    messages = [
        {"role": "system", "content": "You are a rigorous contract quality review expert."},
        {"role": "user", "content": prompt}
    ]
    config = GenConfig(max_new_tokens=1200, temperature=0.3)
    raw = model.generate(messages, config)

    cleaned_raw = raw.strip()
    if cleaned_raw.startswith("```json"):
        cleaned_raw = cleaned_raw[len("```json"):].strip()
    if cleaned_raw.endswith("```"):
        cleaned_raw = cleaned_raw[:-len("```")].strip()
    cleaned_raw = cleaned_raw.rstrip(",").strip()

    if debug_input or debug_output:
        print("\n--- [DEBUG] judge_contract ---")
    if debug_input:
        print(f"INPUT MESSAGES:\n{json.dumps(messages, ensure_ascii=False, indent=2)}")
    if debug_output:
        print(f"\nRAW OUTPUT:\n{raw}")
        print(f"\nCLEANED OUTPUT (for JSON parse):\n{cleaned_raw}")
    if debug_input or debug_output:
        print("------------------------------\n")

    data = safe_json_loads(cleaned_raw) or {}
    data["__raw__"] = raw
    return data


def judge_risk_resolution(model: BaseModel, contract_text: str, gold_high_risks: List[Dict[str, Any]],
                          debug: bool = False) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    sys_prompt = "You are a rigorous and objective legal reviewer, skilled at determining whether risks have been substantively resolved by comparing the risks with the revised contract. Your output must be a JSON array."
    tmpl = (
        "Full contract text as follows:\n{contract}\n\n"
        "Now judge ONLY the following [Golden High-Risk] item:\n"
        "- Risk Category: {cat}\n- Relevant Text Segment: {loc}\n- Golden Revision Suggestion: {sug}\n\n"
        "Question: Has the final contract substantively resolved this risk? Output ONLY a JSON object:"
        "{{\"risk_id\": {rid}, \"resolved\": <true/false>, \"confidence\": 0~1, \"rationale\": \"...\"}}"
    )
    config = GenConfig(max_new_tokens=1500, temperature=0.2)

    for item in gold_high_risks:
        msg = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": tmpl.format(
                contract=contract_text,
                cat=item.get("risk_category", ""),
                loc=item.get("risk_location_text", ""),
                sug=item.get("golden_revision_suggestion", ""),
                rid=item.get("risk_id", 0)
            )}
        ]
        raw = model.generate(msg, config)

        if debug:
            print(f"\n--- [DEBUG] judge_risk_resolution (ID: {item.get('risk_id', 0)}) ---")
            print(f"INPUT MESSAGES:\n{json.dumps(msg, ensure_ascii=False, indent=2)}")
            print(f"\nRAW OUTPUT:\n{raw}")
            print("------------------------------\n")

        data = safe_json_loads(raw)
        if isinstance(data, dict):
            results.append({
                "risk_id": item.get("risk_id", 0),
                "resolved": bool(data.get("resolved", False)),
                "confidence": float(data.get("confidence", 0.0)),
                "rationale": data.get("rationale", ""),
                "__raw__": raw
            })
        else:
            results.append({
                "risk_id": item.get("risk_id", 0),
                "resolved": False,
                "confidence": 0.0,
                "rationale": "Model output is unstructured; default to unresolved.",
                "__raw__": raw
            })
    return results


def judge_resolution_from_seed_categories(model: BaseModel, contract_text: str, seed_items: List[Dict[str, Any]],
                                          debug_input: bool = False, debug_output: bool = False) -> List[Dict[str, Any]]:
    def _normalize_text(s: str) -> str:
        if s is None:
            return ""
        s = s.replace("\r\n", "\n").replace("\r", "\n")
        return s.lstrip("\ufeff")

    def _try_json_fix(s: str) -> Optional[str]:
        if not s:
            return None
        cleaned = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F]', '', s)
        l = cleaned.find('{')
        r = cleaned.rfind('}')
        if l == -1 or r == -1 or r <= l:
            return None
        candidate = cleaned[l:r + 1]
        candidate = re.sub(r',\s*}', '}', candidate)
        candidate = re.sub(r',\s*]', ']', candidate)
        candidate = re.sub(r'\bTrue\b', 'true', candidate)
        candidate = re.sub(r'\bFalse\b', 'false', candidate)
        candidate = re.sub(r'\bNone\b', 'null', candidate)
        return candidate

    results: List[Dict[str, Any]] = []

    sys_prompt = (
        "You are a rigorous and neutral contract risk control reviewer. Based solely on the literal content of the contract text, "
        "review whether the final contract has substantively mitigated/eliminated the risk in accordance with the given [Risk Basis]. "
        "Speculation is strictly prohibited. Output ONLY a JSON object containing only the specified fields. "
        "\nJudgment Criteria (Mandatory): If the contract contains clauses directly related to the risk, and the clauses include "
        "verifiable elements (e.g., specific numerical values, clear time limits, boundaries/processes/remedies, caps or carve-outs, "
        "audit frequency, etc.), and there are no directly conflicting statements, mark it as resolved; otherwise, mark it as unresolved."
    )

    tmpl = (
        "This is the full text of the final contract:\n{contract}\n\n"
        "The following is a [Risk Basis] (from seed dynamic risk categories):\n"
        "- Category: {cat}\n"
        "- Basis/Details: {basis}\n\n"
        "Judgment Criteria: If the contract has substantively mitigated or eliminated the risk through clause setting/definition/boundary/remedy, "
        "etc., with verifiable elements (numerical values/time limits/processes/remedies/caps or carve-outs, etc.), and there are no directly "
        "conflicting clauses, mark it as unresolved.\n"
        "Output ONLY a JSON object containing ONLY the following keys:\n"
        "{{\"seed_id\": {sid}, \"category\": \"{cat}\", \"resolved\": true/false, \"confidence\": 0~1, \"rationale\": \"brief rationale (no more than 80 words)\"}}"
    )

    config = GenConfig(max_new_tokens=1500, temperature=0.2)

    contract_text = _normalize_text(contract_text)

    for item in seed_items:
        sid = int(item.get("id", 0))
        cat = str(item.get("category", "")).strip() or f"seed_{sid}"
        basis = str(item.get("basis", "")).strip()

        msg = [
            {"role": "system", "content": sys_prompt},
            {"role": "user",
             "content": tmpl.format(contract=contract_text,
                                    cat=cat.replace('{', '').replace('}', ''),
                                    basis=basis,
                                    sid=sid)}
        ]

        raw = model.generate(msg, config)

        if debug_input or debug_output:
            print(f"\n--- [DEBUG] judge_resolution_from_seed_categories (ID: {sid}) ---")
        if debug_input:
            print(f"INPUT MESSAGES:\n{json.dumps(msg, ensure_ascii=False, indent=2)}")
        if debug_output:
            print(f"\nRAW OUTPUT:\n{raw}")
        if debug_input or debug_output:
            print("------------------------------\n")

        data = safe_json_loads(raw)

        if not isinstance(data, dict):
            fixed = _try_json_fix(raw)
            if fixed:
                data = safe_json_loads(fixed)

        if not isinstance(data, dict):
            repair_messages = [
                {"role": "system", "content": "You are a strict JSON fixer. Output ONLY a valid JSON object; no additional text is allowed."},
                {"role": "user", "content":
                    ("Convert the following non-standard output into a valid JSON object, retaining ONLY the following keys:"
                     f"{{\"seed_id\": {sid}, \"category\": \"{cat}\", \"resolved\": true/false, "
                     "\"confidence\": 0~1, \"rationale\": \"brief rationale (no more than 80 words)\"}}\n\n"
                     f"Original output:\n{raw}")
                 }
            ]
            raw_repair = model.generate(repair_messages, GenConfig(max_new_tokens=400, temperature=0.1))
            if debug_output:
                print("[DEBUG] REPAIR RAW OUTPUT:\n", raw_repair)
            data = safe_json_loads(raw_repair)
            if not isinstance(data, dict):
                fixed2 = _try_json_fix(raw_repair)
                if fixed2:
                    data = safe_json_loads(fixed2)

        if isinstance(data, dict):
            resolved = bool(data.get("resolved", False))
            try:
                conf = float(data.get("confidence", 0.0))
            except Exception:
                conf = 0.0
            conf = 0.0 if math.isnan(conf) else max(0.0, min(1.0, conf))
            rationale = str(data.get("rationale", ""))

            results.append({
                "seed_id": sid,
                "category": cat,
                "resolved": resolved,
                "confidence": conf,
                "rationale": rationale,
                "__raw__": raw
            })
        else:
            results.append({
                "seed_id": sid,
                "category": cat,
                "resolved": False,
                "confidence": 0.0,
                "rationale": "Model output is unstructured; default to unresolved.",
                "__raw__": raw
            })

    return results