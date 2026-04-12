
import os, json, re, sys
from typing import Any, Dict, List, Optional
CUR = os.path.abspath(os.path.dirname(__file__))
PARENT = os.path.abspath(os.path.join(CUR, ".."))
if PARENT not in sys.path:
    sys.path.append(PARENT)
if os.path.join(PARENT, "fixed_risk_eval_OR", "fixed_risk_eval_OR") not in sys.path:
    sys.path.append(os.path.join(PARENT, "fixed_risk_eval_OR", "fixed_risk_eval_OR"))
def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)
def parse_json_loose(maybe_json: str) -> Optional[dict]:
    try:
        return json.loads(maybe_json)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", maybe_json)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None

def filter_seed_items(raw: List[dict]) -> List[dict]:
    out = []
    seen = set()
    for i, it in enumerate(raw or []):
        cat = str(it.get("category","")).strip()
        basis = str(it.get("basis","")).strip()
        if not cat or not basis:
            continue
        key = cat.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append({"id": i, "category": cat, "basis": basis})
    return out

def read_json(p: str) -> Any:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(p: str, obj: Any):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
