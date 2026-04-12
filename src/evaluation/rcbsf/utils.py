import os, json, re, random, time, math
from typing import Any, Dict, List, Optional, Tuple
from rapidfuzz import fuzz

def set_seed(seed: int = 42):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def write_text(path: str, text: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def safe_json_loads(s: str) -> Optional[dict]:
    try:
        return json.loads(s)
    except Exception:
        try:
            start = s.find("{")
            end = s.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(s[start:end+1])
        except Exception:
            pass
    return None

def extract_first_json_block(s: str) -> Optional[str]:
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start:end+1]
    return None

def fuzzy_contains(a: str, b: str, threshold: int = 80) -> bool:
    a = a or ""
    b = b or ""
    if not a or not b:
        return False
    return fuzz.partial_ratio(a, b) >= threshold

def count_tokens_approx(text: str) -> int:
    return max(1, int(len(text) / 4))

def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

COMMON_RISK_CATEGORIES = []
