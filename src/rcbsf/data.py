import os, json, random, itertools, hashlib
from typing import List, Dict, Any, Optional
from .llm import LocalQwen, GenConfig
from .utils import (
    read_text,
    write_text,
    ensure_dir,
    safe_json_loads,
    extract_first_json_block,
    set_seed,
)

PREFIX_MAP_FILE = os.path.join(
    os.path.dirname(__file__), "..", "prompts", "prefix_map.txt"
)


def _format_risk_list(risks: List[str]) -> str:
    lines = []
    for i, r in enumerate(risks, 1):
        lines.append(f"    {i}. {r}")
    return "\n".join(lines)


def _derive_risk_categories(
    teacher: LocalQwen,
    contract_type: str,
    scenario: str,
    role: str,
    max_k: int = 6,
) -> List[str]:
    prompt = (
        "You are a senior legal expert.\n"
        "First, outline the main clause points of this contract based on the following information "
        "(point-form summary within 10 lines), then provide [4~8] risk categories "
        "(categories or dimensions, not clause edits).\n"
        "These categories should be strongly related to the key points and can guide the "
        "subsequent implantation of easily overlooked traps in the contract.\n\n"
        f"Contract Type: {contract_type}\n"
        f"Business Scenario: {scenario}\n"
        f"Our Party's Position: {role}\n\n"
        "Please strictly output JSON:\n"
        "{\n"
        "  \"content_outline\": [\"...\"],\n"
        "  \"risk_categories\": [\"...\"]\n"
        "}\n"
    )

    messages = [
        {"role": "system", "content": "You are a senior legal expert."},
        {"role": "user", "content": prompt},
    ]

    raw = teacher.generate(
        messages,
        GenConfig(max_new_tokens=800, temperature=0.3),
    )

    data = safe_json_loads(raw)
    if data is None:
        jb = extract_first_json_block(raw) or "{}"
        data = safe_json_loads(jb) or {}

    risks = data.get("risk_categories") or []
    cleaned, seen = [], set()

    for r in risks:
        if not isinstance(r, str):
            continue
        s = r.strip().replace("。", "").replace("\n", " ")
        if not s or s in seen:
            continue
        seen.add(s)
        cleaned.append(s)

    if not cleaned:
        cleaned = [
            "Unclear definitions and scope",
            "Delivery and acceptance criteria",
            "Payment and settlement terms",
            "Liability for breach and compensation",
            "Intellectual property rights and licensing",
            "Dispute resolution and governing law",
        ]

    return cleaned[:max_k]


def _load_prefix_map(path: str = PREFIX_MAP_FILE) -> Dict[str, str]:
    m: Dict[str, str] = {}
    if not os.path.exists(path):
        return m

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if "\t" not in s:
                continue
            k, v = s.split("\t", 1)
            k, v = k.strip(), v.strip()
            if k and v:
                m[k] = v

    return m


def _fallback_prefix(contract_type: str) -> str:
    h = hashlib.md5(contract_type.encode("utf-8")).hexdigest()[:4].upper()
    return f"C{h}"


def _make_case_id(contract_type: str, idx: int) -> str:
    mp = _load_prefix_map()
    pfx = mp.get(contract_type) or _fallback_prefix(contract_type)
    return f"{pfx}-{idx:04d}"


def _render_dataset_prompt(
    template: str,
    case_id: str,
    contract_type: str,
    scenario: str,
    role: str,
    risks_str: str,
) -> str:
    s = template
    s = s.replace("{{contract_type}}", contract_type)
    s = s.replace("{{scenario_description}}", scenario)
    s = s.replace("{{our_party_role}}", role)
    s = s.replace("{{risk_points_list}}", risks_str)
    s = s.replace("{{case_id}}", case_id)
    return s


def generate_golden_set(
    out_dir: str,
    config_path: str,
    teacher_model_path: Optional[str] = None,
    num_cases: int = 50,
    seed: int = 42,
):
    set_seed(seed)
    ensure_dir(out_dir)

    cfg = json.loads(
        open(config_path, "r", encoding="utf-8").read()
    )

    dataset_tmpl = read_text(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "prompts",
            "dataset_prompt_template.txt",
        )
    )

    teacher = LocalQwen(model_path=teacher_model_path)

    combos = list(
        itertools.product(
            cfg["contract_types"],
            cfg["scenarios"],
            cfg["roles"],
        )
    )

    random.shuffle(combos)
    combos = combos[:num_cases]

    items: List[Dict[str, Any]] = []

    for idx, (ctype, scenario, role) in enumerate(combos, 1):
        case_id = _make_case_id(ctype, idx)

        dyn_risks = _derive_risk_categories(
            teacher,
            ctype,
            scenario,
            role,
            max_k=6,
        )

        risks_str = _format_risk_list(dyn_risks)

        prompt = _render_dataset_prompt(
            dataset_tmpl,
            case_id,
            ctype,
            scenario,
            role,
            risks_str,
        )

        messages = [
            {"role": "system", "content": "You are a senior legal expert."},
            {"role": "user", "content": prompt},
        ]

        raw = teacher.generate(
            messages,
            GenConfig(max_new_tokens=1500, temperature=0.7),
        )

        data = safe_json_loads(raw)
        if data is None:
            jb = extract_first_json_block(raw) or "{}"
            data = safe_json_loads(jb) or {}

        data["__raw_model_output__"] = raw
        data["__case_id__"] = case_id
        data["__seed_risk_categories_dynamic__"] = dyn_risks

        data.setdefault("annotations", [])
        data.setdefault("contract_type", ctype)
        data.setdefault("scenario_desc", scenario)
        data.setdefault("my_party_role", role)
        data.setdefault("qa_pairs", [])

        out_path = os.path.join(out_dir, f"{case_id}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        items.append(data)

    with open(
        os.path.join(out_dir, "manifest.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "count": len(items),
                "cases": [x["__case_id__"] for x in items],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def load_cases(dir_path: str) -> List[Dict[str, Any]]:
    files = [
        f
        for f in os.listdir(dir_path)
        if f.endswith(".json") and f != "manifest.json"
    ]

    out: List[Dict[str, Any]] = []

    for f in sorted(files):
        with open(
            os.path.join(dir_path, f),
            "r",
            encoding="utf-8",
        ) as fp:
            out.append(json.load(fp))

    return out
