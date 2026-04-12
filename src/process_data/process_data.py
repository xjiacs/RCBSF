import os
import re
import json
import argparse
import time
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

from openai import OpenAI


@dataclass
class GenConfig:
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.95
    do_sample: bool = True
    seed: Optional[int] = 42
    stop: Optional[List[str]] = None


class ChatGPTClient:
    def __init__(self, model: str, client: Optional[OpenAI] = None):
        self.model = model
        self.client = client or OpenAI()

    def generate(self, messages: List[Dict[str, str]], gen_config: Optional[GenConfig] = None) -> str:
        gen_config = gen_config or GenConfig()
        params: Dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "temperature": gen_config.temperature,
            "top_p": gen_config.top_p,
            "max_completion_tokens": gen_config.max_new_tokens,
        }
        if gen_config.stop:
            params["stop"] = gen_config.stop
        if gen_config.seed is not None:
            params["seed"] = gen_config.seed
        completion = self.client.chat.completions.create(**params)
        content = completion.choices[0].message.content or ""
        return content.strip()


EN_WORDS_PATTERN = re.compile(r"[A-Za-z]+")

BANNED_RISK_PATTERNS = [
    r"\bcontract\s+validity\b",
    r"\bvalidity\b",
    r"\beffective\s+date\b",
    r"\bdates?\b",
    r"\bparty(?:\s+[ab])?\s+name\b",
    r"\bpart(?:y|ies)\s+identity\b",
    r"\bsignature[s]?\b",
    r"\bseal\b|\bcompany\s+chop\b",
    r"\bid(?:entification)?\b",
    r"\btermination\s+clause\s+ambigu(?:ity|ous)\b",
    r"\bambiguous\s+termination\b",
]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_text(path: Path, max_chars: int = 120_000) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if len(text) > max_chars:
        return text[:max_chars]
    return text


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


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def limit_words(text: str, max_words: int = 1500) -> str:
    tokens = re.findall(r"\S+", text)
    if len(tokens) <= max_words:
        return text.strip()
    return " ".join(tokens[:max_words]).strip()


def canonicalize_two_word_category(raw: str) -> str:
    if not isinstance(raw, str):
        return "General Contract"
    s = re.sub(r"[^A-Za-z\s-]", " ", raw)
    s = s.replace("-", " ")
    words = [w for w in s.split() if w]
    if len(words) >= 2:
        return f"{words[0].capitalize()} {words[1].capitalize()}"
    if len(words) == 1:
        return f"{words[0].capitalize()} Agreement"
    return "General Contract"


def make_unique_filename(base_dir: Path, category_two_words: str) -> str:
    slug = "_".join(category_two_words.split())
    slug = re.sub(r"[^A-Za-z0-9_]", "", slug)
    if not slug:
        slug = "Contract"
    n = 1
    while True:
        fn = f"{slug}_{n:02d}.json"
        if not (base_dir / fn).exists():
            return fn
        n += 1


def filter_banned_categories(cats: List[str]) -> List[str]:
    out: List[str] = []
    for c in cats:
        cc = c.strip()
        if not cc:
            continue
        banned = any(re.search(pat, cc, flags=re.IGNORECASE) for pat in BANNED_RISK_PATTERNS)
        if not banned:
            out.append(cc)
    seen = set()
    deduped: List[str] = []
    for c in out:
        k = c.lower()
        if k not in seen:
            seen.add(k)
            deduped.append(c)
    return deduped


def build_category_messages(source_contract: str, original_filename: str) -> List[Dict[str, str]]:
    system = (
        "You are a senior legal counsel. Your job is to classify the contract TYPE from its content. "
        "Return EXACTLY TWO English words representing a generic category (e.g., 'Software License', 'Service Contract', 'Data Processing'). "
        "Avoid any party names, dates, or identifiers."
    )
    user = f"""
Classify the contract type based on the following content. 
Output STRICTLY a single JSON object:
{{
  "category": "<Two English Words>"
}}

If uncertain, use "General Contract".

Original filename: {original_filename}

CONTRACT CONTENT (may be truncated):
<<<SOURCE_START>>>
{source_contract}
<<<SOURCE_END>>>
""".strip()
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_template_messages(source_contract: str, detected_category: str, original_filename: str) -> List[Dict[str, str]]:
    system = (
        "You are a senior legal counsel. You draft precise, neutral, enforceable contract templates.\n"
        "All outputs MUST be in English only.\n"
    )
    user = f"""
Task:
Read the original contract text and produce a concise, best-practice ENGLISH CONTRACT TEMPLATE that:
- Starts with: Category: {detected_category}
- Uses clear sections with headings: Definitions; Parties and Scope; Deliverables/Acceptance (objective metrics); Service Levels & Support; Fees/Payment; IP Ownership/License; Confidentiality & Data Protection; Security & Compliance; Warranties; Indemnity; Liability Cap & Exclusions; Change Control; Subcontracting/Assignment; Audit; Insurance; Governing Law & Dispute Resolution; Force Majeure; Notices; Termination & Effects; Miscellaneous.
- Uses neutral placeholders like [Party A], [Party B], [Effective Date], [Fee], [SLA Metric], etc.
- Abstracts the source into a reusable template (do not copy private details).
- Keep the FULL TEMPLATE within 1500 WORDS. Prioritize clarity, enforceability, objective acceptance criteria, and operational details.
- Output PLAIN TEXT ONLY (no markdown, no JSON).

Context:
- Original filename: {original_filename}
- Detected category: {detected_category}

Original contract begins:
<<<SOURCE_START>>>
{source_contract}
<<<SOURCE_END>>>
""".strip()
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_risk_categories_messages(contract_template_text: str) -> List[Dict[str, str]]:
    system = (
        "You are a senior legal counsel enumerating risk dimensions from a contract template. "
        "Output MUST be English-only and STRICTLY a single JSON object."
    )
    prompt = (
        "Produce 8–12 SPECIFIC and ACTIONABLE RISK CATEGORIES that are higher-level dimensions "
        "(not clause edits). Aim for detailed, high-signal categories.\n"
        "EXCLUDE categories about: contract validity, parties' names/identities, dates/signatures/seals/IDs, "
        "and 'ambiguous termination clause' as a standalone category (since templates omit personal/dated info).\n\n"
        "Strictly output JSON only:\n"
        "{\n"
        '  "risk_categories": ["..."]\n'
        "}\n"
        "Rules:\n"
        "- 8 to 12 items\n"
        "- 3–10 words each; crisp but specific\n"
        "- Strongly aligned with the template’s core obligations and operational controls\n"
        "- English only\n"
    )
    user = f"CONTRACT TEMPLATE:\n<<<TEMPLATE_START>>>\n{contract_template_text}\n<<<TEMPLATE_END>>>"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
        {"role": "user", "content": user},
    ]


def run_pipeline(
    input_dir: Path,
    templates_out_dir: Path,
    enriched_out_dir: Path,
    template_model: str,
    risk_model: str,
    max_source_chars: int = 120_000,
    gen_category: GenConfig = GenConfig(max_new_tokens=128, temperature=0.2, top_p=0.9, do_sample=False),
    gen_a: GenConfig = GenConfig(max_new_tokens=2048, temperature=0.6, top_p=0.9),
    gen_b_categories: GenConfig = GenConfig(max_new_tokens=700, temperature=0.2, top_p=0.9, do_sample=False),
) -> None:
    ensure_dir(templates_out_dir)
    ensure_dir(enriched_out_dir)

    client = OpenAI()
    llm_template = ChatGPTClient(model=template_model, client=client)
    llm_risk = ChatGPTClient(model=risk_model, client=client)

    txt_files = sorted([p for p in input_dir.glob("*.txt") if p.is_file()])
    total = len(txt_files)
    if total == 0:
        print(f"[Info] No .txt files found in: {input_dir}")
        return

    print(f"[Start] Found {total} .txt files.")
    success_t = 0
    success_r = 0
    failures: List[Dict[str, str]] = []

    for idx, txt_path in enumerate(txt_files, start=1):
        t0 = time.time()
        original_name = txt_path.name
        try:
            source = load_text(txt_path, max_chars=max_source_chars)

            cat_msgs = build_category_messages(source, original_name)
            cat_raw = llm_template.generate(cat_msgs, gen_config=gen_category)
            cat_json = parse_json_loose(cat_raw)
            if isinstance(cat_json, dict) and isinstance(cat_json.get("category"), str):
                detected_category = canonicalize_two_word_category(cat_json["category"])
            else:
                words = EN_WORDS_PATTERN.findall(cat_raw)
                detected_category = canonicalize_two_word_category(" ".join(words[:2]))

            out_name = make_unique_filename(templates_out_dir, detected_category)
            case_id = Path(out_name).stem

            messages_a = build_template_messages(source, detected_category, original_name)
            template_text_raw = llm_template.generate(messages_a, gen_config=gen_a)
            template_text = limit_words(template_text_raw, max_words=1500)
            wc = word_count(template_text)

            stage1_obj = {
                "case_id": case_id,
                "contract_text": template_text,
            }
            stage1_path = templates_out_dir / out_name
            stage1_path.write_text(json.dumps(stage1_obj, ensure_ascii=False, indent=2), encoding="utf-8")
            success_t += 1

            messages_b_cat = build_risk_categories_messages(template_text)
            risk_raw = llm_risk.generate(messages_b_cat, gen_config=gen_b_categories)
            risk_json = parse_json_loose(risk_raw)
            if not risk_json or "risk_categories" not in risk_json or not isinstance(risk_json["risk_categories"], list):
                strict_messages = messages_b_cat[:-1] + [
                    {
                        "role": "user",
                        "content": "IMPORTANT: Output ONLY one JSON object with key 'risk_categories' as an array of 8–12 short English strings. No commentary.",
                    },
                    messages_b_cat[-1],
                ]
                risk_raw = llm_risk.generate(strict_messages, gen_config=gen_b_categories)
                risk_json = parse_json_loose(risk_raw)

            if not risk_json or "risk_categories" not in risk_json or not isinstance(risk_json["risk_categories"], list):
                raise ValueError(f"Risk model did not return valid risk_categories JSON. Raw: {risk_raw[:300]} ...")

            filtered_categories = filter_banned_categories(risk_json["risk_categories"])
            if len(filtered_categories) < 6:
                reprompt_strict = build_risk_categories_messages(template_text)
                reprompt_strict[1]["content"] += "\nHARD CONSTRAINT: Do NOT include categories related to validity, identity, dates/signatures/seals/IDs, or 'ambiguous termination clause'."
                risk_raw2 = llm_risk.generate(reprompt_strict, gen_config=gen_b_categories)
                risk_json2 = parse_json_loose(risk_raw2)
                if risk_json2 and isinstance(risk_json2.get("risk_categories"), list):
                    filtered_categories = filter_banned_categories(risk_json2["risk_categories"])

            if not filtered_categories:
                raise ValueError("All risk categories were filtered out as banned or invalid.")

            enriched = {
                "case_id": case_id,
                "contract_text": template_text,
                "risk_categories": filtered_categories,
            }
            enriched_path = enriched_out_dir / out_name
            enriched_path.write_text(json.dumps(enriched, ensure_ascii=False, indent=2), encoding="utf-8")
            success_r += 1

            dt = time.time() - t0
            print(f"[{idx}/{total}] OK  | {original_name} -> {out_name} | category='{detected_category}' | words={wc} | template ✓, risks ✓ | {dt:.1f}s")

        except Exception as e:
            dt = time.time() - t0
            print(f"[{idx}/{total}] FAIL| {original_name} | {dt:.1f}s | {e}")
            failures.append({"file": original_name, "error": str(e)})

    print("\n=== Summary ===")
    print(f"Templates (≤1500 words) : {success_t}/{total}")
    print(f"Enriched (with categories): {success_r}/{total}")
    if failures:
        print("Failures:")
        for f in failures:
            print(f" - {f['file']}: {f['error']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Use ChatGPT to generate concise contract templates (≤1500 words) and risk categories from source .txt contracts."
    )
    parser.add_argument("--input_dir", required=True, type=Path, help="Directory with source .txt contracts")
    parser.add_argument("--templates_out_dir", required=True, type=Path, help="Output dir for stage1 JSONs (case_id + contract_text)")
    parser.add_argument(
        "--enriched_out_dir",
        required=True,
        type=Path,
        help="Output dir for stage2 JSONs (case_id + contract_text + risk_categories)",
    )
    parser.add_argument(
        "--template_model",
        required=False,
        default="gpt-4o-mini",
        help="ChatGPT model name for template generation and category detection",
    )
    parser.add_argument(
        "--risk_model",
        required=False,
        default="gpt-4o-mini",
        help="ChatGPT model name for risk category generation",
    )
    parser.add_argument("--max_source_chars", type=int, default=120_000, help="Soft cap on source txt length")
    parser.add_argument("--cat_max_new_tokens", type=int, default=128)
    parser.add_argument("--cat_temperature", type=float, default=0.2)
    parser.add_argument("--cat_top_p", type=float, default=0.9)
    parser.add_argument("--a_max_new_tokens", type=int, default=2048)
    parser.add_argument("--a_temperature", type=float, default=0.6)
    parser.add_argument("--a_top_p", type=float, default=0.9)
    parser.add_argument("--b_cat_max_new_tokens", type=int, default=700)
    parser.add_argument("--b_cat_temperature", type=float, default=0.2)
    parser.add_argument("--b_cat_top_p", type=float, default=0.9)

    args = parser.parse_args()

    if not args.template_model or not args.risk_model:
        raise SystemExit("Provide --template_model and --risk_model.")

    gen_category = GenConfig(
        max_new_tokens=args.cat_max_new_tokens,
        temperature=args.cat_temperature,
        top_p=args.cat_top_p,
        do_sample=False,
    )
    gen_a = GenConfig(
        max_new_tokens=args.a_max_new_tokens,
        temperature=args.a_temperature,
        top_p=args.a_top_p,
        do_sample=True,
    )
    gen_b_categories = GenConfig(
        max_new_tokens=args.b_cat_max_new_tokens,
        temperature=args.b_cat_temperature,
        top_p=args.b_cat_top_p,
        do_sample=False,
    )

    run_pipeline(
        input_dir=args.input_dir,
        templates_out_dir=args.templates_out_dir,
        enriched_out_dir=args.enriched_out_dir,
        template_model=args.template_model,
        risk_model=args.risk_model,
        max_source_chars=args.max_source_chars,
        gen_category=gen_category,
        gen_a=gen_a,
        gen_b_categories=gen_b_categories,
    )


if __name__ == "__main__":
    main()
