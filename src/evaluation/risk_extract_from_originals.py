
import os, json, argparse, sys
from typing import Dict, Any, List, Tuple
from prompts import build_original_risk_messages
from utils_local import ensure_dir, parse_json_loose, filter_seed_items, read_json, write_json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "fixed_risk_eval_OR", "fixed_risk_eval_OR")))
from rcbsf.llm import (
    GenConfig, BaseModel, QwenModel, LlamaChatModel,
    BaichuanChatModel, ChatGLMModel, MistralChatModel,
    GemmaChatModel, LlamaCppModel, GenericModel, GptOssModel
)
MODEL_FACTORY = {
    "qwen": QwenModel,
    "llama": LlamaChatModel,
    "baichuan": BaichuanChatModel,
    "chatglm": ChatGLMModel,
    "mistral": MistralChatModel,
    "gemma": GemmaChatModel,
    "llama_cpp": LlamaCppModel,  # for .gguf
    "generic": GenericModel,
    "gpt-oss": GptOssModel,
}

def detect_label_and_cls(model_path: str) -> Tuple[str, Any]:
    path = model_path.strip()
    base = os.path.basename(path).lower()
    ext = os.path.splitext(base)[1]
    if ext == ".gguf":
        return ("llama_cpp", MODEL_FACTORY["llama_cpp"])
    if "qwen" in base:
        return ("qwen", MODEL_FACTORY["qwen"])
    if "baichuan" in base:
        return ("baichuan", MODEL_FACTORY["baichuan"])
    if "chatglm" in base or "glm" in base:
        return ("chatglm", MODEL_FACTORY["chatglm"])
    if "mistral" in base or "mixtral" in base:
        return ("mistral", MODEL_FACTORY["mistral"])
    if "gemma" in base:
        return ("gemma", MODEL_FACTORY["gemma"])
    if "llama" in base or "vicuna" in base:
        return ("llama", MODEL_FACTORY["llama"])
    if "gpt-oss" in base:
        return ("gpt-oss", MODEL_FACTORY["gpt-oss"])
    return ("generic", MODEL_FACTORY["generic"])

def load_original_cases(original_dir: str):
    items = []
    for fn in os.listdir(original_dir):
        if not fn.endswith(".json"):
            continue
        p = os.path.join(original_dir, fn)
        obj = read_json(p)
        cid = obj.get("case_id") or os.path.splitext(fn)[0]
        txt = obj.get("contract_text","")
        if txt:
            items.append({"case_id": cid, "contract_text": txt})
    items.sort(key=lambda x: x["case_id"])
    return items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original_dir", required=True, help="Path to ORIGINAL contracts JSONs (each has case_id, contract_text)")
    ap.add_argument("--seeds_out_dir", required=True, help="Output dir for per-model seed items")
    ap.add_argument("--model_paths", nargs="+", required=True, help="One or more local model paths")
    ap.add_argument("--max_new_tokens", type=int, default=700)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--top_p", type=float, default=0.95)
    args = ap.parse_args()

    ensure_dir(args.seeds_out_dir)

    cases = load_original_cases(args.original_dir)
    if not cases:
        print(f"[WARN] No JSON files with contract_text found under {args.original_dir}", file=sys.stderr)
        return

    model_specs = []
    for mp in args.model_paths:
        base = os.path.basename(mp)
        label = base
        if base.lower().endswith(".gguf"):
            label = f"llama_cpp@{base}"
        elif "qwen" in base.lower(): label = f"qwen@{base}"
        elif "baichuan" in base.lower(): label = f"baichuan@{base}"
        elif "chatglm" in base.lower() or "glm" in base.lower(): label = f"chatglm@{base}"
        elif "mistral" in base.lower() or "mixtral" in base.lower(): label = f"mistral@{base}"
        elif "gemma" in base.lower(): label = f"gemma@{base}"
        elif "llama" in base.lower() or "vicuna" in base.lower(): label = f"llama@{base}"
        elif "gpt-oss" in base.lower(): label = f"gpt-oss@{base}"
        else: label = f"generic@{base}"

        # select class
        from_path = detect_label_and_cls(mp)[1]
        model_specs.append((label, from_path, mp))

    for mlabel, cls, mp in model_specs:
        print(f"[Init] Loading model: {mlabel} ({mp})")
        model: BaseModel = cls(mp)
        out_dir = os.path.join(args.seeds_out_dir, mlabel)
        ensure_dir(out_dir)
        gen_conf = GenConfig(max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p)

        for item in cases:
            cid = item["case_id"]
            text = item["contract_text"]
            msgs = build_original_risk_messages(text)
            raw = model.generate(msgs, gen_conf)
            # try:
            #     data = json.loads(raw)
            # except Exception:
            #     import re
            #     m = re.search(r"\{[\s\S]*\}", raw)
            #     data = json.loads(m.group(0)) if m else {}
            try:
                data = json.loads(raw)
            except Exception:
                import re
                m = re.search(r"\{[\s\S]*\}", raw)
                if m:
                    try:
                        data = json.loads(m.group(0))
                    except json.JSONDecodeError:
                        data = {}  # 正则提取的内容仍无法解析，设为空
                else:
                    data = {}
            seeds = data.get("seed_items") or []
            # simple filter: drop items w/o category or basis
            filtered = []
            seen = set()
            for i, it in enumerate(seeds):
                cat = str(it.get("category","")).strip()
                basis = str(it.get("basis","")).strip()
                if not cat or not basis:
                    continue
                if cat.lower() in seen:
                    continue
                seen.add(cat.lower())
                filtered.append({"id": i, "category": cat, "basis": basis})
            with open(os.path.join(out_dir, f"{cid}.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "case_id": cid,
                    "model": mlabel,
                    "seed_items": filtered,
                    "__raw__": raw[:2000]
                }, f, ensure_ascii=False, indent=2)
        print(f"[Done] Seeds for {mlabel} -> {out_dir}")

if __name__ == "__main__":
    main()
