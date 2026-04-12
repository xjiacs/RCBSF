
import os, json, argparse, csv, statistics, sys, difflib
from typing import Dict, Any, List, Tuple
from utils_local import ensure_dir, read_json, write_json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "fixed_risk_eval_OR", "fixed_risk_eval_OR")))
from rcbsf.llm import (
    BaseModel, QwenModel, LlamaChatModel, BaichuanChatModel, ChatGLMModel,
    MistralChatModel, GemmaChatModel, LlamaCppModel, GenericModel, GptOssModel
)
from rcbsf.judge import judge_contract, judge_resolution_from_seed_categories

def detect_label_and_cls(model_path: str):
    base = os.path.basename(model_path).lower()
    ext = os.path.splitext(base)[1]
    if ext == ".gguf": return ("llama_cpp", LlamaCppModel)
    if "qwen" in base: return ("qwen", QwenModel)
    if "baichuan" in base: return ("baichuan", BaichuanChatModel)
    if "chatglm" in base or "glm" in base: return ("chatglm", ChatGLMModel)
    if "mistral" in base or "mixtral" in base: return ("mistral", MistralChatModel)
    if "gemma" in base: return ("gemma", GemmaChatModel)
    if "llama" in base or "vicuna" in base: return ("llama", LlamaChatModel)
    if "gpt-oss" in base: return ("gpt-oss", GptOssModel)
    return ("generic", GenericModel)

def load_final_cases(dir_path: str):
    by_id = {}
    for fn in os.listdir(dir_path):
        if not fn.endswith(".json"): continue
        p = os.path.join(dir_path, fn)
        obj = read_json(p)
        cid = obj.get("case_id") or os.path.splitext(fn)[0]
        final = obj.get("final_contract","") or obj.get("contract_text","")
        by_id[cid] = {"case_id": cid, "final_contract": final, "__raw__": obj}
    return by_id

def load_all_methods(final_dirs: List[str]):
    out = {}
    for d in final_dirs:
        method_label = os.path.basename(os.path.normpath(d))
        out[method_label] = load_final_cases(d)
    return out

def load_seeds_for_model(seeds_root: str, mlabel: str):
    d = os.path.join(seeds_root, mlabel)
    out = {}
    if not os.path.isdir(d): return out
    for fn in os.listdir(d):
        if not fn.endswith(".json"): continue
        obj = read_json(os.path.join(d, fn))
        cid = obj.get("case_id") or os.path.splitext(fn)[0]
        seeds = obj.get("seed_items") or []
        out[cid] = seeds
    return out

def ratio(a, b):
    return difflib.SequenceMatcher(None, a or "", b or "").ratio()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final_dirs", nargs="+", required=True, help="FINAL dirs; each is a different method")
    ap.add_argument("--seeds_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_paths", nargs="+", required=True)
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    out_csv = os.path.join(args.out_dir, "evaluation_two_stage_multi.csv")
    avg_csv = os.path.join(args.out_dir, "method_avg_two_stage_multi.csv")
    stats_json = os.path.join(args.out_dir, "summary_stats_two_stage_multi.json")
    change_csv = os.path.join(args.out_dir, "changes_across_methods.csv")
    diffs_dir = os.path.join(args.out_dir, "diffs")
    ensure_dir(diffs_dir)

    methods_map = load_all_methods(args.final_dirs)  # {method: {case_id: {...}}}
    method_names = sorted(methods_map.keys())
    if not method_names:
        print("[WARN] no final dirs content", file=sys.stderr)
        return
    ref_method = method_names[0]

    model_specs = []
    for mp in args.model_paths:
        base = os.path.basename(mp)
        fam, cls = detect_label_and_cls(mp)
        model_specs.append((f"{fam}@{base}", cls, mp))

    all_rows = []
    by_method_and_judge = {}

    for mlabel, cls, mp in model_specs:
        model: BaseModel = cls(mp)
        seeds_map = load_seeds_for_model(args.seeds_root, mlabel)

        for method in method_names:
            per_case = methods_map[method]
            rows = []
            for cid, fin in per_case.items():
                final_contract = fin.get("final_contract","")
                seed_items = seeds_map.get(cid, [])
                judge = judge_contract(model, final_contract) if final_contract else {}
                decisions = judge_resolution_from_seed_categories(model, final_contract, seed_items) if (final_contract and seed_items) else []
                resolved_rate = (sum(1 for d in decisions if d.get("resolved")) / len(decisions)) if decisions else ""
                used_risk_categories = [it.get("category","") for it in seed_items]
                rows.append({
                    "case_id": cid,
                    "method": method,     # FINAL method name (dir label)
                    "judge": mlabel,      # evaluating model
                    "clarity": (judge.get("scores") or {}).get("clarity", 0.0),
                    "rigor": (judge.get("scores") or {}).get("rigor", 0.0),
                    "protection": (judge.get("scores") or {}).get("balance", 0.0),
                    "professionalism": (judge.get("scores") or {}).get("professionalism", 0.0),
                    "used_risk_categories": "|".join(used_risk_categories),
                    "high_risk_category_coverage": "",
                    "high_risk_resolved_rate": "",
                    "llm_seed_resolved_rate": resolved_rate,
                    "llm_seed_total": len(seed_items),
                })
            by_method_and_judge.setdefault((method, mlabel), []).extend(rows)
            all_rows.extend(rows)

    cols = ["case_id","method","judge","clarity","rigor","protection","professionalism",
            "used_risk_categories","high_risk_category_coverage","high_risk_resolved_rate",
            "llm_seed_resolved_rate","llm_seed_total"]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(all_rows)

    avg_rows = []
    for (method, judge_label), rows in by_method_and_judge.items():
        vals = lambda k: [r[k] for r in rows if isinstance(r.get(k), (int,float))]
        avg_rows.append({
            "method": method, "judge": judge_label, "case_count": len(rows),
            "avg_clarity": round(statistics.mean(vals("clarity")), 4) if vals("clarity") else 0.0,
            "avg_rigor": round(statistics.mean(vals("rigor")), 4) if vals("rigor") else 0.0,
            "avg_protection": round(statistics.mean(vals("protection")), 4) if vals("protection") else 0.0,
            "avg_professionalism": round(statistics.mean(vals("professionalism")), 4) if vals("professionalism") else 0.0,
            "avg_llm_seed_resolved_rate": round(statistics.mean([r["llm_seed_resolved_rate"] for r in rows if isinstance(r.get("llm_seed_resolved_rate"), (int,float))]), 4) if rows else "",
        })
    with open(avg_csv, "w", encoding="utf-8", newline="") as f:
        if avg_rows:
            w = csv.DictWriter(f, fieldnames=list(avg_rows[0].keys())); w.writeheader(); w.writerows(avg_rows)
    with open(stats_json, "w", encoding="utf-8") as f:
        json.dump({f"{r['method']}__{r['judge']}": r for r in avg_rows}, f, ensure_ascii=False, indent=2)

    union_ids = set()
    for m in method_names: union_ids |= set(methods_map[m].keys())
    union_ids = sorted(list(union_ids))

    change_rows = []
    for cid in union_ids:
        ref_text = (methods_map.get(ref_method, {}).get(cid) or {}).get("final_contract","")
        for m in method_names:
            cur_text = (methods_map.get(m, {}).get(cid) or {}).get("final_contract","")
            changed = (cur_text != ref_text)
            sim = ratio(ref_text, cur_text) if (ref_text or cur_text) else 1.0
            change_rows.append({
                "case_id": cid, "ref_method": ref_method, "method": m,
                "changed_vs_ref": bool(changed), "similarity_ratio": round(sim, 4)
            })
            if changed and ref_text and cur_text:
                case_dir = os.path.join(diffs_dir, cid); ensure_dir(case_dir)
                diff_path = os.path.join(case_dir, f"{ref_method}_vs_{m}.txt")
                diff = difflib.unified_diff(ref_text.splitlines(keepends=True),
                                            cur_text.splitlines(keepends=True),
                                            fromfile=f"{ref_method}:{cid}",
                                            tofile=f"{m}:{cid}", lineterm="")
                with open(diff_path, "w", encoding="utf-8") as df:
                    df.write("\n".join(diff))

    with open(change_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case_id","ref_method","method","changed_vs_ref","similarity_ratio"])
        w.writeheader(); w.writerows(change_rows)

    print(f"[OK] Detailed     -> {out_csv}")
    print(f"[OK] Averages     -> {avg_csv}")
    print(f"[OK] Summary JSON -> {stats_json}")
    print(f"[OK] Changes CSV  -> {change_csv}")
    print(f"[OK] Diffs        -> {diffs_dir}")

if __name__ == "__main__":
    main()
