import os, json, argparse
from typing import List, Dict, Any, Optional, Tuple

from rcbsf.llm import LocalQwen, GenConfig
from rcbsf.utils import set_seed
from rcbsf.maf import DrafterAgent
from rcbsf.audit import RiskEditorAgent
from fusion.fusion_methods import fuse_signals
from outer import decide_with_budget


_DRAFTER_FALLBACK = """You are an expert who resolves risks item by item"""


def _load_drafter_prompt_template() -> str:
    cands = [
        os.getenv("DRAFTER_PROMPT_TEMPLATE") or "",
        os.path.join(os.path.dirname(__file__), "prompts", "drafter_prompt.txt"),
        os.path.join(os.path.dirname(__file__), "prompts", "drafter_prompt.txt"),
        "/mnt/data/drafter_prompt.txt",
    ]
    for p in cands:
        if p and os.path.exists(p):
            try:
                return open(p, "r", encoding="utf-8").read()
            except Exception:
                pass
    return _DRAFTER_FALLBACK


def _compose_drafter_instruction(outer_hint: str, fusion_hint: str, inner_hint: Optional[str]) -> str:
    tpl = _load_drafter_prompt_template()
    return (
        tpl.replace("{{OUTER_HINT}}", outer_hint.strip())
        .replace("{{FUSION_HINT}}", (fusion_hint or "").strip())
        .replace(
            "{{INNER_HINT}}",
            (inner_hint or "(No inner review key points in the first round)").strip(),
        )
    )


def _ensure_changed_text(
    prev_text: str,
    new_text: str,
    model: LocalQwen,
    drafter: DrafterAgent,
    outer_hint: str,
    fusion_hint: str,
    inner_hint: Optional[str],
) -> str:
    if (prev_text or "").strip() == (new_text or "").strip():
        force_note = (
            "\n\n[Force Rewrite] The previous version has not changed at all. "
            "Please perform substantive modifications or supplements to risk-related clauses item by item, "
            "and do not return the original text as is."
        )
        instr = _compose_drafter_instruction(
            outer_hint, fusion_hint + force_note, inner_hint
        )
        retry = drafter.revise(
            prev_text,
            instruction=instr,
            gen_config=GenConfig(max_new_tokens=3000, temperature=0.35, do_sample=True),
        )
        return retry
    return new_text


def load_cases(path: str) -> List[Dict]:
    cases = []

    if os.path.isdir(path):
        for filename in sorted(os.listdir(path)):
            file_path = os.path.join(path, filename)
            if not os.path.isfile(file_path):
                continue
            if filename.endswith(".jsonl"):
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            try:
                                case = json.loads(line)
                                if isinstance(case, dict):
                                    cases.append(case)
                            except Exception:
                                pass
            elif filename.endswith(".json"):
                try:
                    case_data = json.load(open(file_path, "r", encoding="utf-8"))
                    if isinstance(case_data, list):
                        cases.extend(
                            [c for c in case_data if isinstance(c, dict)]
                        )
                    elif isinstance(case_data, dict):
                        cases.append(case_data)
                except Exception:
                    pass

    elif os.path.isfile(path):
        if path.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            case = json.loads(line)
                            if isinstance(case, dict):
                                cases.append(case)
                        except Exception:
                            pass
        elif path.endswith(".json"):
            try:
                case_data = json.load(open(path, "r", encoding="utf-8"))
                if isinstance(case_data, list):
                    cases.extend(
                        [c for c in case_data if isinstance(c, dict)]
                    )
                elif isinstance(case_data, dict):
                    cases.append(case_data)
            except Exception:
                pass
    else:
        raise ValueError(f"Invalid path: {path}")

    seen_ids = set()
    filtered_cases = []
    for case in cases:
        if not isinstance(case, dict) or not case:
            continue
        cid = (
            case.get("case_id")
            or case.get("__case_id__")
            or str(id(case))
        )
        if cid not in seen_ids:
            seen_ids.add(cid)
            filtered_cases.append(case)

    if not filtered_cases:
        print(f"Warning: No valid dictionary cases loaded from {path}")
    return filtered_cases


def inner_loop_three(
    drafter: DrafterAgent,
    risk_v2: RiskEditorAgent,
    outer_categories: List[str],
    outer_qmap: Dict[str, Dict[str, Any]],
    outer_hint_text: str,
    fusion_method: str,
    q_weights: Tuple[float, float, float, float],
    contract_budget: Optional[int],
    audit_budget: Optional[int],
    init_contract: Optional[str] = None,
):
    history = []
    current = init_contract or ""

    fusion_hint1 = (
        "[Iteration #1] Generate the initial draft according to outer constraints and budget. "
        "Risk-related clauses must be executable. Do not return the original text as is."
    )
    instr1 = _compose_drafter_instruction(
        outer_hint_text, fusion_hint1, inner_hint=None
    )
    if current:
        revised = drafter.revise(
            current,
            instruction=instr1,
            outer_hint=outer_hint_text,
            fusion_hint=fusion_hint1,
            inner_hint=None,
            gen_config=GenConfig(
                max_new_tokens=3000, temperature=0.35, do_sample=True
            ),
        )
        revised = _ensure_changed_text(
            current,
            revised,
            drafter.model,
            drafter,
            outer_hint_text,
            fusion_hint1,
            None,
        )
    else:
        revised = drafter.draft(
            instruction=instr1,
            outer_hint=outer_hint_text,
            fusion_hint=fusion_hint1,
            inner_hint=None,
            gen_config=GenConfig(
                max_new_tokens=3000, temperature=0.35, do_sample=True
            ),
        )
    current = revised
    history.append(
        {"stage": "draft#1", "contract": current, "instruction": instr1}
    )

    audit2 = risk_v2.audit(current, outer_categories)
    q_inner = audit2["qmap"]

    fused2, topk2, budget_hint2 = fuse_signals(
        method=fusion_method,
        categories=outer_categories,
        outer_q=outer_qmap,
        inner_q=q_inner,
        q_weights=q_weights,
        contract_budget=contract_budget,
        audit_budget=audit_budget,
    )
    inner_hint2 = risk_v2.to_drafter_prompt(outer_categories, q_inner)

    fusion_hint2 = (
        f"[Iteration #2 Fusion] {budget_hint2}\n"
        f"Please prioritize revisions: {topk2}."
    )
    instr2 = _compose_drafter_instruction(
        outer_hint_text, fusion_hint2, inner_hint2
    )
    revised2 = drafter.revise(
        current,
        instruction=instr2,
        outer_hint=outer_hint_text,
        fusion_hint=fusion_hint2,
        inner_hint=inner_hint2,
        gen_config=GenConfig(
            max_new_tokens=3000, temperature=0.35, do_sample=True
        ),
    )
    revised2 = _ensure_changed_text(
        current,
        revised2,
        drafter.model,
        drafter,
        outer_hint_text,
        fusion_hint2,
        inner_hint2,
    )
    current = revised2
    history.append(
        {
            "stage": "draft#2",
            "contract": current,
            "instruction": instr2,
            "audit": audit2,
            "fused": fused2,
            "topk": topk2,
        }
    )

    audit3 = risk_v2.audit(current, outer_categories)
    q_inner3 = audit3["qmap"]
    fused3, topk3, budget_hint3 = fuse_signals(
        method=fusion_method,
        categories=outer_categories,
        outer_q=outer_qmap,
        inner_q=q_inner3,
        q_weights=q_weights,
        contract_budget=contract_budget,
        audit_budget=audit_budget,
    )
    inner_hint3 = risk_v2.to_drafter_prompt(outer_categories, q_inner3)

    fusion_hint3 = (
        f"[Iteration #3 Fusion] {budget_hint3}\n"
        f"Please finalize the contract to ensure Top-{len(topk3)} risks are closed or significantly mitigated."
    )
    instr3 = _compose_drafter_instruction(
        outer_hint_text, fusion_hint3, inner_hint3
    )
    revised3 = drafter.revise(
        current,
        instruction=instr3,
        outer_hint=outer_hint_text,
        fusion_hint=fusion_hint3,
        inner_hint=inner_hint3,
        gen_config=GenConfig(
            max_new_tokens=3000, temperature=0.35, do_sample=True
        ),
    )
    revised3 = _ensure_changed_text(
        current,
        revised3,
        drafter.model,
        drafter,
        outer_hint_text,
        fusion_hint3,
        inner_hint3,
    )
    current = revised3
    history.append(
        {
            "stage": "draft#3",
            "contract": current,
            "instruction": instr3,
            "audit": audit3,
            "fused": fused3,
            "topk": topk3,
        }
    )

    return current, {"audit2": audit2, "audit3": audit3}, history


def build_outer_qmap(details: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    qmap = {}
    for item in details.get("risk_categories", []):
        cat = item.get("category")
        if not cat:
            continue
        q_scores = item.get("q", {})
        qmap[cat] = {
            "Q1": q_scores.get("Q1", "B"),
            "Q2": q_scores.get("Q2", "B"),
            "Q3": q_scores.get("Q3", "B"),
            "Q4": q_scores.get("Q4", "B"),
            "location": item.get("location", "Unknown"),
            "evidence": item.get("evidence", ""),
            "issue": item.get("issue", ""),
            "suggestion": item.get("suggestion", ""),
        }
    return qmap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--outer_rounds", type=int, default=3)
    ap.add_argument("--q_weights", type=str, default="0.4,0.2,0.2,0.2")
    ap.add_argument("--softmax_temp", type=float, default=1.0)
    ap.add_argument(
        "--fusion_method",
        type=str,
        default="weighted_sum",
        choices=["weighted_sum", "poe", "topk_with_budget", "moe"],
    )
    ap.add_argument("--contract_budget", type=int, default=1800)
    ap.add_argument("--audit_budget", type=int, default=900)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    model = LocalQwen(args.model) if args.model else LocalQwen()
    drafter = DrafterAgent(model)
    risk_v2 = RiskEditorAgent(model)
    cases = load_cases(args.data)
    q_weights = tuple(float(x) for x in args.q_weights.split(","))

    print(f"Loaded {len(cases)} valid cases from {args.data}")

    for idx, case in enumerate(cases):
        cid = (
            case.get("case_id")
            or case.get("__case_id__")
            or f"case_{idx + 1}"
        )
        base_contract = (
            case.get("contract_text")
            or case.get("contract")
            or case.get("text")
            or ""
        )
        out_history = []
        current_contract = base_contract

        print(f"Processing case: {cid} (rounds={args.outer_rounds})")

        for round_id in range(1, args.outer_rounds + 1):
            outer = decide_with_budget(
                model,
                current_contract or base_contract,
                q_weights=q_weights,
                softmax_temp=args.softmax_temp,
                contract_budget=args.contract_budget,
                audit_budget=args.audit_budget,
            )

            outer_qmap = build_outer_qmap(outer.details)

            final_contract, inner_artifacts, inner_hist = inner_loop_three(
                drafter,
                risk_v2,
                outer.categories,
                outer_qmap,
                outer.hint_text,
                args.fusion_method,
                q_weights,
                args.contract_budget,
                args.audit_budget,
                init_contract=current_contract,
            )

            out_history.append(
                {
                    "round": round_id,
                    "outer": {
                        "categories": outer.categories,
                        "weights": outer.weights,
                        "details": outer.details,
                        "hint_text": outer.hint_text,
                    },
                    "inner": {
                        "history": inner_hist,
                        "artifacts": inner_artifacts,
                    },
                    "final_contract_of_round": final_contract,
                }
            )
            current_contract = final_contract

        out = {
            "case_id": cid,
            "final_contract": current_contract,
            "rounds": out_history,
            "fusion_method": args.fusion_method,
            "contract_budget": args.contract_budget,
            "audit_budget": args.audit_budget,
        }
        with open(
            os.path.join(args.out_dir, f"{cid}.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"All cases processed. Output saved to {args.out_dir}")


if __name__ == "__main__":
    main()
