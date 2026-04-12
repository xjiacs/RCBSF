import os, json
from typing import Dict, Any, List, Optional
from .llm import LocalQwen, GenConfig
from .utils import read_text, safe_json_loads, count_tokens_approx


def _safe_format(template: str, **kwargs) -> str:
    class SafeDict(dict):
        def __missing__(self, k):
            return "{%s}" % k

    base = {
        "instruction": kwargs.get("instruction", ""),
        "previous_contract": kwargs.get("previous_contract", ""),
        "contract_text": kwargs.get("contract_text", ""),
        "outer_hint": kwargs.get("outer_hint", ""),
        "fusion_hint": kwargs.get("fusion_hint", ""),
        "inner_hint": kwargs.get("inner_hint", ""),
    }
    base.update(kwargs or {})
    return template.format_map(SafeDict(base))


class DrafterAgent:
    _DEFAULT_PROMPT_TMPL = """You are an expert contract lawyer. Your goal is to rewrite the contract to resolve specific risks.

The risk instructions include 5 core dimensions: Category, Location, Evidence, Issue, and Suggestion.
Please strictly follow these rules to revise the contract:
1. Read the "Issue" and "Suggestion" for each risk category carefully.
2. Locate the clauses mentioned in "Location"/"Evidence" (create them if they do not exist).
3. Apply the "Suggestion" to fix the "Issue".

[Outer Layer Analysis]
{OUTER_HINT}

[Fusion & Budget Constraints]
{FUSION_HINT}

[Inner Layer Detailed Review]
{INNER_HINT}

Important: Return ONLY the full revised contract text.
"""

    def __init__(self, model: LocalQwen):
        self.model = model
        self.prompt_tmpl = self._load_drafter_prompt_template()

    def _load_drafter_prompt_template(self) -> str:
        candidate_paths = [
            os.getenv("DRAFTER_PROMPT_TEMPLATE") or "",
            "./prompts/drafter_prompt.txt",
            "./prompts_ext/drafter_prompt.txt",
            "/public/home/chengtongtong/xsj/process/out_inner/rcbse_llm_leader_1_gpt/prompts/drafter_prompt.txt",
            "/mnt/data/drafter_prompt.txt",
        ]
        for path in candidate_paths:
            if path and os.path.exists(path):
                try:
                    return read_text(path, encoding="utf-8")
                except Exception:
                    pass
        return self._DEFAULT_PROMPT_TMPL

    def draft(
        self,
        instruction: str,
        gen_config: Optional[GenConfig] = None,
        **kwargs,
    ) -> str:
        outer_hint = kwargs.get("outer_hint")
        fusion_hint = kwargs.get("fusion_hint")
        inner_hint = kwargs.get("inner_hint")

        if "{instruction}" in self.prompt_tmpl:
            prompt_content = _safe_format(
                self.prompt_tmpl,
                instruction=instruction,
                previous_contract=kwargs.get("previous_contract", ""),
            )
        else:
            prompt_content = _safe_format(
                self.prompt_tmpl,
                OUTER_HINT=outer_hint or instruction,
                FUSION_HINT=fusion_hint or "",
                INNER_HINT=inner_hint or "",
                instruction=instruction,
            )

        messages = [
            {
                "role": "system",
                "content": "You are a professional legal contract expert. Revise the contract according to risk prompts (Location, Evidence, Issue, Suggestion). Enrich the contract while resolving risks. Output ONLY the full revised contract text.",
            },
            {"role": "user", "content": prompt_content},
        ]
        final_gen_config = gen_config or GenConfig(
            max_new_tokens=3000, temperature=0.35, do_sample=True
        )
        return self.model.generate(messages, final_gen_config).strip()

    def revise(
        self,
        previous_contract: str,
        instruction: str,
        gen_config: Optional[GenConfig] = None,
        **kwargs,
    ) -> str:
        outer_hint = kwargs.get("outer_hint")
        fusion_hint = kwargs.get("fusion_hint")
        inner_hint = kwargs.get("inner_hint")

        if "{instruction}" in self.prompt_tmpl:
            prompt_content = _safe_format(
                self.prompt_tmpl,
                instruction=instruction,
                previous_contract=kwargs.get("previous_contract", ""),
            )
        else:
            prompt_content = _safe_format(
                self.prompt_tmpl,
                OUTER_HINT=outer_hint or instruction,
                FUSION_HINT=fusion_hint or "",
                INNER_HINT=inner_hint or "",
                instruction=instruction,
            )

        full_prompt = f"""
[Previous Version of Contract]
{previous_contract}

[Risk Modification Instructions]
{prompt_content}
""".strip()

        messages = [
            {
                "role": "system",
                "content": "You are a professional legal contract expert. You must substantively modify the contract based on the provided Issue and Suggestion. Enrich the contract while resolving risks. Return ONLY the full revised contract text in plain text.",
            },
            {"role": "user", "content": full_prompt},
        ]
        final_gen_config = gen_config or GenConfig(
            max_new_tokens=3000, temperature=0.35, do_sample=True
        )
        return self.model.generate(messages, final_gen_config).strip()
