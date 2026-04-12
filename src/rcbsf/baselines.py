import os, json
from typing import Dict, Any
from .llm import LocalQwen, GenConfig
from .utils import count_tokens_approx

def baseline_b1_zeroshot(model: LocalQwen, instruction: str) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": "You are an experienced contract drafting expert."},
        {"role": "user", "content": instruction},
    ]
    out = model.generate(messages, GenConfig(max_new_tokens=1500, temperature=0.7))
    return {
        "final_contract": out,
        "history": [{"stage": "single_pass", "text": out}],
        "approx_tokens": count_tokens_approx(out),
    }

def baseline_b3_pipeline(model: LocalQwen, initial_contract_instruction: str) -> Dict[str, Any]:
    draft = model.generate(
        [
            {"role": "system", "content": "You are a professional contract drafter."},
            {"role": "user", "content": initial_contract_instruction},
        ],
        GenConfig(max_new_tokens=1500, temperature=0.7),
    )

    risk_prompt = (
        "You are a senior contract risk control expert. Please directly rewrite the following contract "
        "to correct potential high risks and improve fairness and enforceability. "
        "It is forbidden to output suggestions or lists; you must output the complete revised contract:\n"
        + draft
    )
    revised_by_risk = model.generate(
        [
            {"role": "system", "content": "You are a strict risk control officer."},
            {"role": "user", "content": risk_prompt},
        ],
        GenConfig(max_new_tokens=1500, temperature=0.3),
    )

    polish_inst = (
        "On the premise of not reducing the protection for our party, polish the wording and optimize "
        "the structure of the following contract:\n"
        f"{revised_by_risk}\n"
        "Only output the full text of the polished contract."
    )
    polished = model.generate(
        [
            {"role": "system", "content": "You are a professional contract drafter."},
            {"role": "user", "content": polish_inst},
        ],
        GenConfig(max_new_tokens=1200, temperature=0.6),
    )

    total = (
        count_tokens_approx(draft)
        + count_tokens_approx(revised_by_risk)
        + count_tokens_approx(polished)
    )
    return {
        "final_contract": polished,
        "history": [
            {"stage": "draft", "text": draft},
            {"stage": "risk_edit", "text": revised_by_risk},
            {"stage": "polish", "text": polished},
        ],
        "approx_tokens": total,
    }
