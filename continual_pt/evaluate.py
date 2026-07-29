"""Held-out, deterministic evaluation used before and after learning."""
from __future__ import annotations

from .model import answer
from .schema import CaseResult, EvalCase, EvalReport


def _matches(response: str, case: EvalCase) -> bool:
    normalized = response.lower()
    required = all(term.lower() in normalized for term in case.must_contain)
    forbidden = any(term.lower() in normalized for term in case.forbidden)
    return required and not forbidden


def evaluate(model, tokenizer, cases: list[EvalCase], label: str) -> EvalReport:
    print(f"[eval] {label}: {len(cases)} held-out cases", flush=True)
    results = []
    for index, case in enumerate(cases, start=1):
        print(f"[eval] {label}: {index}/{len(cases)} {case.id}", flush=True)
        response = answer(model, tokenizer, case.prompt, max_new_tokens=96)
        results.append(CaseResult(case_id=case.id, passed=_matches(response, case), response=response))
    score = sum(result.passed for result in results) / len(results) if results else 1.0
    print(f"[eval] {label}: score={score:.3f}", flush=True)
    return EvalReport(label=label, score=score, cases=results)
