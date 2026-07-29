"""Typed data exchanged by the continual-learning runtime."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvalCase:
    id: str
    prompt: str
    must_contain: list[str] = field(default_factory=list)
    forbidden: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LearningGoal:
    name: str
    objective: str
    research_queries: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    target_eval: list[EvalCase] = field(default_factory=list)
    retention_eval: list[EvalCase] = field(default_factory=list)
    trusted_domains: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LearningGoal":
        def cases(key: str) -> list[EvalCase]:
            # YAML treats unquoted values such as `4` as integers. Evaluation
            # is textual, so normalize every matcher at the configuration edge.
            return [
                EvalCase(
                    id=str(item["id"]),
                    prompt=str(item["prompt"]),
                    must_contain=[str(term) for term in item.get("must_contain", [])],
                    forbidden=[str(term) for term in item.get("forbidden", [])],
                )
                for item in data.get(key, [])
            ]

        return cls(
            name=data["name"],
            objective=data["objective"],
            research_queries=data.get("research_queries", []),
            sources=data.get("sources", []),
            target_eval=cases("target_eval"),
            retention_eval=cases("retention_eval"),
            trusted_domains=data.get("trusted_domains", []),
        )


@dataclass
class SourceDocument:
    url: str
    title: str
    text: str
    sections: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class LearningExample:
    prompt: str
    answer: str
    source_url: str
    evidence_quote: str
    attempted_answer: str = ""
    attempt_score: float = 0.0


@dataclass
class CaseResult:
    case_id: str
    passed: bool
    response: str


@dataclass
class EvalReport:
    label: str
    score: float
    cases: list[CaseResult]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
