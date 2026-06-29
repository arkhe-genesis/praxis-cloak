from dataclasses import dataclass, field, asdict
from typing import Optional

CATEGORIES: set[str] = {
    "name",
    "exact_amount",
    "exact_date",
    "location",
    "address",
    "phone",
    "email",
    "employer",
    "account_id",
    "medical_specifics",
    "legal_specifics",
    "secret",
    "other_identifier",
}

REHYDRATE_DEFAULT: set[str] = {
    "name",
    "address",
    "phone",
    "email",
    "account_id",
    "location",
    "employer",
}

NEVER_REHYDRATE: set[str] = {
    "exact_amount",
    "exact_date",
    "medical_specifics",
    "legal_specifics",
    "secret",
}

# Non-attributable categories kept verbatim by default (the "zone" of ADR 0004).
# A bare amount or date does not identify a person, and generalizing it costs
# answer quality for no privacy gain (see the medical-001 over-redaction in the
# cloud-response comparison). The pipeline does not generalize these, and the
# missed-entity validator does not require them to be hidden. Combination-aware
# exceptions (a precise amount inside a tight quasi-identifier profile) are
# deferred to Phase 3.
KEEP_CATEGORIES: set[str] = {"exact_amount", "exact_date"}


@dataclass
class Replacement:
    original: str
    replacement: str
    category: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Case:
    id: str
    category: str
    raw_prompt: str
    sensitive_entities: list[dict]
    essential_context: list[str]
    notes: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "Case":
        return cls(
            id=d["id"],
            category=d["category"],
            raw_prompt=d["raw_prompt"],
            sensitive_entities=d.get("sensitive_entities", []),
            essential_context=d.get("essential_context", []),
            notes=d.get("notes", ""),
        )


@dataclass
class TransformResult:
    protected_message: str
    replacements: list[Replacement]
    raw_local_output: str
    parse_ok: bool
    parse_error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "protected_message": self.protected_message,
            "replacements": [r.to_dict() for r in self.replacements],
            "raw_local_output": self.raw_local_output,
            "parse_ok": self.parse_ok,
            "parse_error": self.parse_error,
        }


@dataclass
class ValidationResult:
    schema_ok: bool
    schema_errors: list[str]
    substring_violations: list[dict]
    leak_violations: list[dict]
    detector_hits: dict[str, list[str]]
    missed_entities: list[dict]
    empty_name_replacements: list[dict]
    diff_coverage_violation: Optional[dict]
    strict_noop_violations: dict

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def overall_ok(self) -> bool:
        return (
            self.schema_ok
            and not self.substring_violations
            and not self.leak_violations
            and not any(self.detector_hits.values())
            and not self.missed_entities
            and not self.empty_name_replacements
            and not self.diff_coverage_violation
            and not self.strict_noop_violations
        )


@dataclass
class RehydrationEvent:
    original: str
    replacement: str
    category: str
    outcome: str  # "substituted", "not_found", "ambiguous_origin", "category_disabled", "no_replacement"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RehydrationResult:
    rehydrated_response: str
    events: list[RehydrationEvent]

    def to_dict(self) -> dict:
        return {
            "rehydrated_response": self.rehydrated_response,
            "events": [e.to_dict() for e in self.events],
        }


@dataclass
class CaseRun:
    case_id: str
    raw_prompt: str
    transform: Optional[TransformResult] = None
    validation: Optional[ValidationResult] = None
    raw_response: Optional[str] = None
    protected_response: Optional[str] = None
    rehydration: Optional[RehydrationResult] = None
    timings_ms: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "raw_prompt": self.raw_prompt,
            "transform": self.transform.to_dict() if self.transform else None,
            "validation": self.validation.to_dict() if self.validation else None,
            "raw_response": self.raw_response,
            "protected_response": self.protected_response,
            "rehydration": self.rehydration.to_dict() if self.rehydration else None,
            "timings_ms": self.timings_ms,
            "errors": self.errors,
        }
