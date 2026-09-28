from typing import Dict, Any, List

from app.tools.evidence import citation_gate


def _gate(
    stage: str,
    text: str,
    evidence_records: List[Dict[str, Any]],
    hard_fail: bool = True,
) -> Dict[str, Any]:
    """Run deterministic citation validation for one pipeline stage."""
    result = citation_gate(
        text=text,
        evidence_records=evidence_records,
        require_sources_section=False,
    )

    errors = list(result.get("errors", []))
    warnings = list(result.get("warnings", []))
    passed = len(errors) == 0

    # V1 policy: only Research is a hard gate.
    if not hard_fail and errors:
        warnings.extend(errors)
        passed = True

    return {
        "stage": stage,
        "passed": passed,
        "hard_fail": hard_fail,
        "errors": errors if hard_fail else [],
        "warnings": warnings,
        "used_source_ids": result.get("used_source_ids", []),
        "invalid_source_ids": result.get("invalid_source_ids", []),
        "uncited_claims": result.get("uncited_claims", []),
        "unused_source_ids": result.get("unused_source_ids", []),
    }


def research_gate(state: Dict[str, Any]) -> Dict[str, Any]:
    return _gate(
        "research",
        state.get("research_result", ""),
        state.get("evidence_records", []),
        hard_fail=True,
    )


def analysis_gate(state: Dict[str, Any]) -> Dict[str, Any]:
    return _gate(
        "analysis",
        state.get("analysis_result", ""),
        state.get("evidence_records", []),
        hard_fail=False,
    )


def strategy_gate(state: Dict[str, Any]) -> Dict[str, Any]:
    return _gate(
        "strategy",
        state.get("strategy_result", ""),
        state.get("evidence_records", []),
        hard_fail=False,
    )
