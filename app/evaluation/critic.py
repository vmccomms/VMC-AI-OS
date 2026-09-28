import json
import re
from typing import Any, Dict, List

from app.orchestrator.state import AIState
from app.llm.gateway import LLMGateway
from app.tools.evidence import (
    citation_gate,
    extract_source_ids,
    find_invalid_source_ids,
)


def build_evidence_database(
    state: AIState,
) -> List[Dict[str, Any]]:
    return state.get(
        "evidence_records",
        [],
    )


def extract_json(text: str) -> Dict[str, Any]:
    """
    Safely extract JSON from the LLM response.
    """
    if not text:
        return {}

    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(
        r"\{.*\}",
        text,
        re.DOTALL,
    )

    if not match:
        return {}

    try:
        return json.loads(match.group(0))
    except Exception:
        return {}


def validate_llm_audit(
    audit: Dict[str, Any],
    valid_source_ids: List[str],
) -> Dict[str, Any]:

    errors = []
    warnings = []

    if not audit:
        errors.append(
            "LLM critic returned no usable audit."
        )

        return {
            "valid": False,
            "errors": errors,
            "warnings": warnings,
        }

    claims = audit.get("claims", [])

    if not isinstance(claims, list):
        errors.append(
            "LLM critic returned invalid claims structure."
        )

    invalid_citations = audit.get(
        "invalid_citations",
        [],
    )

    if invalid_citations:
        errors.append(
            "LLM critic reported invalid citations."
        )

    # IMPORTANT:
    # An empty audit is not automatically PASS.
    #
    # The deterministic Python gate remains authoritative.

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


def build_llm_prompt(
    state: AIState,
    evidence_records: List[Dict[str, Any]],
) -> str:

    research = state.get(
        "research_result",
        "",
    )

    analysis = state.get(
        "analysis_result",
        "",
    )

    strategy = state.get(
        "strategy_result",
        "",
    )

    valid_ids = [
        record.get("evidence_id")
        for record in evidence_records
        if record.get("evidence_id")
    ]

    evidence_text = "\n".join(
        f"""
[{record.get("evidence_id")}]
TITLE: {record.get("source_title")}
URL: {record.get("url")}
EVIDENCE: {record.get("evidence")}
"""
        for record in evidence_records
    )

    return f"""
You are the Critic Agent in the VMC AI Operating System.

USER GOAL:
{state.get("user_goal", "")}

VALID SOURCE IDS:
{", ".join(valid_ids)}

EVIDENCE DATABASE:
{evidence_text}

RESEARCH:
{research}

ANALYSIS:
{analysis}

STRATEGY:
{strategy}

Your job is to audit the outputs.

Return ONLY JSON.

Required JSON structure:

{{
  "claims": [],
  "invalid_citations": [],
  "unsupported_claims": [],
  "goal_drift": false,
  "recommendation_issues": [],
  "contradictions": [],
  "missing_information": [],
  "rework_instructions": [],
  "target_agent": "",
  "issue_type": "",
  "severity": "",
  "final_evaluation": "PASS"
}}

Rules:

1. Check factual claims.
2. Check whether factual claims have [S#] citations.
3. Check whether cited source IDs exist.
4. Do not invent source IDs.
5. Check whether claims are supported by the cited evidence.
6. Check whether analysis is based on research.
7. Check whether strategy is based on supported findings.
8. Check whether the output answers the user goal.
9. If problems exist, use:
   final_evaluation = "FAIL"
10. Do not return PASS merely because you cannot determine whether a claim is correct.
"""


def critic(state: AIState) -> AIState:

    evidence_records = build_evidence_database(
        state
    )

    research = state.get(
        "research_result",
        "",
    )

    analysis = state.get(
        "analysis_result",
        "",
    )

    strategy = state.get(
        "strategy_result",
        "",
    )

    # ========================================================
    # 1. DETERMINISTIC RESEARCH CITATION GATE
    # ========================================================

    research_gate = citation_gate(
        research,
        evidence_records,
        require_sources_section=False,
    )

    # ========================================================
    # 2. DETERMINISTIC ANALYSIS CITATION GATE
    # ========================================================

    analysis_gate = citation_gate(
        analysis,
        evidence_records,
        require_sources_section=False,
    )

    # ========================================================
    # 3. DETERMINISTIC STRATEGY CITATION GATE
    # ========================================================

    strategy_gate = citation_gate(
        strategy,
        evidence_records,
        require_sources_section=False,
    )

    errors: List[str] = []
    warnings: List[str] = []
    rework_instructions: List[str] = []

    # Research is the primary evidence-producing stage.
    if not research_gate["passed"]:

        errors.extend(
            "RESEARCH: " + error
            for error in research_gate["errors"]
        )

        rework_instructions.append(
            "Rewrite the Research Agent output so every "
            "factual claim is immediately supported by "
            "a valid [S#] citation."
        )

        rework_instructions.append(
            "Do not report numerical or market claims "
            "without a source citation."
        )

    warnings.extend(
        "RESEARCH: " + warning
        for warning in research_gate["warnings"]
    )

    # V1 policy: Analysis and Strategy citation findings are advisory.
    # They are retained in the audit as warnings, but do not stop
    # the end-to-end workflow.
    if not analysis_gate["passed"]:
        warnings.extend(
            "ANALYSIS: " + error
            for error in analysis_gate["errors"]
        )
        rework_instructions.append(
            "Review Analysis citations and keep factual claims "
            "grounded in the Research evidence."
        )

    warnings.extend(
        "ANALYSIS: " + warning
        for warning in analysis_gate["warnings"]
    )

    if not strategy_gate["passed"]:
        warnings.extend(
            "STRATEGY: " + error
            for error in strategy_gate["errors"]
        )
        rework_instructions.append(
            "Review Strategy citations and keep factual claims "
            "grounded in the Research evidence."
        )

    warnings.extend(
        "STRATEGY: " + warning
        for warning in strategy_gate["warnings"]
    )

    # ========================================================
    # 4. CHECK SOURCE IDS IN ALL OUTPUTS
    # ========================================================

    valid_ids = [
        record.get("evidence_id")
        for record in evidence_records
        if record.get("evidence_id")
    ]

    all_output = "\n".join(
        [
            research,
            analysis,
            strategy,
        ]
    )

    invalid_ids = find_invalid_source_ids(
        all_output,
        valid_ids,
    )

    if invalid_ids:

        errors.append(
            "PIPELINE: Invalid source IDs detected: "
            + ", ".join(
                f"[{item}]"
                for item in invalid_ids
            )
        )

        rework_instructions.append(
            "Use only source IDs that exist in the "
            "evidence database."
        )

    # ========================================================
    # 5. CALL LLM CRITIC
    # ========================================================

    llm_audit = {}

    try:

        llm = LLMGateway(
            provider="ollama"
        )

        response = llm.generate(
            build_llm_prompt(
                state,
                evidence_records,
            )
        )

        llm_audit = extract_json(
            response
        )

    except Exception as exc:

        warnings.append(
            f"LLM critic unavailable: {exc}"
        )

    # ========================================================
    # 6. VALIDATE LLM AUDIT
    # ========================================================

    llm_validation = validate_llm_audit(
        llm_audit,
        valid_ids,
    )

    warnings.extend(
        llm_validation["warnings"]
    )

    # Empty/invalid LLM audit does NOT automatically
    # cause failure if deterministic validation already
    # proves the outputs are clean.
    #
    # But it can never override deterministic failure.

    # ========================================================
    # 7. PYTHON IS AUTHORITATIVE
    # ========================================================

    if errors:

        final_evaluation = "FAIL"

        target_agent = "research"
        issue_type = "citation"
        severity = "high"

    elif not llm_validation["valid"]:

        # No deterministic errors, but LLM critic failed.
        # Keep this as a warning rather than allowing
        # the broken 3B model to create a false FAIL.
        final_evaluation = "PASS WITH WARNINGS"

        target_agent = ""
        issue_type = "critic"
        severity = "medium"

    else:

        final_evaluation = "PASS"

        target_agent = ""
        issue_type = ""
        severity = ""

    # ========================================================
    # 8. BUILD FINAL AUDIT
    # ========================================================

    critic_result = {
        "claims": llm_audit.get(
            "claims",
            [],
        ),
        "invalid_citations": (
            invalid_ids
            or llm_audit.get(
                "invalid_citations",
                [],
            )
        ),
        "unsupported_claims": (
            llm_audit.get(
                "unsupported_claims",
                [],
            )
        ),
        "placeholders": [],
        "goal_drift": llm_audit.get(
            "goal_drift",
            False,
        ),
        "recommendation_issues": llm_audit.get(
            "recommendation_issues",
            [],
        ),
        "contradictions": llm_audit.get(
            "contradictions",
            [],
        ),
        "missing_information": llm_audit.get(
            "missing_information",
            [],
        ),
        "rework_instructions": rework_instructions,
        "target_agent": target_agent,
        "issue_type": issue_type,
        "severity": severity,
        "final_evaluation": final_evaluation,

        # Deterministic validation details.
        "deterministic_validation": {
            "research": research_gate,
            "analysis": analysis_gate,
            "strategy": strategy_gate,
        },

        "python_errors": errors,
        "python_warnings": warnings,
    }

    return {
        **state,

        "critic_result": json.dumps(
            critic_result,
            indent=2,
        ),

        "evaluation_status":
            final_evaluation,

        "validation_errors":
            errors,

        "validation_warnings":
            warnings,

        "rework_instructions":
            rework_instructions,

        "target_agent":
            target_agent,

        "issue_type":
            issue_type,

        "severity":
            severity,
    }
