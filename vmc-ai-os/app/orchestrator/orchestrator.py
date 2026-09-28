from typing import Dict, Any
from app.orchestrator.state import AIState

from langgraph.graph import StateGraph, START, END


from app.agents.research_agent import research_agent
from app.agents.analysis_agent import analysis_agent
from app.agents.strategy_agent import strategy_agent

from app.evaluation.critic import critic
from app.evaluation.pipeline_gate import (
    research_gate,
    analysis_gate,
    strategy_gate,
)

from app.output.aggregator import aggregator


# ============================================================
# PLANNER
# ============================================================

def planner(state: AIState) -> AIState:

    goal = state.get(
        "user_goal",
        "",
    )

    return {
        **state,

        "tasks": [
            {
                "agent": "research",
                "task": (
                    "Research the user's goal using "
                    "available evidence."
                ),
            },
            {
                "agent": "analysis",
                "task": (
                    "Analyze the research findings."
                ),
            },
            {
                "agent": "strategy",
                "task": (
                    "Develop strategy based on "
                    "the research and analysis."
                ),
            },
        ],

        "revision_count": 0,

        "max_revisions": 1,

        "validation_errors": [],

        "validation_warnings": [],

        "rework_instructions": [],

        "target_agent": "",

        "issue_type": "",

        "severity": "",

        "evaluation_status": "RUNNING",

        "pipeline_status": "RUNNING",

        "errors": [],
    }


# ============================================================
# APPLY GATE RESULT
# ============================================================

def apply_gate_result(
    state: AIState,
    gate_result: dict,
) -> AIState:

    stage = gate_result.get(
        "stage",
        "",
    )

    errors = list(
        gate_result.get(
            "errors",
            [],
        )
    )

    warnings = list(
        gate_result.get(
            "warnings",
            [],
        )
    )

    uncited_claims = list(
        gate_result.get(
            "uncited_claims",
            [],
        )
    )

    invalid_source_ids = list(
        gate_result.get(
            "invalid_source_ids",
            [],
        )
    )

    rework_instructions = []

    # --------------------------------------------------------
    # GENERAL VALIDATION ERRORS
    # --------------------------------------------------------

    rework_instructions.extend(
        errors
    )

    # --------------------------------------------------------
    # ACTUAL UNCITED CLAIMS
    # --------------------------------------------------------

    if uncited_claims:

        rework_instructions.append(
            "The following factual-looking claim(s) "
            "were detected without citations:"
        )

        for claim in uncited_claims:

            rework_instructions.append(
                f"UNCITED CLAIM: {claim}"
            )

        rework_instructions.extend(
            [
                "Correct each uncited claim.",
                "Do not invent a citation.",
                "Do not attach a source merely to "
                "satisfy the validator.",
                "If the available evidence does not "
                "support the claim, remove it or write "
                "\"Insufficient evidence.\"",
            ]
        )

    # --------------------------------------------------------
    # INVALID SOURCE IDs
    # --------------------------------------------------------

    if invalid_source_ids:

        rework_instructions.append(
            "The following invalid source ID(s) "
            "were used:"
        )

        for source_id in invalid_source_ids:

            rework_instructions.append(
                f"INVALID SOURCE ID: {source_id}"
            )

        rework_instructions.extend(
            [
                "Use only source IDs supplied in "
                "the current evidence database.",
                "Never invent source IDs.",
            ]
        )

    # --------------------------------------------------------
    # WARNINGS
    # --------------------------------------------------------

    rework_instructions.extend(
        warnings
    )

    return {
        **state,

        f"{stage}_gate": gate_result,

        "validation_errors": errors,

        "validation_warnings": warnings,

        "target_agent": stage,

        "issue_type": (
            "validation_error"
            if errors
            else "validation_warning"
        ),

        "severity": (
            "error"
            if errors
            else "warning"
        ),

        "rework_instructions": (
            rework_instructions
        ),
    }


# ============================================================
# RESEARCH GATE
# ============================================================

def run_research_gate(
    state: AIState,
) -> AIState:

    result = research_gate(
        state
    )

    return apply_gate_result(
        state,
        result,
    )


# ============================================================
# ANALYSIS GATE
# ============================================================

def run_analysis_gate(
    state: AIState,
) -> AIState:

    result = analysis_gate(
        state
    )

    return apply_gate_result(
        state,
        result,
    )


# ============================================================
# STRATEGY GATE
# ============================================================

def run_strategy_gate(
    state: AIState,
) -> AIState:

    result = strategy_gate(
        state
    )

    return apply_gate_result(
        state,
        result,
    )


# ============================================================
# RESEARCH ROUTER
# ============================================================

def route_after_research_gate(
    state: AIState,
) -> str:

    gate = state.get(
        "research_gate",
        {},
    )

    # --------------------------------------------------------
    # RESEARCH PASSED
    # --------------------------------------------------------

    if gate.get("passed"):

        return "analysis"

    # --------------------------------------------------------
    # RESEARCH FAILED
    # --------------------------------------------------------

    revision_count = state.get(
        "revision_count",
        0,
    )

    max_revisions = state.get(
        "max_revisions",
        1,
    )

    # --------------------------------------------------------
    # ALLOW REWORK
    # --------------------------------------------------------

    if revision_count < max_revisions:

        return "rework"

    # --------------------------------------------------------
    # NO MORE REVISIONS
    # --------------------------------------------------------

    return "failed"


# ============================================================
# ANALYSIS ROUTER
# ============================================================

def route_after_analysis_gate(
    state: AIState,
) -> str:

    gate = state.get(
        "analysis_gate",
        {},
    )

    # Analysis passed.

    if gate.get("passed"):

        return "strategy"

    # Analysis failed.

    revision_count = state.get(
        "revision_count",
        0,
    )

    max_revisions = state.get(
        "max_revisions",
        1,
    )

    if revision_count < max_revisions:

        return "rework"

    return "failed"


# ============================================================
# STRATEGY ROUTER
# ============================================================

def route_after_strategy_gate(
    state: AIState,
) -> str:

    gate = state.get(
        "strategy_gate",
        {},
    )

    # Strategy passed.

    if gate.get("passed"):

        return "critic"

    # Strategy failed.

    revision_count = state.get(
        "revision_count",
        0,
    )

    max_revisions = state.get(
        "max_revisions",
        1,
    )

    if revision_count < max_revisions:

        return "rework"

    return "failed"


# ============================================================
# PREPARE REWORK
# ============================================================

def prepare_rework(
    state: AIState,
) -> AIState:

    revision_count = state.get(
        "revision_count",
        0,
    )

    return {
        **state,

        "revision_count": (
            revision_count + 1
        ),
    }


# ============================================================
# ROUTE REWORK AGENT
# ============================================================

def route_rework_agent(
    state: AIState,
) -> str:

    target_agent = state.get(
        "target_agent",
        "",
    )

    # --------------------------------------------------------
    # Explicit target from gate
    # --------------------------------------------------------

    if target_agent in {
        "research",
        "analysis",
        "strategy",
    }:

        return target_agent

    # --------------------------------------------------------
    # Fallback based on issue text
    # --------------------------------------------------------

    issue_text = " ".join(
        state.get(
            "rework_instructions",
            [],
        )
    ).lower()

    if "research" in issue_text:

        return "research"

    if "analysis" in issue_text:

        return "analysis"

    if "strategy" in issue_text:

        return "strategy"

    # Safe default.

    return "research"


# ============================================================
# PIPELINE FAILED
# ============================================================

def pipeline_failed(
    state: AIState,
) -> AIState:

    errors = list(
        state.get(
            "validation_errors",
            [],
        )
    )

    target_agent = state.get(
        "target_agent",
        "unknown",
    )

    error_text = "\n".join(
        f"- {error}"
        for error in errors
    )

    final_result = f"""
VMC AI OPERATING SYSTEM
=======================

PIPELINE FAILED

Failed stage: {target_agent}

Validation errors:

{error_text}
"""

    return {
        **state,

        "pipeline_status": "FAIL",

        "evaluation_status": "FAIL",

        "final_result": final_result,
    }


# ============================================================
# BUILD GRAPH
# ============================================================

def build_graph():

    graph = StateGraph(
        AIState
    )

    # --------------------------------------------------------
    # ADD NODES
    # --------------------------------------------------------

    graph.add_node(
        "planner",
        planner,
    )

    graph.add_node(
        "research",
        research_agent,
    )

    graph.add_node(
        "research_gate",
        run_research_gate,
    )

    graph.add_node(
        "analysis",
        analysis_agent,
    )

    graph.add_node(
        "analysis_gate",
        run_analysis_gate,
    )

    graph.add_node(
        "strategy",
        strategy_agent,
    )

    graph.add_node(
        "strategy_gate",
        run_strategy_gate,
    )

    graph.add_node(
        "prepare_rework",
        prepare_rework,
    )

    graph.add_node(
        "critic",
        critic,
    )

    graph.add_node(
        "aggregator",
        aggregator,
    )

    graph.add_node(
        "pipeline_failed",
        pipeline_failed,
    )

    # --------------------------------------------------------
    # START
    # --------------------------------------------------------

    graph.add_edge(
        START,
        "planner",
    )

    # --------------------------------------------------------
    # PLANNER → RESEARCH
    # --------------------------------------------------------

    graph.add_edge(
        "planner",
        "research",
    )

    # --------------------------------------------------------
    # RESEARCH → RESEARCH GATE
    # --------------------------------------------------------

    graph.add_edge(
        "research",
        "research_gate",
    )

    # --------------------------------------------------------
    # RESEARCH GATE ROUTING
    # --------------------------------------------------------

    graph.add_conditional_edges(
        "research_gate",
        route_after_research_gate,
        {
            "analysis": "analysis",
            "rework": "prepare_rework",
            "failed": "pipeline_failed",
        },
    )

    # --------------------------------------------------------
    # ANALYSIS → ANALYSIS GATE
    # --------------------------------------------------------

    graph.add_edge(
        "analysis",
        "analysis_gate",
    )

    # --------------------------------------------------------
    # ANALYSIS GATE ROUTING
    # --------------------------------------------------------

    graph.add_conditional_edges(
        "analysis_gate",
        route_after_analysis_gate,
        {
            "strategy": "strategy",
            "rework": "prepare_rework",
            "failed": "pipeline_failed",
        },
    )

    # --------------------------------------------------------
    # STRATEGY → STRATEGY GATE
    # --------------------------------------------------------

    graph.add_edge(
        "strategy",
        "strategy_gate",
    )

    # --------------------------------------------------------
    # STRATEGY GATE ROUTING
    # --------------------------------------------------------

    graph.add_conditional_edges(
        "strategy_gate",
        route_after_strategy_gate,
        {
            "critic": "critic",
            "rework": "prepare_rework",
            "failed": "pipeline_failed",
        },
    )

    # --------------------------------------------------------
    # PREPARE REWORK → TARGET AGENT
    # --------------------------------------------------------

    graph.add_conditional_edges(
        "prepare_rework",
        route_rework_agent,
        {
            "research": "research",
            "analysis": "analysis",
            "strategy": "strategy",
        },
    )

    # --------------------------------------------------------
    # CRITIC → AGGREGATOR
    # --------------------------------------------------------

    graph.add_edge(
        "critic",
        "aggregator",
    )

    # --------------------------------------------------------
    # AGGREGATOR → END
    # --------------------------------------------------------

    graph.add_edge(
        "aggregator",
        END,
    )

    # --------------------------------------------------------
    # PIPELINE FAILURE → END
    # --------------------------------------------------------

    graph.add_edge(
        "pipeline_failed",
        END,
    )

    return graph.compile()
