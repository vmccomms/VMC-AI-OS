from app.orchestrator.state import AIState


def aggregator(state: AIState) -> AIState:
    """
    Combine the outputs from all agents into a final result.
    """

    final_result = f"""
VMC AI OPERATING SYSTEM
=======================

USER GOAL
--------
{state.get("user_goal", "")}

RESEARCH
--------
{state.get("research_result", "")}

ANALYSIS
--------
{state.get("analysis_result", "")}

STRATEGY
--------
{state.get("strategy_result", "")}

EVALUATION
----------
{state.get("critic_result", "")}
"""

    return {
        **state,
        "final_result": final_result.strip(),
    }
