from app.orchestrator.state import AIState
from app.llm.gateway import LLMGateway


def strategy_agent(state: AIState) -> AIState:
    goal = state["user_goal"]

    research = state.get(
        "research_result",
        "",
    )

    analysis = state.get(
        "analysis_result",
        "",
    )

    critic_feedback = state.get(
        "critic_result",
        "",
    )

    rework_instructions = state.get(
        "rework_instructions",
        [],
    )

    revision_count = state.get(
        "revision_count",
        0,
    )

    llm = LLMGateway(
        provider="ollama"
    )

    rework_text = "\n".join(
        f"- {item}"
        for item in rework_instructions
    )

    prompt = f"""
You are the Strategy Agent in the
VMC AI Operating System.

USER GOAL:
{goal}

RESEARCH:
{research}

ANALYSIS:
{analysis}

PREVIOUS CRITIC FEEDBACK:
{critic_feedback}

STRUCTURED REWORK INSTRUCTIONS:
{rework_text}

REVISION NUMBER:
{revision_count}

==================================================
TASK
==================================================

Develop a practical strategy based on the
research and analysis.

If structured rework instructions are present,
address EVERY applicable instruction.

Do not simply repeat the previous strategy.

==================================================
IMPORTANT RULES
==================================================

1. Do not invent external facts.

2. Do not invent statistics.

3. Do not invent market sizes.

4. Do not invent companies.

5. Do not invent regulations.

6. Proposed targets must be clearly labelled
   as PROPOSED TARGETS.

7. Do not convert assumptions into facts.

8. Preserve the original user goal.

9. Do not silently change:

   "AI training platforms"

   into:

   "AI training datasets"

10. If evidence is insufficient, explicitly state:

    "Insufficient evidence."

==================================================
OUTPUT
==================================================

1. Strategic Objective

2. Recommended Approach

3. Target Opportunities

4. Key Actions

5. Technology Considerations

6. Risks and Mitigation

7. Implementation Phases

8. Success Metrics

9. Information That Must Be Verified

10. Assumptions

Keep the strategy practical, structured and concise.
"""

    result = llm.generate(prompt)

    return {
        **state,
        "strategy_result": result,
    }
