from typing import List, Dict

from app.orchestrator.state import AIState
from app.llm.gateway import LLMGateway
from app.tools.web_search import web_search
from app.tools.evidence import filter_evidence


def build_queries(goal: str) -> List[str]:

    return [
        '"AI training platforms" India',
        '"AI learning platforms" India',
        '"AI education" India market',
        '"AI skills training" India jobs',
        '"AI courses" India students',
        '"AI training" India workforce',
    ]


def build_rework_context(state: AIState) -> str:
    """
    Build explicit instructions for a Research Agent rework attempt.

    The pipeline gate places validation feedback in:
        state["rework_instructions"]

    The previous research result is also supplied so the agent can
    preserve valid findings while correcting unsupported claims.
    """

    revision_count = state.get("revision_count", 0)
    rework_instructions = state.get("rework_instructions", [])
    previous_result = state.get("research_result", "")

    if not rework_instructions:
        return ""

    instructions_text = "\n".join(
        f"- {item}"
        for item in rework_instructions
    )

    return f"""
============================================================
REWORK REQUIRED
============================================================

This is Research revision attempt #{revision_count}.

The previous Research output FAILED validation.

Validation feedback:

{instructions_text}

You MUST correct these problems before producing the new
Research report.

IMPORTANT:

1. Do NOT invent citations.

2. Do NOT attach [S#] to a statement merely to satisfy
   the validator.

3. A citation may be used only when the corresponding
   evidence actually supports the statement.

4. Remove unsupported factual claims.

5. If an important statement cannot be supported by the
   available evidence, write:

Insufficient evidence.

6. Preserve valid evidence-supported findings where
   appropriate.

7. Do not introduce new facts from your own knowledge.

8. Do not invent companies, statistics, market sizes,
   dates, percentages, growth rates, employment figures,
   or other factual information.

============================================================
PREVIOUS RESEARCH OUTPUT
============================================================

{previous_result}

============================================================
REWORK OBJECTIVE
============================================================

Produce a corrected Research report that satisfies the
validation feedback and the citation rules below.
"""


def research_agent(state: AIState) -> AIState:

    goal = state["user_goal"]

    llm = LLMGateway(
        provider="ollama"
    )

    raw_results: List[Dict] = []

    # ========================================================
    # SEARCH
    # ========================================================

    queries = build_queries(goal)

    for query in queries:

        try:

            results = web_search(
                query=query,
                max_results=5,
            )

            for result in results:

                result["query"] = query

                raw_results.append(result)

        except Exception as exc:

            print(
                f"[Research Agent] Search failed "
                f"for '{query}': {exc}"
            )

    # ========================================================
    # EVIDENCE FILTER
    # ========================================================

    evidence_records = filter_evidence(
        goal=goal,
        raw_results=raw_results,
    )

    # Keep the evidence database manageable.
    evidence_records = evidence_records[:12]

    # Assign stable source IDs.
    for index, record in enumerate(
        evidence_records,
        start=1,
    ):

        record["evidence_id"] = f"S{index}"

    # ========================================================
    # BUILD EVIDENCE DATABASE
    # ========================================================

    evidence_parts = []

    for record in evidence_records:

        evidence_parts.append(
            f"""
============================================================
SOURCE [{record["evidence_id"]}]
============================================================

TITLE:
{record["source_title"]}

URL:
{record["url"]}

EVIDENCE:
{record["evidence"]}
"""
        )

    evidence_text = "\n".join(
        evidence_parts
    )

    if not evidence_text:

        evidence_text = (
            "NO RELEVANT EVIDENCE FOUND."
        )

    valid_ids = [
        record["evidence_id"]
        for record in evidence_records
    ]

    valid_ids_text = ", ".join(
        f"[{item}]"
        for item in valid_ids
    )

    # ========================================================
    # REWORK CONTEXT
    # ========================================================

    rework_context = build_rework_context(
        state
    )

    # ========================================================
    # RESEARCH PROMPT
    # ========================================================

    prompt = f"""
You are the Research Agent in the
VMC AI Operating System.

============================================================
USER GOAL
============================================================

{goal}

{rework_context}

============================================================
VALID SOURCE IDs
============================================================

{valid_ids_text}

SOURCE ID SAFETY RULE:

The ONLY citation IDs you are allowed to write are exactly the IDs listed above.

Do not create, guess, continue, or increment the numbering.

For example, if the list ends at [S12], [S13] and [S14] are INVALID and MUST NOT appear anywhere in your answer.

Before writing each citation, verify that the ID appears in the VALID SOURCE IDs list.

If you cannot identify a valid source ID for a factual claim, do not cite it with a guessed ID. Remove the claim or write:
"Insufficient evidence in the supplied sources."

============================================================
EVIDENCE DATABASE
============================================================

{evidence_text}

============================================================
YOUR TASK
============================================================

Produce a research report answering ONLY the user goal.

The topic is:

AI TRAINING PLATFORMS IN INDIA

CITATION REQUIREMENTS:

1. Every factual or externally verifiable statement must have at least one [S#] citation.
2. This includes the Executive Summary, Key Findings, Market / Industry Information, Demand / Opportunities, Competitors / Existing Solutions, Technologies / Capabilities, Risks and Challenges, Important Gaps, and Claims Requiring Verification sections.
3. Do not repeat a factual conclusion in the Executive Summary without its citation.
4. If a statement cannot be supported by the supplied evidence, either:
   - remove the statement, or
   - write "Insufficient evidence in the supplied sources."
5. Statements describing the structure or contents of this report do not require citations.
6. Never invent a citation. Only use source IDs present in the supplied evidence database.


IMPORTANT EVIDENCE DISCIPLINE:

If the supplied evidence does not identify specific competitors, technologies,
risks, challenges, gaps, or other requested information, do NOT create a
placeholder factual sentence.

For example, NEVER write:
"Some of the key players in the market include [no specific companies mentioned...]"

Instead write:
"Specific competitors were not identified in the supplied evidence."

Likewise, do not convert missing evidence into a factual claim.
Use "Specific information was not identified in the supplied evidence."
when appropriate.

============================================================
CRITICAL CITATION RULES
============================================================

These rules are mandatory.

RULE 1
Every factual claim MUST have an immediate [S#] citation.

Correct:

The online education market in India is expected
to grow by USD 8.53 billion from 2025 to 2029. [S9]

Incorrect:

The online education market in India is expected
to grow by USD 8.53 billion from 2025 to 2029.

RULE 2
Put the citation immediately after the claim.

Correct:

The market is expected to grow at a CAGR of 29%. [S9]

Incorrect:

The market is expected to grow at a CAGR of 29%.

Source: [S9]

RULE 3
Only use source IDs that exist in:

{valid_ids_text}

RULE 4
NEVER invent a source ID.

RULE 5
NEVER invent statistics.

RULE 6
NEVER invent market sizes.

RULE 7
NEVER invent companies.

RULE 8
NEVER invent dates.

RULE 9
NEVER invent percentages.

RULE 10
NEVER use outside knowledge.

RULE 11
If evidence does not support a factual claim,
write exactly:

Insufficient evidence.

RULE 12
Recommendations and interpretations must be
clearly labelled as such.

RULE 13
Do not present an interpretation as a FACT.

RULE 14
Keep AI TRAINING PLATFORMS distinct from:

- AI training datasets
- GPU infrastructure
- AI chips
- semiconductor manufacturing
- healthcare simulation
- defense training
- 3D printing
- metaverse
- unrelated AI products

RULE 15
Do not turn a general online education statistic
into an AI training platform statistic unless the
evidence explicitly supports that connection.

============================================================
IMPORTANT CITATION METHOD
============================================================

Before writing each factual sentence, ask:

"Which exact source supports this statement?"

Then immediately append its source ID.

For example:

The India online education market is projected to grow
by USD 8.53 billion from 2025 to 2029. [S9]

If no source supports the statement:

Insufficient evidence.

============================================================
REWORK SAFETY RULE
============================================================

If this is a rework attempt, the validation feedback above
takes priority.

Do NOT try to satisfy the validator by adding citations
without evidence.

The correct response to unsupported information is to:

- remove the claim, OR
- rewrite it as "Insufficient evidence."

Do not preserve an unsupported claim merely because it
appeared in the previous report.

============================================================
OUTPUT FORMAT
============================================================

## Executive Summary

The Executive Summary must contain citations for every factual claim. Do not write uncited market-size, growth, demand, competitor, technology, or trend statements.

Provide only evidence-supported findings.

## Key Findings

Use concise factual bullets.

Every factual bullet must have [S#].

## Market / Industry Information

Only include evidence directly relevant to
AI training platforms or clearly relevant online
AI education/training.

Every factual claim requires [S#].

## Competitors / Existing Solutions

Only name companies/platforms when supported
by the evidence.

Every company claim requires [S#].

If unavailable:

Insufficient evidence.

## Demand / Opportunities

Separate:

FACT:
... [S#]

from:

INTERPRETATION:
...

Facts require [S#].

Interpretations must not be presented as facts.

## Technologies / Capabilities

Only report evidence-supported technology information.

## Risks and Challenges

Only report evidence-supported risks.

## Important Gaps

Identify information that the evidence does not establish.

## Claims Requiring Verification

List claims that need additional evidence.

## Sources

List every source actually used in the report.

Use:

Use the exact source IDs from the VALID SOURCE IDs list.

Do not create additional source IDs.

List only sources actually cited in the report.


============================================================
FINAL CHECK BEFORE ANSWERING
============================================================

Check every factual sentence.

If it contains:

- a number
- percentage
- market size
- funding
- CAGR
- jobs
- companies
- growth
- employment
- dates
- named organizations
- factual market statements

it MUST contain [S#].

Do not output an uncited factual statement.

============================================================
FINAL REWORK CHECK
============================================================

Before returning the answer, specifically check:

1. Did I remove unsupported factual claims?

2. Does every factual claim have a valid [S#]?

3. Does every [S#] exist in the supplied source list?

4. Does the cited evidence actually support the claim?

5. Did I avoid adding outside knowledge?

6. Did I avoid inventing statistics?

7. Did I avoid inventing companies or market information?

8. If evidence was insufficient, did I use:
   "Insufficient evidence."


For every section, use only information actually supported by the supplied
evidence. If a section has no supported information, explicitly state that
the supplied evidence does not identify specific information for that section.
Do not invent, infer, or create placeholder claims.

Return ONLY the research report.
"""

    # ========================================================
    # GENERATE RESEARCH
    # ========================================================

    result = llm.generate(
        prompt
    )

    return {
        **state,
        "research_result": result,
        "research_evidence": raw_results,
        "evidence_records": evidence_records,
    }
