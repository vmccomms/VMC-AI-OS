from typing import List, Dict, Any

from app.orchestrator.state import AIState
from app.llm.gateway import LLMGateway


def _build_evidence_context(
    evidence_records: List[Dict[str, Any]],
) -> tuple[str, List[str]]:
    """
    Build the evidence database supplied to the Analysis Agent.
    """

    evidence_parts = []
    valid_source_ids = []

    for record in evidence_records:
        evidence_id = record.get("evidence_id", "")

        if not evidence_id:
            continue

        valid_source_ids.append(evidence_id)

        evidence_parts.append(
            f"""
============================================================
SOURCE [{evidence_id}]
============================================================

TITLE:
{record.get("source_title", "")}

URL:
{record.get("url", "")}

EVIDENCE:
{record.get("evidence", "")}
"""
        )

    if not evidence_parts:
        return (
            "NO EVIDENCE RECORDS WERE SUPPLIED.",
            [],
        )

    return (
        "\n".join(evidence_parts),
        valid_source_ids,
    )


def _build_rework_text(
    rework_instructions: List[str],
) -> str:
    """
    Convert structured rework instructions into prompt text.
    """

    if not rework_instructions:
        return (
            "No specific rework instructions were supplied."
        )

    return "\n".join(
        f"- {item}"
        for item in rework_instructions
    )


def analysis_agent(state: AIState) -> AIState:
    """
    Evidence-bound Analysis Agent.

    The Research Agent is responsible for discovering facts.

    The Analysis Agent is responsible for interpreting those
    facts without introducing new external information.

    A second pass repairs/verifies citations before returning
    the analysis to the deterministic pipeline gate.
    """

    goal = state.get(
        "user_goal",
        "",
    )

    research = state.get(
        "research_result",
        "",
    )

    evidence_records: List[Dict[str, Any]] = state.get(
        "evidence_records",
        [],
    )

    rework_instructions = state.get(
        "rework_instructions",
        [],
    )

    revision_count = state.get(
        "revision_count",
        0,
    )

    # ------------------------------------------------------------
    # Evidence
    # ------------------------------------------------------------

    evidence_text, valid_source_ids = (
        _build_evidence_context(
            evidence_records
        )
    )

    valid_ids_text = ", ".join(
        f"[{source_id}]"
        for source_id in valid_source_ids
    )

    if not valid_ids_text:
        valid_ids_text = "NONE"

    # ------------------------------------------------------------
    # Rework
    # ------------------------------------------------------------

    rework_text = _build_rework_text(
        rework_instructions
    )

    # ------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------

    llm = LLMGateway(
        provider="ollama"
    )

    # ============================================================
    # PASS 1
    # ============================================================

    analysis_prompt = f"""
You are the ANALYSIS AGENT in the VMC AI OPERATING SYSTEM.

Your role is to analyze an existing Research Agent report.

IMPORTANT:

The Research Agent is responsible for discovering facts.

You are NOT a research agent.

You MUST NOT introduce new external facts.

You MUST NOT perform new research.

You MUST NOT fill evidence gaps using your general knowledge.

Your job is to interpret and organize information that already
exists in the supplied Research Report and Evidence Database.

============================================================
USER GOAL
============================================================

{goal}

============================================================
RESEARCH REPORT
============================================================

{research}

============================================================
EVIDENCE DATABASE
============================================================

{evidence_text}

============================================================
VALID SOURCE IDS
============================================================

ONLY these source IDs exist:

{valid_ids_text}

============================================================
REWORK INSTRUCTIONS
============================================================

{rework_text}

============================================================
REVISION NUMBER
============================================================

{revision_count}

============================================================
MOST IMPORTANT RULE
============================================================

DO NOT INTRODUCE NEW FACTS.

You may:

- summarize the research
- organize findings
- compare findings already present
- identify relationships explicitly supported by the research
- identify information gaps
- identify limitations
- identify implications that directly follow from supplied evidence

You may NOT:

- add new statistics
- add new market sizes
- add new forecasts
- add new percentages
- add new companies
- add new competitors
- add new dates
- add new technologies
- add new employment figures
- add new investment figures
- add new market claims
- add new industry claims

If the Research Report does not support a point, write:

Insufficient evidence.

============================================================
CITATION RULE
============================================================

Every factual claim carried into the Analysis Report must retain
or receive its supporting [S#] citation.

Example:

Research says:

The market is expected to grow by 26.65%. [S1]

Analysis may say:

The research indicates a projected growth rate of 26.65%. [S1]

Analysis MUST NOT say:

The market is expected to grow rapidly.

unless that statement is supported by the supplied evidence.

============================================================
SOURCE ID SAFETY
============================================================

Only these IDs exist:

{valid_ids_text}

NEVER create another ID.

If the valid list ends at [S12], then [S13] DOES NOT EXIST.

Do not guess source IDs.

============================================================
GOAL PRESERVATION
============================================================

The original user goal is:

{goal}

Do not silently change the subject.

Do not replace the user's goal with a narrower or different
market unless the Research Report itself explicitly establishes
that scope.

============================================================
OUTPUT
============================================================

Produce exactly these sections:

1. Executive Summary

2. Key Findings

3. Market / Industry Information

4. Competitors / Existing Solutions

5. Demand / Opportunities

6. Technologies / Capabilities

7. Risks and Challenges

8. Important Gaps

9. Claims Requiring Verification

10. Sources

============================================================
SECTION GUIDANCE
============================================================

EXECUTIVE SUMMARY

Summarize the most important supported findings.

Do not introduce new facts.

------------------------------------------------------------

KEY FINDINGS

Use findings already established in the Research Report.

Each factual finding must have its supporting citation.

------------------------------------------------------------

MARKET / INDUSTRY INFORMATION

Only include market information already present in the
Research Report and supported by the evidence.

------------------------------------------------------------

COMPETITORS / EXISTING SOLUTIONS

Only mention competitors explicitly supported by the
Research Report/evidence.

If none are supported:

Specific competitors were not identified in the supplied evidence.

------------------------------------------------------------

DEMAND / OPPORTUNITIES

Only identify opportunities that can be directly supported
by the research.

Do not invent new opportunities from general knowledge.

------------------------------------------------------------

TECHNOLOGIES / CAPABILITIES

Only use technologies and capabilities already present in
the Research Report/evidence.

------------------------------------------------------------

RISKS AND CHALLENGES

Only use risks and challenges already supported by the
research.

------------------------------------------------------------

IMPORTANT GAPS

Identify genuine information gaps visible in the supplied
research.

Do not turn assumptions into market facts.

------------------------------------------------------------

CLAIMS REQUIRING VERIFICATION

Identify claims from the Research Report that are not adequately
supported by the supplied evidence.

------------------------------------------------------------

SOURCES

List only source IDs actually cited in the report.

Do not automatically list all sources.

============================================================
FINAL SELF-CHECK
============================================================

Before returning the report, verify:

1. Did I preserve the user goal?

2. Did I use only the Research Report and supplied evidence?

3. Did I introduce any new factual information?

4. Did I introduce any new statistics?

5. Did I introduce any new companies?

6. Did I introduce any new market figures?

7. Did I introduce any new dates?

8. Does every factual statement have [S#]?

9. Does every [S#] exist in the valid source list?

10. Did I avoid [S13], [S14], etc. when they do not exist?

11. Did I mark unsupported information as:

Insufficient evidence.

12. Did I keep the Analysis focused on interpretation rather
than new research?

Return ONLY the completed Analysis Report.
"""

    first_pass = llm.generate(
        analysis_prompt
    )

    # ============================================================
    # PASS 2 — CITATION / EVIDENCE REPAIR
    # ============================================================

    citation_prompt = f"""
You are the FINAL EVIDENCE REVIEWER in the VMC AI OPERATING SYSTEM.

Review the Analysis Report below.

Your task is to make sure the report contains ONLY information
supported by the Research Report and Evidence Database.

Do NOT perform new research.

Do NOT add new facts.

Do NOT expand the subject.

============================================================
USER GOAL
============================================================

{goal}

============================================================
RESEARCH REPORT
============================================================

{research}

============================================================
EVIDENCE DATABASE
============================================================

{evidence_text}

============================================================
VALID SOURCE IDS
============================================================

{valid_ids_text}

============================================================
ANALYSIS REPORT
============================================================

{first_pass}

============================================================
REVIEW RULES
============================================================

RULE 1
------

Every factual statement must have an immediate [S#] citation.

Correct:

The research reports a 26.65% CAGR. [S1]

Incorrect:

The research reports a 26.65% CAGR.

[S1]


RULE 2
------

The citation must actually support the statement.

Do not attach a random source merely to make validation pass.


RULE 3
------

Do NOT invent facts to improve the report.


RULE 4
------

Do NOT invent citations.


RULE 5
------

Only these source IDs are valid:

{valid_ids_text}


RULE 6
------

If a factual claim cannot be supported, replace it with:

Insufficient evidence.


RULE 7
------

Do not introduce:

- new market figures
- new percentages
- new forecasts
- new companies
- new competitors
- new dates
- new technologies
- new job figures
- new investment figures
- new external claims


RULE 8
------

Do not treat the Sources section as evidence.

A factual claim needs its citation next to the claim.


RULE 9
------

Do not change the user's research goal.

============================================================
GOAL DRIFT CHECK
============================================================

Original goal:

{goal}

If the Analysis Report introduces a different subject that is
not established by the Research Report, remove that material
or replace it with:

Insufficient evidence.

============================================================
FINAL OUTPUT
============================================================

Return the complete corrected Analysis Report.

Keep the same ten-section structure:

1. Executive Summary
2. Key Findings
3. Market / Industry Information
4. Competitors / Existing Solutions
5. Demand / Opportunities
6. Technologies / Capabilities
7. Risks and Challenges
8. Important Gaps
9. Claims Requiring Verification
10. Sources

Return ONLY the corrected Analysis Report.
"""

    final_analysis = llm.generate(
        citation_prompt
    )

    return {
        **state,
        "analysis_result": final_analysis,
    }
