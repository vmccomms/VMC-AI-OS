from typing import TypedDict, List, Dict, Any


class AIState(TypedDict, total=False):
    user_goal: str
    tasks: List[Dict[str, Any]]

    research_result: str
    research_evidence: List[Dict[str, Any]]
    evidence_records: List[Dict[str, Any]]

    analysis_result: str
    strategy_result: str

    research_gate: Dict[str, Any]
    analysis_gate: Dict[str, Any]
    strategy_gate: Dict[str, Any]

    critic_result: str
    evaluation_status: str
    pipeline_status: str


    validation_errors: List[str]
    validation_warnings: List[str]
    rework_instructions: List[str]

    target_agent: str
    issue_type: str
    severity: str

    revision_count: int
    max_revisions: int

    final_result: str
    errors: List[str]
