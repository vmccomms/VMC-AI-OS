from typing import TypedDict, Optional, List, Dict
from urllib.parse import urlparse


class EvidenceRecord(TypedDict, total=False):
    evidence_id: str
    query: str
    source_title: str
    url: str
    domain: str
    evidence: str
    relevance: str
    support: str
    quality: str
    score: Optional[float]
    retrieved_at: str


def get_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def normalize_text(text: str) -> str:
    return " ".join(
        text.lower()
        .replace("-", " ")
        .replace("/", " ")
        .split()
    )


# -------------------------------------------------------------
# POSITIVE TERMS
# -------------------------------------------------------------

AI_TERMS = {
    "ai",
    "artificial intelligence",
    "machine learning",
    "ml",
    "generative ai",
    "genai",
    "llm",
    "large language model",
    "deep learning",
    "ai engineer",
    "ai engineering",
    "ai skills",
    "ai workforce",
}


TRAINING_TERMS = {
    "training",
    "learning",
    "education",
    "course",
    "courses",
    "certification",
    "certifications",
    "skills",
    "skill",
    "workforce",
    "learner",
    "learners",
    "student",
    "students",
    "academy",
    "academic",
    "upskilling",
    "reskilling",
    "bootcamp",
    "curriculum",
    "institute",
    "platform",
    "online learning",
    "training platform",
    "learning platform",
}


INDIA_TERMS = {
    "india",
    "indian",
    "india's",
    "new delhi",
    "bengaluru",
    "bangalore",
    "hyderabad",
    "mumbai",
    "chennai",
    "pune",
}


# -------------------------------------------------------------
# STRONG NEGATIVE TERMS
# -------------------------------------------------------------

EXCLUSION_TERMS = {
    "chip manufacturing",
    "semiconductor manufacturing",
    "semiconductor fabrication",
    "chip fabrication",
    "gpu manufacturing",
    "gpu",
    "gpus",
    "hpc",
    "exascale",
    "processor",
    "processors",
    "3d printing",
    "3d printer",
    "anatomical",
    "anatomy",
    "medical simulator",
    "surgical simulator",
    "defense training",
    "military training",
    "metaverse",
    "consumer electronics",
    "smartphone",
    "smartphones",
    "smart air purifier",
    "air purifier",
}


def contains_any(text: str, terms: set) -> bool:
    text = normalize_text(text)

    for term in terms:
        if term in text:
            return True

    return False


def count_matches(text: str, terms: set) -> int:
    text = normalize_text(text)

    count = 0

    for term in terms:
        if term in text:
            count += 1

    return count


def topic_relevance(
    goal: str,
    title: str,
    snippet: str,
) -> float:

    goal_text = normalize_text(goal)
    result_text = normalize_text(
        f"{title} {snippet}"
    )

    goal_words = set(goal_text.split())
    result_words = set(result_text.split())

    if not goal_words:
        return 0.0

    overlap = goal_words.intersection(
        result_words
    )

    return len(overlap) / len(goal_words)


def source_quality(url: str) -> str:

    domain = get_domain(url)

    if not domain:
        return "unknown"

    trusted_domains = [
        ".gov",
        ".gov.in",
        ".edu",
        ".ac.in",
        "nasscom.in",
        "niti.gov.in",
        "who.int",
        "worldbank.org",
        "oecd.org",
        "ibef.org",
    ]

    for trusted in trusted_domains:
        if trusted in domain:
            return "high"

    return "standard"


def is_ai_training_goal(goal: str) -> bool:

    goal_text = normalize_text(goal)

    return (
        "ai training" in goal_text
        or "ai learning" in goal_text
        or "ai education" in goal_text
        or "training platform" in goal_text
        or "learning platform" in goal_text
        or "ai skills" in goal_text
        or "ai courses" in goal_text
    )


def passes_training_platform_filter(
    goal: str,
    title: str,
    snippet: str,
) -> bool:

    combined = normalize_text(
        f"{title} {snippet}"
    )

    # ---------------------------------------------------------
    # Must contain AI-related context
    # ---------------------------------------------------------

    has_ai = contains_any(
        combined,
        AI_TERMS,
    )

    if not has_ai:
        return False

    # ---------------------------------------------------------
    # Must contain training / education context
    # ---------------------------------------------------------

    has_training = contains_any(
        combined,
        TRAINING_TERMS,
    )

    if not has_training:
        return False

    # ---------------------------------------------------------
    # Must contain India context
    # ---------------------------------------------------------

    has_india = contains_any(
        combined,
        INDIA_TERMS,
    )

    if not has_india:
        return False

    # ---------------------------------------------------------
    # Reject strong unrelated topics
    # ---------------------------------------------------------

    negative_matches = count_matches(
        combined,
        EXCLUSION_TERMS,
    )

    if negative_matches > 0:
        return False

    return True


def filter_evidence(
    goal: str,
    raw_results: List[Dict],
) -> List[EvidenceRecord]:

    records: List[EvidenceRecord] = []

    seen_urls = set()

    training_goal = is_ai_training_goal(goal)

    for result in raw_results:

        url = result.get("url", "").strip()

        if not url:
            continue

        if url in seen_urls:
            continue

        seen_urls.add(url)

        title = result.get(
            "title",
            "",
        ).strip()

        snippet = result.get(
            "snippet",
            "",
        ).strip()

        if not title and not snippet:
            continue

        # -----------------------------------------------------
        # Specialized filter for AI training platform research
        # -----------------------------------------------------

        if training_goal:

            if not passes_training_platform_filter(
                goal,
                title,
                snippet,
            ):
                continue

        # -----------------------------------------------------
        # General relevance
        # -----------------------------------------------------

        relevance_score = topic_relevance(
            goal,
            title,
            snippet,
        )

        if relevance_score < 0.05:
            continue

        record: EvidenceRecord = {
            "evidence_id": f"S{len(records) + 1}",
            "query": result.get(
                "query",
                "",
            ),
            "source_title": title,
            "url": url,
            "domain": get_domain(url),
            "evidence": snippet,
            "relevance": f"{relevance_score:.3f}",
            "support": "unverified",
            "quality": source_quality(url),
            "score": result.get("score"),
            "retrieved_at": result.get(
                "retrieved_at",
                "",
            ),
        }

        records.append(record)

    return records


def normalize_evidence(
    raw_results: List[Dict],
) -> List[EvidenceRecord]:

    records = []
    seen_urls = set()

    for result in raw_results:

        url = result.get(
            "url",
            "",
        ).strip()

        if not url:
            continue

        if url in seen_urls:
            continue

        seen_urls.add(url)

        records.append({
            "evidence_id": f"S{len(records) + 1}",
            "query": result.get(
                "query",
                "",
            ),
            "source_title": result.get(
                "title",
                "",
            ),
            "url": url,
            "domain": get_domain(url),
            "evidence": result.get(
                "snippet",
                "",
            ),
            "relevance": "unverified",
            "support": "unverified",
            "quality": source_quality(url),
            "score": result.get("score"),
            "retrieved_at": result.get(
                "retrieved_at",
                "",
            ),
        })

    return records

# ============================================================
# CITATION & EVIDENCE GATE v1
# ============================================================

import re
from typing import Any


SOURCE_ID_PATTERN = re.compile(r"\[S(\d+)\]")

NUMERIC_CLAIM_PATTERN = re.compile(
    r"""
    (
        \b\d+(?:\.\d+)?\s*%
        |
        \b\d+(?:\.\d+)?\s*(?:million|billion|thousand|M|B|K)
        |
        \$\s*\d+(?:\.\d+)?
        |
        \b\d+(?:,\d{3})+(?:\.\d+)?
        |
        \b\d+(?:\.\d+)?\s*(?:USD|INR|rupees|dollars)
        |
        \bCAGR\b
        |
        \bmarket\s+(?:size|growth|share)
        |
        \bfunding\b
        |
        \brevenue\b
        |
        \bjobs?\b
        |
        \busers?\b
        |
        \bcompanies\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

FACTUAL_CLAIM_STARTERS = (
    "according to",
    "the report",
    "the study",
    "the market",
    "the sector",
    "the industry",
    "the company",
    "companies",
    "india",
    "indian",
    "research",
    "funding",
    "employment",
    "growing",
    "growth",
    "there are",
    "there were",
    "there is",
    "there was",
    "lists",
    "reported",
    "shows",
    "states",
)


def extract_source_ids(text: str) -> List[str]:
    """
    Extract source IDs such as [S1], [S2], etc.
    """
    if not text:
        return []

    return [
        f"S{number}"
        for number in SOURCE_ID_PATTERN.findall(text)
    ]


def find_invalid_source_ids(
    text: str,
    valid_ids: List[str],
) -> List[str]:
    """
    Return citation IDs used in text that do not exist
    in the evidence database.
    """
    used = extract_source_ids(text)

    return sorted(
        set(used) - set(valid_ids),
        key=lambda value: int(value[1:]),
    )


def _clean_line(line: str) -> str:
    """
    Normalize a line for heuristic inspection.
    """
    line = line.strip()

    # Remove markdown bullets/numbers.
    line = re.sub(
        r"^[\-\*\u2022]\s*",
        "",
        line,
    )

    line = re.sub(
        r"^\d+[\.\)]\s*",
        "",
        line,
    )

    return line.strip()


def _looks_like_factual_claim(line: str) -> bool:
    """
    Conservative detector for factual claims.

    The goal is NOT to require citations on every sentence.
    It focuses on externally verifiable claims, especially
    numbers, market facts, companies, funding, employment,
    growth and statistics.
    """

    if not line:
        return False

    text = _clean_line(line)

    if len(text) < 20:
        return False

    lower = text.lower()

    # --------------------------------------------------------
    # Things that should NEVER require a source citation
    # --------------------------------------------------------

    if text.startswith("#"):
        return False

    if text.endswith(":") and len(text) < 100:
        return False

    if lower.endswith("?"):
        return False

    ignored_prefixes = (
        "user goal",
        "preserved user goal",
        "insufficient evidence",
        "assumption:",
        "assumptions:",
        "interpretation:",
        "recommendation:",
        "recommendations:",
        "recommended approach",
        "recommended areas",
        "key actions",
        "implementation phases",
        "success metrics",
        "information gaps",
        "information that must be verified",
        "claims requiring verification",
        "risks and mitigation",
        "technology considerations",
        "strategic objective",
        "target opportunities",
        "note:",
    )

    if lower.startswith(ignored_prefixes):
        return False

    # A line explicitly saying evidence is insufficient
    # is not itself a factual claim.
    if "insufficient evidence" in lower:
        return False

    # --------------------------------------------------------
    # Recommendations / proposed actions
    # --------------------------------------------------------

    recommendation_prefixes = (
        "recommend",
        "consider",
        "should",
        "could",
        "may want",
        "propose",
        "develop",
        "establish",
        "conduct",
        "implement",
        "create",
        "launch",
        "monitor",
        "analyze",
        "investigate",
        "partner",
        "build",
        "identify",
        "evaluate",
        "review",
        "use ",
        "utilize ",
        "leverage ",
        "focus on",
    )

    if lower.startswith(recommendation_prefixes):
        return False

    # --------------------------------------------------------
    # Explicit numerical / statistical claims
    # --------------------------------------------------------

    if NUMERIC_CLAIM_PATTERN.search(text):
        return True

    # --------------------------------------------------------
    # Explicit factual attribution
    # --------------------------------------------------------

    factual_prefixes = (
        "according to",
        "the report states",
        "the study states",
        "the report found",
        "the study found",
        "data shows",
        "data indicates",
        "research shows",
        "research indicates",
        "the market is",
        "the market was",
        "the sector is",
        "the sector was",
        "the industry is",
        "the industry was",
        "the company is",
        "the company was",
        "companies have",
        "companies raised",
        "india has",
        "india had",
        "india's",
    )

    if lower.startswith(factual_prefixes):
        return True

    # --------------------------------------------------------
    # Strong factual verbs
    # --------------------------------------------------------

    factual_verbs = (
        " is ",
        " are ",
        " was ",
        " were ",
        " has ",
        " have ",
        " had ",
        " reached ",
        "increased ",
        "decreased ",
        "grew ",
        "raised ",
        "reported ",
        "listed ",
        "employs ",
        "serves ",
        "generated ",
        "recorded ",
        "received ",
        "grew by ",
        "declined ",
    )

    padded = f" {lower} "

    if any(
        verb in padded
        for verb in factual_verbs
    ):
        return True

    return False

def find_uncited_claim_lines(
    text: str,
) -> List[str]:
    """
    Find factual-looking lines without an [S#] citation.

    Report-status statements such as:
    "No specific competitors were mentioned in the report."
    describe the contents of the generated report rather than
    making an external factual claim, so they are not citation-required.
    """
    if not text:
        return []

    problems = []

    report_status_prefixes = (
        "no specific ",
        "no particular ",
        "no clear ",
        "no additional ",
        "no further ",
        "nothing specific ",
        "not specifically ",
        "no claims ",
        "no evidence ",
        "no information ",
    )

    report_status_phrases = (
        "were mentioned in the report",
        "was mentioned in the report",
        "were identified in the report",
        "was identified in the report",
        "were provided in the report",
        "was provided in the report",
        "were found in the report",
        "was found in the report",
        "were discussed in the report",
        "was discussed in the report",
        "were included in the report",
        "was included in the report",
        "require additional evidence",
        "requires additional evidence",
    )

    for raw_line in text.splitlines():
        line = _clean_line(raw_line)

        if not line:
            continue

        lower = line.lower().strip()

        # These statements describe what the report does or does not
        # contain. They are not external factual claims requiring [S#].
        if (
            lower.startswith(report_status_prefixes)
            and any(
                phrase in lower
                for phrase in report_status_phrases
            )
        ):
            continue

        if not _looks_like_factual_claim(line):
            continue

        if SOURCE_ID_PATTERN.search(line):
            continue

        problems.append(line)

    return problems

def find_unused_source_ids(
    text: str,
    valid_ids: List[str],
) -> List[str]:
    """
    Sources supplied to the model but never cited in its output.
    """
    used = set(extract_source_ids(text))

    return [
        source_id
        for source_id in valid_ids
        if source_id not in used
    ]


def citation_gate(
    text: str,
    evidence_records: List[Dict[str, Any]],
    require_sources_section: bool = False,
) -> Dict[str, Any]:
    """
    Deterministic validation of citations.

    Python, not the LLM, is authoritative here.
    """

    valid_ids = [
        record.get("evidence_id")
        for record in evidence_records
        if record.get("evidence_id")
    ]

    errors: List[str] = []
    warnings: List[str] = []

    if not text:
        errors.append("Output is empty.")
        return {
            "passed": False,
            "errors": errors,
            "warnings": warnings,
            "used_source_ids": [],
            "invalid_source_ids": [],
            "uncited_claims": [],
            "unused_source_ids": valid_ids,
        }

    used_source_ids = extract_source_ids(text)

    invalid_source_ids = find_invalid_source_ids(
        text,
        valid_ids,
    )

    if invalid_source_ids:
        errors.append(
            "Invalid source IDs used: "
            + ", ".join(
                f"[{item}]"
                for item in invalid_source_ids
            )
        )

    uncited_claims = find_uncited_claim_lines(text)

    if uncited_claims:
        errors.append(
            f"Found {len(uncited_claims)} "
            "factual-looking claim(s) without citations."
        )

    unused_source_ids = find_unused_source_ids(
        text,
        valid_ids,
    )

    if unused_source_ids:
        warnings.append(
            "Evidence sources supplied but not cited: "
            + ", ".join(
                f"[{item}]"
                for item in unused_source_ids
            )
        )

    if require_sources_section:
        lower = text.lower()

        source_headers = (
            "sources",
            "references",
            "source list",
            "evidence",
        )

        if not any(
            header in lower
            for header in source_headers
        ):
            errors.append(
                "Required Sources/Evidence section is missing."
            )

    return {
        "passed": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "used_source_ids": used_source_ids,
        "invalid_source_ids": invalid_source_ids,
        "uncited_claims": uncited_claims,
        "unused_source_ids": unused_source_ids,
    }
