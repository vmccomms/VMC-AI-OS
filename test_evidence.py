from app.tools.evidence import normalize_evidence


raw_results = [
    {
        "query": "AI training platforms India",
        "title": "Source One",
        "url": "https://example.com/article",
        "snippet": "Evidence from source one.",
        "score": 0.91,
        "retrieved_at": "2026-09-17T00:00:00+00:00",
    },
    {
        "query": "AI training platforms India",
        "title": "Source One Duplicate",
        "url": "https://example.com/article",
        "snippet": "Duplicate source.",
        "score": 0.85,
        "retrieved_at": "2026-09-17T00:00:00+00:00",
    },
    {
        "query": "AI training platforms India",
        "title": "Source Two",
        "url": "https://example.org/research",
        "snippet": "Evidence from source two.",
        "score": 0.88,
        "retrieved_at": "2026-09-17T00:00:00+00:00",
    },
]


records = normalize_evidence(raw_results)


print("=" * 60)
print("VMC AI OS - Evidence Normalization Test")
print("=" * 60)

print(f"\nInput results : {len(raw_results)}")
print(f"Evidence records: {len(records)}")

print("\nRecords")
print("-" * 60)

for record in records:
    print(
        f'{record["evidence_id"]} | '
        f'{record["source_title"]} | '
        f'{record["url"]}'
    )

print("\nExpected:")
print("Input results     : 3")
print("Evidence records  : 2")
print("IDs               : S1, S2")
