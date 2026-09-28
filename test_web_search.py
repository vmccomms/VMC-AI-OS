from app.tools.web_search import web_search


query = "AI training platforms in India"

results = web_search(query)

print()
print("=" * 70)
print("VMC AI OS - WEB SEARCH TOOL TEST")
print("=" * 70)

for i, result in enumerate(results, 1):
    print(f"\nRESULT {i}")
    print("-" * 70)
    print("TITLE:", result["title"])
    print("URL:", result["url"])
    print("SNIPPET:", result["snippet"])
    print("RETRIEVED:", result["retrieved_at"])
