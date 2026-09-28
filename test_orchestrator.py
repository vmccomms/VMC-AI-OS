from app.orchestrator.orchestrator import build_graph


graph = build_graph()

state = {
    "user_goal": "Research the market for AI training platforms in India",
    "revision_count": 0,
    "max_revisions": 1,
}

result = graph.invoke(state)

print("\n")
print("=" * 70)
print("VMC AI OPERATING SYSTEM")
print("=" * 70)

print("\nEVALUATION STATUS")
print("-----------------")
print(result.get("evaluation_status"))

print("\nREVISION COUNT")
print("--------------")
print(result.get("revision_count"))

print("\nFINAL RESULT")
print("=" * 70)
print(result["final_result"])
