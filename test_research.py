from app.agents.research_agent import research_agent


state = {
    "user_goal": "Research the market for AI training platforms in India"
}

result = research_agent(state)

print("\nRESEARCH AGENT OUTPUT")
print("====================")
print(result["research_result"])
