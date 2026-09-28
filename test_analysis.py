from app.agents.research_agent import research_agent
from app.agents.analysis_agent import analysis_agent


state = {
    "user_goal": "Research the market for AI training platforms in India"
}

state = research_agent(state)
state = analysis_agent(state)

print("\nRESEARCH")
print("========")
print(state["research_result"])

print("\nANALYSIS")
print("========")
print(state["analysis_result"])
