from app.agents.research_agent import research_agent
from app.agents.analysis_agent import analysis_agent
from app.agents.strategy_agent import strategy_agent
from app.evaluation.critic import critic


state = {
    "user_goal": "Research the market for AI training platforms in India"
}

state = research_agent(state)
state = analysis_agent(state)
state = strategy_agent(state)
state = critic(state)

print("\nCRITIC")
print("======")
print(state["critic_result"])
