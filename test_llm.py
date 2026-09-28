from app.llm.gateway import LLMGateway

llm = LLMGateway(provider="ollama")

response = llm.generate(
    "You are the local intelligence engine of the VMC AI Operating System. "
    "Explain your role in exactly three short sentences."
)

print("\nLLM RESPONSE")
print("------------")
print(response)
