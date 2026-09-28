import os
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()


class LLMGateway:
    """
    Central gateway for all LLM calls.

    Supported providers:
        - ollama
        - openai

    Agents should use this gateway instead of
    directly depending on a specific LLM provider.
    """

    def __init__(
        self,
        provider: str = "ollama",
        model: Optional[str] = None,
    ):
        self.provider = provider

        if provider == "ollama":
            self.model = model or os.getenv(
                "OLLAMA_MODEL",
                "llama3.2:3b",
            )

            self.llm = ChatOllama(
                model=self.model,
                temperature=0,
            )

        elif provider == "openai":
            self.model = model or os.getenv(
                "OPENAI_MODEL",
                "gpt-5-mini",
            )

            self.llm = ChatOpenAI(
                model=self.model,
                temperature=0,
            )

        else:
            raise ValueError(
                f"Unsupported LLM provider: {provider}"
            )

    def generate(self, prompt: str) -> str:
        """
        Send a prompt to the configured LLM
        and return the generated text.
        """

        response = self.llm.invoke(prompt)

        return response.content
