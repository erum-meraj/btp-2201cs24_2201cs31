# agents/base_agent.py
from langchain_google_genai import ChatGoogleGenerativeAI

class BaseAgent:
    """Base class for all Gemini-powered agents."""

    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash", temperature: float = 0.3):
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature,
        )

    def think(self, prompt: str):
        """Run LLM reasoning and return response."""
        response = self.llm.invoke(prompt)
        return response.content.strip()
