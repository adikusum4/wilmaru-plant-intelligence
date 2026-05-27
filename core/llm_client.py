"""LLM client — Groq (gratis) default, Anthropic opsional."""
import os
from dotenv import load_dotenv
load_dotenv()

def get_llm(provider: str = "groq"):
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=os.getenv("GROQ_MODEL","llama-3.1-70b-versatile"),
                        api_key=os.getenv("GROQ_API_KEY"), temperature=0.1)
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(model="claude-3-5-haiku-20241022",
                         api_key=os.getenv("ANTHROPIC_API_KEY"))
