from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
# .env lives in the project root (langchain-agents/), not necessarily the process cwd
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

tavily_client = TavilyClient()
@tool
def search(query: str) -> str:
    """Search the web for information
    Args:
        query: The query to search for
    Returns:
        The search results
    """
    print(f"Searching for {query}")
    results = tavily_client.search(query)
    return results

def main() -> None:
    print("Hello from langchain-agents!")
    model = ChatOpenAI(model="gpt-5")
    agent = create_agent(model, tools=[search])
    result = agent.invoke({"messages": HumanMessage(content="search for jobs in canada , USA as AI Engineer")})
    print(result)

if __name__ == "__main__":
    main()