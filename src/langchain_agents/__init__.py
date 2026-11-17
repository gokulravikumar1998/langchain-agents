from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

# .env lives in the project root (langchain-agents/), not necessarily the process cwd
load_dotenv(Path(__file__).resolve().parents[2] / ".env")


# --- Structured output schema (final agent answer shape) ---
# Each JobListing = one TABLE ROW. AgentResponse.jobs = the full TABLE.


class JobListing(BaseModel):
    """One job row for tabular display."""

    title: str = Field(description="Job title")
    company: str = Field(description="Company name")
    location: str = Field(description="City / country / remote")
    salary: Optional[str] = Field(
        default=None, description="Salary if mentioned, else null"
    )
    url: str = Field(description="Link to the job posting")
    source: str = Field(description="Website or source name")


class AgentResponse(BaseModel):
    """Final structured response: summary + job table rows + sources."""

    summary: str = Field(description="Short overview of the search findings")
    jobs: List[JobListing] = Field(
        default_factory=list,
        description="Job listings as table rows (title, company, location, salary, url, source)",
    )
    sources: List[str] = Field(
        default_factory=list, description="URLs used to generate the answer"
    )


def jobs_to_markdown_table(jobs: List[JobListing]) -> str:
    """Render structured job rows as a Markdown table."""
    if not jobs:
        return "_No jobs found._"

    headers = ["Title", "Company", "Location", "Salary", "URL", "Source"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for job in jobs:
        row = [
            job.title.replace("|", "/"),
            job.company.replace("|", "/"),
            job.location.replace("|", "/"),
            (job.salary or "—").replace("|", "/"),
            job.url.replace("|", "/"),
            job.source.replace("|", "/"),
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    print("Hello from langchain-agents!")
    model = ChatOpenAI(model="gpt-5")

    # response_format options (from docs):
    # 1) AgentResponse              → auto: ProviderStrategy for OpenAI/Anthropic/xAI/Gemini
    # 2) ProviderStrategy(schema)   → force native provider structured output
    # 3) ToolStrategy(schema)       → force tool-calling structured output (+ Unions, retries)
    # Schema types supported: Pydantic | dataclass | TypedDict | JSON Schema dict
    agent = create_agent(
        model,
        tools=[TavilySearch()],
        response_format=ProviderStrategy(AgentResponse, strict=True),
        system_prompt=(
            "You are a job search assistant. Use the search tool, then return "
            "structured job listings. Put each job as one item in `jobs` with "
            "title, company, location, salary (if known), url, and source."
        ),
    )

    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Search for AI Engineer jobs in Canada and USA. Return top listings."
                )
            ]
        }
    )

    # Structured data lives here — NOT only in messages
    structured: AgentResponse = result["structured_response"]
    print("\n=== Summary ===")
    print(structured.summary)
    print("\n=== Jobs (Markdown table) ===")
    print(jobs_to_markdown_table(structured.jobs))
    print("\n=== Sources ===")
    for src in structured.sources:
        print(f"- {src}")


if __name__ == "__main__":
    main()
