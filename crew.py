
"""AI Learning Center Support Crew

Multi-agent system that answers user questions using a three-tier approach:
  1. Policy Agent: Checks company policy documents first
  2. Offline Agent: Uses internal knowledge for general questions
  3. Web Agent: Searches the web when needed for current information
"""

import os
from pathlib import Path
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from ddgs import DDGS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Configuration ---
MODEL_NAME, MAX_SEARCH_RESULTS, VERBOSE = "gpt-4.1-mini", 5, True

# Load the policy document used by the policy tool.
POLICY_PATH = Path(__file__).with_name("ai_learning_center_policy.md")
POLICY_TEXT = POLICY_PATH.read_text(encoding="utf-8")

# ============================================================================
# TOOLS: Custom tools for policy lookup and web search
# ============================================================================

class PolicyLookupTool(BaseTool):
    """Returns company policy document for the agent to interpret."""
    name: str = "Policy Knowledge Base"
    description: str = "Useful for answering questions about AI Learning Center policies."
    def _run(self, query: str) -> str: return POLICY_TEXT


class DuckDuckGoSearchTool(BaseTool):
    """Searches the web using DuckDuckGo and returns formatted results."""
    name: str = "DuckDuckGo Search"
    description: str = "Search the web for current info and summarize relevant results."
    def _run(self, query: str) -> str:
        """Run a lightweight web search and return a short, readable list."""
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=MAX_SEARCH_RESULTS))
        if not results: return "No results found."
        # Format results in a simple bullet list for the agent to summarize.
        return "\n".join([f"- {result.get('title') or result.get('heading') or 'Result'}\n"
            f"  {result.get('href') or result.get('url') or ''}\n"
            f"  {result.get('body') or result.get('snippet') or ''}" for result in results])

# Instantiate tools once for reuse
policy_tool = PolicyLookupTool()
search_tool = DuckDuckGoSearchTool()

# ============================================================================
# AGENTS: Three specialized agents forming the support team
# ============================================================================

# Policy specialist: answers strictly from the policy document.
support_agent = Agent(role='Senior Policy Support Specialist', goal='Provide accurate answers based strictly on company policy.',
    backstory='You are the guardian of company rules. You never guess. You use the Policy Knowledge Base.',
    tools=[policy_tool], verbose=VERBOSE, allow_delegation=False, llm=MODEL_NAME)

# Agent 2: Offline generalist (uses internal knowledge, no web)
offline_info_agent = Agent(role='AI Education Generalist', goal='Answer general AI questions without web search, using internal knowledge only.',
    backstory='You are a helpful educator. If you are unsure, say so and suggest a web search.',
    tools=[], verbose=VERBOSE, allow_delegation=False, llm=MODEL_NAME)

# Agent 3: Web-enabled generalist (searches web when needed)
general_info_agent = Agent(role='Web-Enabled AI Education Consultant', goal='Help users with general AI concepts, current trends and other questions.',
    backstory='You are an enthusiastic educator. You can search the web to answer user questions.',
    tools=[search_tool], verbose=VERBOSE, allow_delegation=False, llm=MODEL_NAME)

# ============================================================================
# TASKS: Sequential workflow (policy → offline → web)
# ============================================================================

# Task 1: Check policy first
task_policy = Task(description="Check if the query '{user_query}' requires policy info. If so, answer it. If not, say 'Not a policy question'.",
    expected_output="Policy answer or 'N/A'", agent=support_agent)

# Task 2: Use internal knowledge if not a policy question
task_offline = Task(description="If the previous agent said 'N/A', answer the query '{user_query}' using internal knowledge only. "
    "If you cannot answer confidently, respond with 'NEEDS_WEB'. If policy answered it, just summarize.",
    expected_output="Answer or 'NEEDS_WEB'.", agent=offline_info_agent, context=[task_policy])

# Task 3: Search web only if needed
task_general = Task(description="If the previous agent said 'NEEDS_WEB', answer the query '{user_query}' using web search. "
    "If the previous agent answered it or policy answered it, just summarize.",
    expected_output="Final answer to user.", agent=general_info_agent, context=[task_offline])

# ============================================================================
# CREW: Assemble agents and tasks into a sequential workflow
# ============================================================================

support_crew = Crew(agents=[support_agent, offline_info_agent, general_info_agent],
    tasks=[task_policy, task_offline, task_general], process=Process.sequential, verbose=VERBOSE)


def get_support_response(user_query: str) -> str:
    """Run the crew and return the final response for a given user query."""
    return support_crew.kickoff(inputs={'user_query': user_query})


def main() -> None:
    """CLI entry point for running the support workflow."""
    print("### Welcome to AI Learning Center Support ###")
    user_input = input("\nHow can we help you today? ").strip()
    print(f"\n\n########################\nFINAL ANSWER:\n{get_support_response(user_input)}\n########################")

if __name__ == "__main__": main()
