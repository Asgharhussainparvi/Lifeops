import os
import re
from typing import TypedDict, Optional

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    HumanMessage,
    ToolMessage
)
from tools import (
    search_knowledge,
    compare_options,
    calculate_monthly_runway,
    calculate_break_even_users,
    calculate_risk_score
)
load_dotenv()


# =========================
# SHARED STATE
# =========================

class LifeOpsState(TypedDict):
    user_goal: str
    research_result: str
    financial_result: str
    risk_result: str
    final_decision: str
# =========================
# GEMINI MODEL
# =========================

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)


# =========================
# GEMINI HELPER
# =========================

def ask_gemini(prompt: str) -> str:
    """
    Safely call Gemini.

    If the API quota has been exhausted, return a useful
    message instead of crashing the entire LangGraph.
    """

    try:
        response = model.invoke([
            HumanMessage(content=prompt)
        ])

        return response.content

    except Exception as e:

        error_text = str(e)

        if (
            "429" in error_text
            or "RESOURCE_EXHAUSTED" in error_text
            or "quota" in error_text.lower()
        ):
            return (
                "Gemini API quota is currently exhausted. "
                "This agent could not perform its AI analysis. "
                "Please try again after the Gemini quota resets."
            )

        raise


# =========================
# RESEARCH AGENT
# =========================

def research_agent(state: LifeOpsState):

    prompt = f"""
You are the Research Agent.

User goal:
{state["user_goal"]}

Use your available tools when they can provide useful information.

Analyze:
- Market considerations
- Relevant knowledge
- Important unknowns
- Possible alternatives

Do not invent facts.
Clearly identify limitations.
"""

    result = run_agent(
        model=model,
        tools=[
            search_knowledge,
            compare_options
        ],
        prompt=prompt
    )

    return {
        "research_result": result
    }

# =========================
# FINANCIAL AGENT
# =========================

def financial_agent(state: LifeOpsState):

    prompt = f"""
You are the Financial Agent.

User goal:
{state["user_goal"]}

Use financial tools whenever calculations are needed.

Analyze:
- Budget
- Monthly expenses
- Runway
- Break-even
- Financial assumptions

Do not invent missing numbers.
If required information is missing, explain what is needed.
"""

    result = run_agent(
        model=model,
        tools=[
            calculate_monthly_runway,
            calculate_break_even_users
        ],
        prompt=prompt
    )

    return {
        "financial_result": result
    }

# =========================
# RISK AGENT
# =========================

def risk_agent(state: LifeOpsState):

    prompt = f"""
You are the Risk Agent.

User goal:
{state["user_goal"]}

Use the risk calculation tool when appropriate.

Analyze:
- Technical risks
- Financial risks
- Operational risks
- Probability
- Impact
- Mitigation

Do not invent facts.
Clearly identify assumptions.
"""

    result = run_agent(
        model=model,
        tools=[
            calculate_risk_score
        ],
        prompt=prompt
    )

    return {
        "risk_result": result
    }    
# =========================
# SUPERVISOR
# =========================

def supervisor_agent(state: LifeOpsState):

    prompt = f"""
You are the Supervisor of a multi-agent decision system.

User goal:
{state["user_goal"]}

Decide which specialist should handle this request.

Available specialists:

- research
  Use for market research, competitors, trends, external information,
  comparisons, and unknown facts.

- financial
  Use for budgets, costs, revenue, runway, break-even,
  and financial calculations.

- risk
  Use for risks, uncertainty, technical risks,
  operational risks, and mitigation.

Return ONLY one word:

research
financial
risk
"""

    response = model.invoke([
        HumanMessage(content=prompt)
    ])

    decision = response.content.strip().lower()

    return {
        "next_agent": decision
    }
    
def final_decision_agent(state: LifeOpsState):

    prompt = f"""
You are the Final Decision Agent.

The following specialist agents analyzed the same user decision.

USER GOAL:
{state["user_goal"]}

RESEARCH ANALYSIS:
{state["research_result"]}

FINANCIAL ANALYSIS:
{state["financial_result"]}

RISK ANALYSIS:
{state["risk_result"]}

Combine all three analyses.

Create a practical final decision.

Include:

1. Overall recommendation
2. Important reasoning
3. Financial considerations
4. Major risks
5. Important unknowns
6. Next steps

Do not invent facts.
Clearly mention assumptions.
"""

    response = model.invoke([
        HumanMessage(content=prompt)
    ])

    return {
        "final_decision": response.content
    }
    
def run_agent(model, tools, prompt):

    model_with_tools = model.bind_tools(tools)

    messages = [
        HumanMessage(content=prompt)
    ]

    while True:

        response = model_with_tools.invoke(messages)

        messages.append(response)

        if not response.tool_calls:
            return response.content

        for tool_call in response.tool_calls:

            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            selected_tool = next(
                tool for tool in tools
                if tool.name == tool_name
            )

            tool_result = selected_tool.invoke(tool_args)

            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call["id"]
                )
            )