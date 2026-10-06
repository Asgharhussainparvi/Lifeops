import os
import re
from typing import TypedDict, Optional

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

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
You are the Research Agent in a multi-agent decision system.

Analyze the user's goal from a research perspective.

User goal:
{state["user_goal"]}

Identify:
- Important market/research questions
- Information that should be investigated
- Important unknowns
- Evidence that would be useful

Do not invent facts.
If information is unavailable, clearly say so.
"""

    result = ask_gemini(prompt)

    return {
        "research_result": result
    }


# =========================
# FINANCIAL EXTRACTION
# =========================

def extract_money_after_keyword(
    text: str,
    keywords: list[str]
) -> Optional[float]:

    text_lower = text.lower()

    for keyword in keywords:

        pattern = (
            re.escape(keyword)
            + r".{0,80}?\$?\s*"
            r"([0-9]+(?:\.[0-9]+)?)"
        )

        match = re.search(
            pattern,
            text_lower
        )

        if match:
            return float(match.group(1))

    return None


def calculate_budget_from_text(
    text: str
) -> Optional[str]:

    text_lower = text.lower()

    # -----------------------------------------
    # Budget
    # -----------------------------------------

    budget = extract_money_after_keyword(
        text,
        [
            "budget",
            "have",
            "with a"
        ]
    )

    if budget is None:
        return None

    # -----------------------------------------
    # Infrastructure
    # -----------------------------------------

    infrastructure = extract_money_after_keyword(
        text,
        [
            "infrastructure costs",
            "infrastructure cost",
            "infrastructure"
        ]
    )

    # -----------------------------------------
    # AI
    # -----------------------------------------

    ai_cost = extract_money_after_keyword(
        text,
        [
            "ai costs",
            "ai cost",
            "ai"
        ]
    )

    # -----------------------------------------
    # Database
    # -----------------------------------------

    database = extract_money_after_keyword(
        text,
        [
            "database costs",
            "database cost",
            "database"
        ]
    )

    # -----------------------------------------
    # Other
    # -----------------------------------------

    other = extract_money_after_keyword(
        text,
        [
            "other costs",
            "other cost",
            "other"
        ]
    )

    # -----------------------------------------
    # Monthly operating cost
    # -----------------------------------------

    monthly_cost = extract_money_after_keyword(
        text,
        [
            "monthly operating cost",
            "monthly operating costs",
            "monthly cost",
            "operating cost"
        ]
    )

    # -----------------------------------------
    # If this is only a general budget question,
    # don't pretend we know the other expenses.
    # -----------------------------------------

    has_cost_breakdown = any(
        value is not None
        for value in [
            infrastructure,
            ai_cost,
            database,
            other
        ]
    )

    if not has_cost_breakdown:

        return (
            f"Known financial information:\n"
            f"- Budget: ${budget:,.2f}\n"
            f"- Infrastructure budget/cost: "
            f"${budget:,.2f}\n\n"
            f"Financial analysis:\n"
            f"The available information indicates a "
            f"${budget:,.2f} infrastructure budget, but "
            f"there is not enough information to calculate "
            f"total monthly operating costs, runway, or "
            f"profitability.\n\n"
            f"Important missing information:\n"
            f"- AI/API costs\n"
            f"- Database costs\n"
            f"- Other infrastructure/services\n"
            f"- Expected revenue\n"
            f"- Number of users\n"
            f"- Variable cost per user"
        )

    # -----------------------------------------
    # Missing values become zero only when the
    # user explicitly provides a cost breakdown.
    # -----------------------------------------

    infrastructure = infrastructure or 0
    ai_cost = ai_cost or 0
    database = database or 0
    other = other or 0

    total_costs = (
        infrastructure
        + ai_cost
        + database
        + other
    )

    remaining = budget - total_costs

    result = (
        f"Financial calculation:\n\n"
        f"Budget: ${budget:,.2f}\n"
        f"Infrastructure: ${infrastructure:,.2f}\n"
        f"AI: ${ai_cost:,.2f}\n"
        f"Database: ${database:,.2f}\n"
        f"Other: ${other:,.2f}\n\n"
        f"Total listed costs: ${total_costs:,.2f}\n"
        f"Remaining budget: ${remaining:,.2f}"
    )

    # -----------------------------------------
    # Runway
    # -----------------------------------------

    if monthly_cost is not None and monthly_cost > 0:

        months = remaining / monthly_cost

        result += (
            f"\n\nMonthly operating cost: "
            f"${monthly_cost:,.2f}\n"
            f"Estimated runway: "
            f"{months:.1f} months"
        )

    elif (
        "runway" in text_lower
        or "months can i survive" in text_lower
        or "how many months" in text_lower
    ):

        result += (
            "\n\nRunway cannot be calculated because "
            "the monthly operating cost was not provided."
        )

    return result


# =========================
# FINANCIAL AGENT
# =========================

def financial_agent(state: LifeOpsState):

    user_goal = state["user_goal"]

    # -----------------------------------------
    # First attempt deterministic calculation.
    # -----------------------------------------

    local_result = calculate_budget_from_text(
        user_goal
    )

    if local_result:

        return {
            "financial_result": local_result
        }

    # -----------------------------------------
    # If Python cannot calculate it, use Gemini.
    # -----------------------------------------

    prompt = f"""
You are the Financial Agent in a multi-agent decision system.

Analyze the user's goal from a financial perspective.

User goal:
{user_goal}

Research information:
{state["research_result"]}

Focus on:
- Budget
- Costs
- Revenue
- Financial constraints
- Break-even considerations
- Important financial assumptions
- Missing financial information

Do not invent numerical facts.

If information is missing, clearly identify it.

Provide practical financial guidance.
"""

    result = ask_gemini(prompt)

    return {
        "financial_result": result
    }


# =========================
# RISK AGENT
# =========================

def risk_agent(state: LifeOpsState):

    prompt = f"""
You are the Risk Agent in a multi-agent decision system.

Analyze the user's goal from a risk perspective.

User goal:
{state["user_goal"]}

Research analysis:
{state["research_result"]}

Financial analysis:
{state["financial_result"]}

Identify:
- Major risks
- Technical risks
- Financial risks
- Operational risks
- Important uncertainties
- Possible mitigation strategies

Do not invent facts.
Clearly identify assumptions.

Provide practical risk recommendations.
"""

    result = ask_gemini(prompt)

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