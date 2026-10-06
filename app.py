import os
import re

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool

from langgraph.graph import (
    StateGraph,
    MessagesState,
    START,
    END
)

from langgraph.prebuilt import ToolNode


load_dotenv()


# =========================================================
# MODEL
# =========================================================

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)


# =========================================================
# TOOLS
# =========================================================

@tool
def calculate_project_budget(
    budget: float,
    infrastructure_cost: float,
    ai_cost: float,
    database_cost: float,
    other_costs: float
) -> float:
    """
    Calculate the remaining project budget.
    """

    return (
        budget
        - infrastructure_cost
        - ai_cost
        - database_cost
        - other_costs
    )


@tool
def calculate_monthly_runway(
    available_budget: float,
    monthly_cost: float
) -> float:
    """
    Calculate how many months the available budget can support.
    """

    if monthly_cost <= 0:
        return 0

    return available_budget / monthly_cost


@tool
def calculate_break_even_users(
    monthly_fixed_cost: float,
    price_per_user: float,
    variable_cost_per_user: float
) -> int:
    """
    Calculate the number of paying users required to break even.
    """

    contribution = price_per_user - variable_cost_per_user

    if contribution <= 0:
        return 0

    return int(monthly_fixed_cost / contribution) + 1


@tool
def compare_options(
    option_a: str,
    option_a_cost: float,
    option_b: str,
    option_b_cost: float
) -> str:
    """
    Compare two options based on monthly cost.
    """

    if option_a_cost < option_b_cost:
        cheaper = option_a

    elif option_b_cost < option_a_cost:
        cheaper = option_b

    else:
        cheaper = "Both options have the same cost."

    return (
        f"{option_a} costs ${option_a_cost:.2f}/month. "
        f"{option_b} costs ${option_b_cost:.2f}/month. "
        f"Cheaper option: {cheaper}."
    )


tools = [
    calculate_project_budget,
    calculate_monthly_runway,
    calculate_break_even_users,
    compare_options
]


# =========================================================
# BIND TOOLS TO MODEL
# =========================================================

model_with_tools = model.bind_tools(tools)


# =========================================================
# LOCAL BUDGET CALCULATOR
# =========================================================

def local_budget_calculation(text):
    """
    Handle simple budget/runway questions locally.

    This avoids using the Gemini API for straightforward
    arithmetic and therefore works even when Gemini quota
    is exhausted.
    """

    text_lower = text.lower()

    # Only attempt this for budget-related questions.
    budget_keywords = [
        "budget",
        "infrastructure",
        "ai cost",
        "database",
        "other costs",
        "monthly operating cost",
        "monthly cost",
        "months can i survive",
        "runway",
        "how much remains",
        "how much budget remains",
        "how much remains"
    ]

    if not any(keyword in text_lower for keyword in budget_keywords):
        return None

    # Extract dollar amounts.
    amounts = re.findall(
        r'\$?\s*([0-9]+(?:\.[0-9]+)?)',
        text
    )

    if len(amounts) < 4:
        return None

    numbers = [float(x) for x in amounts]

    # We expect:
    #
    # budget
    # infrastructure
    # AI
    # database
    # other
    #
    # and optionally monthly cost.
    budget = numbers[0]

    infrastructure = numbers[1]
    ai = numbers[2]
    database = numbers[3]
    other = numbers[4] if len(numbers) >= 5 else 0

    remaining = (
        budget
        - infrastructure
        - ai
        - database
        - other
    )

    # Look for monthly operating cost.
    monthly_cost = None

    monthly_patterns = [
        r'monthly operating cost.*?\$?\s*([0-9]+(?:\.[0-9]+)?)',
        r'monthly cost.*?\$?\s*([0-9]+(?:\.[0-9]+)?)',
        r'operating cost.*?\$?\s*([0-9]+(?:\.[0-9]+)?)'
    ]

    for pattern in monthly_patterns:
        match = re.search(
            pattern,
            text_lower
        )

        if match:
            monthly_cost = float(match.group(1))
            break

    # If runway was requested and monthly cost exists.
    if monthly_cost and monthly_cost > 0:
        months = remaining / monthly_cost

        return (
            f"After all the costs, ${remaining:,.2f} remains "
            f"in the budget.\n\n"
            f"With a monthly operating cost of "
            f"${monthly_cost:,.2f}, you have approximately "
            f"{months:.1f} months of runway."
        )

    return (
        f"After all the costs, ${remaining:,.2f} remains "
        f"in the budget."
    )


# =========================================================
# AGENT NODE
# =========================================================

def agent(state: MessagesState):

    user_message = state["messages"][-1].content

    # -----------------------------------------------------
    # Try local calculation first.
    # -----------------------------------------------------

    local_result = local_budget_calculation(
        user_message
    )

    if local_result:

        from langchain_core.messages import AIMessage

        return {
            "messages": [
                AIMessage(
                    content=local_result
                )
            ]
        }

    # -----------------------------------------------------
    # Otherwise use Gemini.
    # -----------------------------------------------------

    try:

        response = model_with_tools.invoke(
            state["messages"]
        )

        return {
            "messages": [response]
        }

    except Exception as e:

        error_text = str(e)

        if (
            "429" in error_text
            or "RESOURCE_EXHAUSTED" in error_text
            or "quota" in error_text.lower()
        ):

            from langchain_core.messages import AIMessage

            return {
                "messages": [
                    AIMessage(
                        content=(
                            "Gemini API quota has been reached. "
                            "Please try again after the quota resets."
                        )
                    )
                ]
            }

        raise


# =========================================================
# TOOL NODE
# =========================================================

tool_node = ToolNode(tools)


# =========================================================
# ROUTING
# =========================================================

def should_continue(state: MessagesState):

    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tools"

    return END


# =========================================================
# GRAPH
# =========================================================

graph_builder = StateGraph(MessagesState)

graph_builder.add_node(
    "agent",
    agent
)

graph_builder.add_node(
    "tools",
    tool_node
)

graph_builder.add_edge(
    START,
    "agent"
)

graph_builder.add_conditional_edges(
    "agent",
    should_continue
)

graph_builder.add_edge(
    "tools",
    "agent"
)

graph = graph_builder.compile()


# =========================================================
# USER INPUT
# =========================================================

goal = input(
    "\nWhat decision do you want help with?\n> "
)


# =========================================================
# RUN GRAPH
# =========================================================

result = graph.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": goal
            }
        ]
    }
)


# =========================================================
# FINAL RESPONSE
# =========================================================

print("\n===== LIFEOPS DECISION =====\n")

print(
    result["messages"][-1].content
)
