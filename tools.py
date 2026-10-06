from langchain_core.tools import tool


# =========================
# RESEARCH TOOLS
# =========================

@tool
def search_knowledge(topic: str) -> str:
    """
    Search LifeOps' local knowledge base for a topic.
    """

    knowledge = {
        "saas": "SaaS products usually require recurring infrastructure, monitoring, authentication, database, and deployment costs.",
        "developers": "Developer-focused SaaS products often compete on usability, integrations, automation, and developer experience.",
        "market": "Market research should identify competitors, customer needs, pricing models, and differentiation opportunities."
    }

    topic_lower = topic.lower()

    for key, value in knowledge.items():
        if key in topic_lower:
            return value

    return "No relevant information found in the local knowledge base."


@tool
def compare_options(option_a: str, option_b: str) -> str:
    """
    Compare two decision options at a high level.
    """

    return f"""
Option A: {option_a}

Option B: {option_b}

Comparison should consider:
- Cost
- Complexity
- Scalability
- Risk
- Maintenance
- Long-term flexibility
"""


# =========================
# FINANCIAL TOOLS
# =========================

@tool
def calculate_monthly_runway(
    available_money: float,
    monthly_expenses: float
) -> float:
    """
    Calculate how many months the available money can cover expenses.
    """

    if monthly_expenses <= 0:
        return 0

    return available_money / monthly_expenses


@tool
def calculate_break_even_users(
    monthly_fixed_cost: float,
    revenue_per_user: float,
    variable_cost_per_user: float
) -> float:
    """
    Calculate the number of users required to break even.
    """

    contribution = revenue_per_user - variable_cost_per_user

    if contribution <= 0:
        return 0

    return monthly_fixed_cost / contribution


# =========================
# RISK TOOL
# =========================

@tool
def calculate_risk_score(
    probability: int,
    impact: int
) -> int:
    """
    Calculate risk score using probability × impact.
    """

    return probability * impact