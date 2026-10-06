from langgraph.graph import StateGraph, START, END

from agents import (
    LifeOpsState,
    research_agent,
    financial_agent,
    risk_agent
)


# =========================
# USER INPUT
# =========================

user_goal = input("What decision do you want help with? ")


# =========================
# INITIAL STATE
# =========================

initial_state: LifeOpsState = {
    "user_goal": user_goal,
    "research_result": "",
    "financial_result": "",
    "risk_result": ""
}


# =========================
# GRAPH
# =========================

graph = StateGraph(LifeOpsState)

graph.add_node("research", research_agent)
graph.add_node("financial", financial_agent)
graph.add_node("risk", risk_agent)

graph.add_edge(START, "research")
graph.add_edge("research", "financial")
graph.add_edge("financial", "risk")
graph.add_edge("risk", END)

app = graph.compile()


# =========================
# RUN
# =========================

result = app.invoke(initial_state)


# =========================
# OUTPUT
# =========================

print("\n========== RESEARCH ==========\n")
print(result["research_result"])

print("\n========== FINANCIAL ==========\n")
print(result["financial_result"])

print("\n========== RISK ==========\n")
print(result["risk_result"])