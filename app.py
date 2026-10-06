from langgraph.graph import StateGraph, START, END

from agents import (
    LifeOpsState,
    research_agent,
    financial_agent,
    risk_agent,
    final_decision_agent
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
    "risk_result": "",
    "final_decision": ""
}


# =========================
# GRAPH
# =========================

graph = StateGraph(LifeOpsState)


graph.add_node("research", research_agent)
graph.add_node("financial", financial_agent)
graph.add_node("risk", risk_agent)
graph.add_node("final_decision", final_decision_agent)


# =========================
# FAN-OUT
# =========================

graph.add_edge(START, "research")
graph.add_edge(START, "financial")
graph.add_edge(START, "risk")


# =========================
# FAN-IN
# =========================

graph.add_edge("research", "final_decision")
graph.add_edge("financial", "final_decision")
graph.add_edge("risk", "final_decision")


# =========================
# FINAL
# =========================

graph.add_edge("final_decision", END)


app = graph.compile()


# =========================
# RUN
# =========================

result = app.invoke(initial_state)


print("\n========== RESEARCH ==========\n")
print(result["research_result"])

print("\n========== FINANCIAL ==========\n")
print(result["financial_result"])

print("\n========== RISK ==========\n")
print(result["risk_result"])

print("\n========== FINAL DECISION ==========\n")
print(result["final_decision"])