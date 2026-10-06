from langgraph.graph import StateGraph, START, END

from agents import (
    LifeOpsState,
    supervisor_agent,
    research_agent,
    financial_agent,
    risk_agent
)


user_goal = input("What decision do you want help with? ")


initial_state: LifeOpsState = {
    "user_goal": user_goal,
    "research_result": "",
    "financial_result": "",
    "risk_result": "",
    "next_agent": ""
}


# =========================
# ROUTER
# =========================

def route_from_supervisor(state: LifeOpsState):

    return state["next_agent"]


# =========================
# GRAPH
# =========================

graph = StateGraph(LifeOpsState)

graph.add_node("supervisor", supervisor_agent)
graph.add_node("research", research_agent)
graph.add_node("financial", financial_agent)
graph.add_node("risk", risk_agent)


# START → SUPERVISOR

graph.add_edge(START, "supervisor")


# SUPERVISOR → SPECIALIST

graph.add_conditional_edges(
    "supervisor",
    route_from_supervisor,
    {
        "research": "research",
        "financial": "financial",
        "risk": "risk"
    }
)


# SPECIALIST → END

graph.add_edge("research", END)
graph.add_edge("financial", END)
graph.add_edge("risk", END)


app = graph.compile()


# =========================
# RUN
# =========================

result = app.invoke(initial_state)


print("\n========== RESULT ==========\n")

if result["research_result"]:
    print(result["research_result"])

if result["financial_result"]:
    print(result["financial_result"])

if result["risk_result"]:
    print(result["risk_result"])