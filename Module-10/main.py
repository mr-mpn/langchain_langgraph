'''
Module 10: Multi-Agent Systems
Supervisor pattern
Subgraphs
Agent handoffs
When multi-agent is (and isn't) worth the complexity

Supervisor pattern: a router node classifies the request, then
    dispatches to one of several specialist nodes, each with its own
    system prompt / persona.
'''

from typing import Annotated, TypedDict
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ---------------------------------------------------------------------------
# 1. Shared state
#    next_agent is scratch space used only for routing; it doesn't need
#    to persist across turns the way "messages" does.
# ---------------------------------------------------------------------------
class SupervisorState(TypedDict):
    messages: Annotated[list, add_messages]
    next_agent: str


# ---------------------------------------------------------------------------
# 2. Supervisor node: classifies the request into a category
# ---------------------------------------------------------------------------
VALID_CATEGORIES = {"billing", "tech_support", "general"}

def supervisor_node(state: SupervisorState) -> dict:
    last_message = state["messages"][-1].content

    routing_prompt = f"""Classify this user request into exactly one category:
billing, tech_support, or general.

Rules:
- billing: payments, invoices, refunds, subscription charges
- tech_support: errors, bugs, something not working, how-to technical questions
- general: anything else

Request: "{last_message}"

Respond with ONLY the category name, nothing else."""

    decision = model.invoke(routing_prompt).content.strip().lower()

    if decision not in VALID_CATEGORIES:
        decision = "general"  # fallback if the model returns something unexpected

    print(f"[supervisor] routed to: {decision}")
    return {"next_agent": decision}


def route_to_specialist(state: SupervisorState) -> str:
    """Reads the routing decision made by supervisor_node."""
    return state["next_agent"]


# ---------------------------------------------------------------------------
# 3. Specialist nodes — each has its own persona via a system prompt
# ---------------------------------------------------------------------------
def billing_node(state: SupervisorState) -> dict:
    system = SystemMessage(
        "You are a billing specialist. Be precise about charges, refunds, "
        "and subscriptions. Ask for an order number if relevant."
    )
    response = model.invoke([system] + state["messages"])
    return {"messages": [response]}


def tech_support_node(state: SupervisorState) -> dict:
    system = SystemMessage(
        "You are a technical support specialist. Give clear, step-by-step "
        "troubleshooting instructions."
    )
    response = model.invoke([system] + state["messages"])
    return {"messages": [response]}


def general_node(state: SupervisorState) -> dict:
    system = SystemMessage("You are a friendly general assistant.")
    response = model.invoke([system] + state["messages"])
    return {"messages": [response]}


# ---------------------------------------------------------------------------
# 4. Build the graph
#
#         START -> supervisor -> (conditional) -> billing       -> END
#                                              \-> tech_support  -> END
#                                              \-> general       -> END
# ---------------------------------------------------------------------------
builder = StateGraph(SupervisorState)

builder.add_node("supervisor", supervisor_node)
builder.add_node("billing", billing_node)
builder.add_node("tech_support", tech_support_node)
builder.add_node("general", general_node)

builder.add_edge(START, "supervisor")

builder.add_conditional_edges(
    "supervisor",
    route_to_specialist,
    {
        "billing": "billing",
        "tech_support": "tech_support",
        "general": "general",
    },
)

builder.add_edge("billing", END)
builder.add_edge("tech_support", END)
builder.add_edge("general", END)

graph = builder.compile()


# ---------------------------------------------------------------------------
# 5. Run it against a few different request types
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Graph structure:")
    print(graph.get_graph().draw_ascii())
    print()

    test_requests = [
        "I was charged twice for my subscription this month.",
        "The app crashes every time I try to upload a file.",
        "What's a good name for a coding blog?",
    ]

    for req in test_requests:
        print("=" * 60)
        print("USER:", req)
        result = graph.invoke({"messages": [HumanMessage(req)], "next_agent": ""})
        print("AGENT USED:", result["next_agent"])
        result["messages"][-1].pretty_print()
        print()