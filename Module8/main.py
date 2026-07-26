'''
Module 8: LangGraph Agents in Practice
Building a ReAct-style agent from scratch in LangGraph
Tool-calling nodes
Human-in-the-loop (interrupts, interrupt_before/interrupt_after)
Streaming outputs

'''

from typing import Annotated, TypedDict
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()


# ---------------------------------------------------------------------------
# 1. Define tools
#    The docstring is sent to the model as the tool description, so the
#    model knows WHEN to use each tool. Type hints define the input schema.
# ---------------------------------------------------------------------------
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    # In a real app this would call a weather API.
    fake_weather_db = {
        "milan": "22°C and sunny",
        "paris": "18°C and cloudy",
        "rome": "26°C and clear",
    }
    return fake_weather_db.get(city.lower(), f"No weather data for {city}.")


@tool
def add(a: int, b: int) -> int:
    """Add two integers together and return the result."""
    return a + b


tools = [get_weather, add]


# ---------------------------------------------------------------------------
# 2. Model, bound to the tools so it can request tool calls
# ---------------------------------------------------------------------------
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
model_with_tools = model.bind_tools(tools)


# ---------------------------------------------------------------------------
# 3. Define the agent's state
#    `Annotated[list, add_messages]` is a REDUCER: when a node returns
#    {"messages": [...]}, LangGraph APPENDS to the existing list instead
#    of overwriting it. This is what lets conversation history accumulate
#    across multiple loop iterations.
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ---------------------------------------------------------------------------
# 4. Define nodes
# ---------------------------------------------------------------------------
def agent_node(state: AgentState) -> dict:
    """
    Calls the model with the full message history so far.
    The model will either:
      - respond directly with a final answer, or
      - respond with one or more tool_calls it wants executed.
    """
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}  # appended via the reducer, not overwritten


# Prebuilt node: reads the last message's tool_calls, executes each tool,
# and returns the results as ToolMessage objects. Replaces the manual
# "for call in response.tool_calls" loop from Module 2.
tool_node = ToolNode(tools)


# ---------------------------------------------------------------------------
# 5. Routing logic: after the agent node runs, does it want a tool,
#    or is it done?
# ---------------------------------------------------------------------------
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return "end"


# ---------------------------------------------------------------------------
# 6. Build the graph
#
#    START -> agent -> (conditional) -> tools -> agent -> ... -> END
#                                  \-> END
# ---------------------------------------------------------------------------
builder = StateGraph(AgentState)

builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)

builder.add_edge(START, "agent")

builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",  # model wants a tool -> go execute it
        "end": END,        # model gave a final answer -> stop
    },
)

builder.add_edge("tools", "agent")  # the cycle: after tool results, go back to agent

graph = builder.compile()


# ---------------------------------------------------------------------------
# 7. Run it
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Graph structure:")
    print(graph.get_graph().draw_ascii())
    print()

    user_question = "What's the weather in Milan, and what's 12 + 30?"

    print("=== Full invoke (no streaming) ===")
    result = graph.invoke(
        {"messages": [HumanMessage(user_question)]},
        config={"recursion_limit": 25},  # safety net against runaway loops
    )

    for m in result["messages"]:
        m.pretty_print()

    print("\n=== Streaming version (shows state after each node) ===")
    for chunk in graph.stream(
        {"messages": [HumanMessage(user_question)]},
        stream_mode="values",
    ):
        chunk["messages"][-1].pretty_print()