'''
Module 9: Persistence & Memory in LangGraph
Checkpointers (MemorySaver, SQLite, Postgres)
Thread-based conversation persistence
Time travel / replaying state

A checkpointer automatically saves the graph's state after every node execution, keyed by a thread ID.
This gives you:

Multi-turn conversations without manually managing message lists
The ability to pause and resume later (even after a process restart, if using a persistent backend)
Time travel — inspecting or rewinding to a previous state

Checkpointer = automatic state snapshots, keyed by thread_id, after every node.
             = memory (Module 3) + pause/resume (interrupts) + time travel,
               all from one mechanism.
'''

# module9_langgraph_persistence.py
#
# Demonstrates:
#   1. Multi-turn conversation via a checkpointer (replaces Module 3's memory pattern)
#   2. Pausing before a tool call and resuming after approval (human-in-the-loop)
#
# Builds on the agent from Module 8.

from typing import Annotated, TypedDict
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()


# ---------------------------------------------------------------------------
# 1. Tools
# ---------------------------------------------------------------------------
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    fake_weather_db = {
        "milan": "22°C and sunny",
        "paris": "18°C and cloudy",
        "rome": "26°C and clear",
    }
    return fake_weather_db.get(city.lower(), f"No weather data for {city}.")


tools = [get_weather]

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
model_with_tools = model.bind_tools(tools)


# ---------------------------------------------------------------------------
# 2. State (same message-list pattern as Module 8)
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ---------------------------------------------------------------------------
# 3. Nodes
# ---------------------------------------------------------------------------
def agent_node(state: AgentState) -> dict:
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}


tool_node = ToolNode(tools) #all the tools are now stored in one single node 


def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return "end"


# ---------------------------------------------------------------------------
# 4. Build the graph
# ---------------------------------------------------------------------------
builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", "end": END},
)                                       #
builder.add_edge("tools", "agent")

# --- The checkpointer: saves state after every node, keyed by thread_id ---
checkpointer = MemorySaver()

# --- Compile TWICE with different settings for the two demos below ---
# (a) plain graph, with memory, no interrupts -> for the multi-turn demo
graph_with_memory = builder.compile(checkpointer=checkpointer)

# (b) same graph, but pauses right before the "tools" node runs
#     -> for the human-in-the-loop approval demo
graph_with_interrupt = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["tools"],
)


# ---------------------------------------------------------------------------
# 5. Demo 1: multi-turn conversation, no manual message tracking needed
# ---------------------------------------------------------------------------
def demo_multi_turn():
    print("=" * 60)
    print("DEMO 1: Multi-turn conversation via checkpointer")
    print("=" * 60)

    config = {"configurable": {"thread_id": "conversation-1"}}

    # Turn 1
    result = graph_with_memory.invoke(
        {"messages": [HumanMessage("My favorite city is Milan.")]},
        config=config,
    )
    result["messages"][-1].pretty_print()

    # Turn 2 — we only send the NEW message.
    # The checkpointer loads prior history for thread_id "conversation-1"
    # and the add_messages reducer appends this new one automatically.
    result = graph_with_memory.invoke(
        {"messages": [HumanMessage("What's the weather like there?")]},
        config=config,
    )
    result["messages"][-1].pretty_print()

    # A DIFFERENT thread_id = a totally separate, isolated conversation
    other_config = {"configurable": {"thread_id": "conversation-2"}}
    result = graph_with_memory.invoke(
        {"messages": [HumanMessage("What's my favorite city?")]},
        config=other_config,
    )
    result["messages"][-1].pretty_print()
    # Expect: the model has NO idea, since this thread never mentioned it


# ---------------------------------------------------------------------------
# 6. Demo 2: pause before a tool call, inspect it, then approve & resume
# ---------------------------------------------------------------------------
def demo_interrupt_and_approve():
    print("\n" + "=" * 60)
    print("DEMO 2: Human-in-the-loop approval before a tool call")
    print("=" * 60)

    config = {"configurable": {"thread_id": "approval-flow-1"}}

    # This runs agent_node, sees the model wants to call a tool,
    # and STOPS right before executing "tools" (because of interrupt_before).
    result = graph_with_interrupt.invoke(
        {"messages": [HumanMessage("What's the weather in Rome?")]},
        config=config,
    )

    pending_call = result["messages"][-1].tool_calls
    print("Graph paused. Pending tool call(s):", pending_call)

    # Inspect what node would run next — confirms we're paused, not finished
    snapshot = graph_with_interrupt.get_state(config)
    print("Next node to run:", snapshot.next)

    # Simulate a human approving the action
    user_approves = True

    if user_approves:
        print("\nApproved. Resuming...")
        # Passing None resumes execution from the saved checkpoint,
        # continuing on to the "tools" node and beyond.
        final_result = graph_with_interrupt.invoke(None, config=config)
        final_result["messages"][-1].pretty_print()
    else:
        print("\nRejected. Not resuming — tool call is discarded.")


# ---------------------------------------------------------------------------
# 7. Run both demos
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo_multi_turn()
    demo_interrupt_and_approve()