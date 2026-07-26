'''
.invoke() VS .stream()
The core idea

.invoke() runs the entire graph and gives you only the final result, once everything is done.

.stream() runs the same graph, but gives you the state after each node 
finishes, one at a time, as it happens — like watching progress instead of waiting for a finished report.
'''

from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    value: int

def step1(state: State) -> dict:
    return {"value": state["value"] + 1}

def step2(state: State) -> dict:
    return {"value": state["value"] * 10}

def step3(state: State) -> dict:
    return {"value": state["value"] - 5}

builder = StateGraph(State)
builder.add_node("step1", step1)
builder.add_node("step2", step2)
builder.add_node("step3", step3)
builder.add_edge(START, "step1")
builder.add_edge("step1", "step2")
builder.add_edge("step2", "step3")
builder.add_edge("step3", END)

graph = builder.compile()

# --- invoke: only the final answer ---
print("invoke result:", graph.invoke({"value": 1}))
# invoke result: {'value': 15}

# --- stream: state after EACH node ---
for chunk in graph.stream({"value": 1}, stream_mode="values"):
    print("chunk:", chunk)