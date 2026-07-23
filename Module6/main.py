'''
Graphs vs. chains: why control flow matters
Core concepts: StateGraph, nodes, edges
Defining state with TypedDict / Pydantic
Building your first graph (simple linear flow)

'''

from typing import TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)


# --- 1. Define the shared state ---
class GraphState(TypedDict):
    question: str
    draft: str
    final: str

# --- 2. Define nodes ---
def draft_node(state: GraphState) -> dict:
    """Generate a quick, informal first-pass answer."""
    print(f'state before draft node : {state}')
    response = model.invoke(f"Answer briefly and informally: {state['question']}")
    return {"draft": response.content}

def polish_node(state: GraphState) -> dict:
    """Rewrite the draft in a more formal tone."""
    print(f'state before polish node : {state}')
    response = model.invoke(
        f"Rewrite this answer to be more formal and well-structured:\n\n{state['draft']}"
    )
    return {"final": response.content}

# --- 3. Build the graph ---
builder = StateGraph(GraphState)

builder.add_node("draft", draft_node)
builder.add_node("polish", polish_node)

builder.add_edge(START, "draft")
builder.add_edge("draft", "polish")
builder.add_edge("polish", END)

graph = builder.compile()

# --- 4. Visualize the graph structure ---
print("Graph structure:")
print(graph.get_graph().draw_ascii())  #get_graph needs to be installed but not imported
print() 


# --- 5. Run it ---

initial_state = {
        "question": "What is a Python decorator?",
        "draft": "",
        "final": "",
    }

result = graph.invoke(initial_state)

print("QUESTION:", result["question"])
print("\nDRAFT:\n", result["draft"])
print("\nFINAL:\n", result["final"])
