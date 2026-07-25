'''
Module 7: LangGraph Control Flow
Conditional edges (routing logic)
Cycles/loops (the key differentiator from LangChain chains)
START / END nodes
Parallel branches and fan-in

'''


from typing import TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# --- 1. Shared state ---
class GraphState(TypedDict):
    question: str
    answer: str
    is_valid: bool
    attempts: int
    max_attempts: int

# --- 2. Nodes ---
def generate_node(state: GraphState) -> dict:
    """Generate an answer. Deliberately loose so it sometimes needs a retry."""
    response = model.invoke(
        f"Answer this math question with ONLY the final number, "
        f"no explanation: {state['question']}"
    )
    attempts = state["attempts"] + 1
    print(f"[generate] attempt {attempts}: {response.content.strip()!r}")
    return {"answer": response.content.strip(), "attempts": attempts}

def validate_node(state: GraphState) -> dict:
    """Check if the answer is a valid number (simulates a real validation step)."""
    is_valid = state["answer"].replace("-", "").replace(".", "").isdigit()
    print(f"[validate] valid={is_valid}")
    return {"is_valid": is_valid}

def finalize_node(state: GraphState) -> dict:
    print("[finalize] done")
    return {}

# --- 3. Routing logic ---
def route_after_validate(state: GraphState) -> str:
    if state["is_valid"]:
        return "done"
    if state["attempts"] >= state["max_attempts"]:
        print("[route] max attempts reached, giving up")
        return "done"
    return "retry"

# --- 4. Build the graph ---
builder = StateGraph(GraphState)

builder.add_node("generate", generate_node)
builder.add_node("validate", validate_node)
builder.add_node("finalize", finalize_node)

builder.add_edge(START, "generate")
builder.add_edge("generate", "validate")

builder.add_conditional_edges(
    "validate",
    route_after_validate,
    {
        "retry": "generate",   # cycle back
        "done": "finalize",
    },
)

builder.add_edge("finalize", END)

graph = builder.compile()

# --- 5. Visualize ---
print("Graph structure:")
print(graph.get_graph().draw_ascii())
print()

# --- 6. Run it ---
if __name__ == "__main__":
    initial_state = {
        "question": "What is 47 * 3?",
        "answer": "",
        "is_valid": False,
        "attempts": 0,
        "max_attempts": 3,
    }

    result = graph.invoke(initial_state, config={"recursion_limit": 25})

    print("\n--- RESULT ---")
    print("Question:", result["question"])
    print("Answer:", result["answer"])
    print("Valid:", result["is_valid"])
    print("Attempts used:", result["attempts"])
    