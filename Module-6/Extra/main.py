'''
Ai agent that changes the price from USD to EUR and then decides if the item can be purchased according to the budget or not
'''

from typing import TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)


# --- 1. Define the shared state ---
class GraphState(TypedDict):
    item_euro: int
    item_dollar: int
    purchasable : bool
    

# --- 2. Define nodes ---
def exchange_currency(state: GraphState)->dict:
    '''
    Exchanges the price from Dollar to EUR
    input : euro : int
    output : float
    '''
    print(f'state before exchange node : {state}')
    response = model.invoke(f"Change the price from Dollar to Euro and in the response give me just the value in euro , use any rating you want :\n\n{state['item_dollar']}")
    return {"item_euro": response.content}


def is_in_budget(state: GraphState) -> dict:
    """Compare the price with the budget"""
    print(f'state before is_in_budget node : {state}')
    response = model.invoke(
        f"The budget is 30 euro , given the price of the item in euro decide if it is in our budget or now do not explain just write the boolean result for me  :\n\n{state['item_euro']}"
    )
    return {"purchasable": response.content}

# --- 3. Build the graph ---
builder = StateGraph(GraphState)

builder.add_node("exchange_currency", exchange_currency)
builder.add_node("is_in_budget", is_in_budget)

builder.add_edge(START, "exchange_currency")
builder.add_edge("exchange_currency", "is_in_budget")
builder.add_edge("is_in_budget", END)

graph = builder.compile()

# --- 4. Visualize the graph structure ---
print("Graph structure:")
print(graph.get_graph().draw_ascii())  #get_graph needs to be installed but not imported
print() 


# --- 5. Run it ---

initial_state = {

        "item_dollar": 30,
        "item_euro": 0,
        "purchasable": False,
    }

result = graph.invoke(initial_state)

print("\nITEM PRICE (USD):\n", result["item_dollar"])
print("\nITEM PRICE (EUR):\n", result["item_euro"])
print("\nPURCHASABLE:\n", result["purchasable"])
