'''
Module 2: Tools & Function Calling
Defining tools with @tool decorator
Binding tools to models (bind_tools)
Tool calling loop (manual implementation before agents abstract it)
Structured tool inputs with Pydantic
'''


from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage,ToolMessage
from pydantic import BaseModel, Field

class WeatherInput(BaseModel):
    city: str = Field(description="The city name")
    units: str = Field(default="celsius", description="celsius or fahrenheit")
'''
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    print("tool : get_weather has been launched")
    # pretend this calls a real API
    return f"It's 22°C and sunny in {city}."
'''
@tool
def add(a: int, b: int) -> int:
    """Add two integers together."""
    print("tool : add has been launched ")
    return a + b

@tool(args_schema=WeatherInput)
def get_weather(city: str, units: str = "celsius") -> str:
    """Get current weather for a city."""
    print("tool : get_weather has been launched")
    return f"Weather in {city}: 22°{units[0].upper()}"

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
tools = [get_weather, add]
model_with_tools = model.bind_tools(tools)

#messages = [HumanMessage("What's the weather in Milan , and what's 5 + 7?")]
messages = [HumanMessage("What's the weather in Milan in farenhite, and what's 5 + 7?")]
response = model_with_tools.invoke(messages) #The model responded with a request to call a tool
print(response.tool_calls)
messages.append(response)

# The model may request multiple tool calls at once
for call in response.tool_calls:
    selected_tool = {"get_weather": get_weather, "add": add}[call["name"]]
    tool_result = selected_tool.invoke(call["args"])
    messages.append(
        ToolMessage(content=str(tool_result), tool_call_id=call["id"])
    )

# Now send results back so the model can produce a final answer
final_response = model.invoke(messages)
print(final_response.content)
