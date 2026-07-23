from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage,ToolMessage
from pydantic import BaseModel, Field

load_dotenv()

@tool
def exchange_currency(euro:int)->float:
    '''
    A function to exchange the rate from Euro to Dollar
    input : euro : int
    output : float
    '''
    return euro *1.14

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
tools = [exchange_currency]
model_with_tools = model.bind_tools(tools)



messages = [HumanMessage("i have 3 euros , how much is it in USD Dollars ?")] # A list that contains the messages 


response = model_with_tools.invoke(messages)
print(f'response.content : {response.content}') #Empty
print(f'response.tool_calls : {response.tool_calls}') #the tools the LLM has decided to call

messages.append(response)
print(f'messages : {messages}')

tool_call = response.tool_calls[0] # the tool call that has arrived from the LLM  response.tool_calls : [{'name': 'exchange_currency', 'args': {'euro': 3}, 'id': 'call_TV5SkS9VJ4sKwEAVivbgh7pg', 'type': 'tool_call'}]
print(f'tool_call : {tool_call}')


tool_result = exchange_currency.invoke(tool_call["args"])
print(f'tool_result : {tool_result}')


messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"]))

final_response = model_with_tools.invoke(messages)
print(f'final_response.content : {final_response.content}')

# what if we never used the tool calling ?

response_without_tool = model.invoke(input = 'i have 3 euros , how much is it in USD Dollars ?')
print(f'response_without_tool.content : {response_without_tool.content}')
