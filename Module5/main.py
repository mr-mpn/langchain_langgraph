'''Module 5: Introduction to Agents (LangChain-native)
What an "agent" is (LLM + tools + loop)
create_tool_calling_agent / AgentExecutor
Limitations of this approach — why LangGraph exists
AgentExecutor -> does all the things we did manually in Module2 by default , without the overhead

create_tool_calling_agent and AgentExecutor are deprecated in langchain 1.x and were moved
to the langchain-classic package (imported here from langchain_classic.agents).
'''

from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

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

agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use tools when needed."),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(model, tools, agent_prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = agent_executor.invoke({
    "input": "i have 3 euros , how much is it in USD Dollars ?"
})
print(f'result : {result}')
print(f'result["output"] : {result["output"]}')