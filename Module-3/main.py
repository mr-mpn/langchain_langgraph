'''
Module 3: Memory & State
Why "memory" is really just state management
Conversation history patterns
RunnableWithMessageHistory
Persisting state (in-memory vs. external stores)

LangChainDeprecationWarning: RunnableWithMessageHistory is deprecated. Use LangGraph's built-in persistence instead.

This code would not work nor it is best practice ! 
'''

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import MessagesPlaceholder


load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

chain = prompt | model 

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)


config = {"configurable": {"session_id": "marco-session-1"}}

chain_with_history.invoke({"question": "My name is Marco."}, config=config)
chain_with_history.invoke({"question": "What's my name?"}, config=config)