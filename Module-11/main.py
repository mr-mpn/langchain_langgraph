'''
Module 11: Intent-based routing with LangGraph

Classifies each user message into "chat", "knowledge", or "code" using structured
output, then routes to a dedicated agent node for that intent (conditional edges).
A checkpointer (InMemorySaver) keeps conversation history across turns in the loop,
keyed by a per-run thread_id.
'''

import sys
from typing import TypedDict , Annotated , Literal
from pydantic import BaseModel, Field
import uuid
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START,END
from langgraph.checkpoint.memory import InMemorySaver


sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

llm = init_chat_model(model = 'openai:gpt-4.1-mini')


class IntentClassifier(BaseModel):
    message_intent : Literal['chat','knowledge','code'] = Field(..., description='Classify where the user wants to just chat , ask for knowledge or chagne code in the project')

class State(TypedDict):
    messages : Annotated[list, add_messages]
    message_intent : str | None

################### Nodes ##################
def classify_intent(state : State):
    structured_llm = llm.with_structured_output(IntentClassifier)

    result = structured_llm.invoke([
        {'role' : 'system' , 'content' : 'Determine / classify whether the user wants to chat ("chat") , retrieve a knowledge ("knowledge") or change code ("code")'},
        {'role' : 'user' , 'content' : state['messages'][-1].content}
    ])

    return {'message_intent' : result.message_intent}


def prompt_llm_chat(state:State):
    messages = [{'role' : 'system' , 'content' : 'You are a talkative chatbot for fun. Be nice'}] + state['messages']

    response = llm.invoke(input=messages)

    return {'messages' : [{'role':'assistant' , 'content' : response.content}]}


def prompt_llm_rag(state:State):
    messages = [{'role' : 'system' , 'content' : 'No matter what the user says always say "I am the Rag Agent"'}] + state['messages']

    response = llm.invoke(input=messages)

    return {'messages' : [{'role':'assistant' , 'content' : response.content}]}



def prompt_llm_code(state:State):
    messages = [{'role' : 'system' , 'content' : 'No matter what the user says always say "I am the Coding Agent"'}] + state['messages']

    response = llm.invoke(input=messages)

    return {'messages' : [{'role':'assistant' , 'content' : response.content}]}


############# Create the Graph ##################

graph_builder = StateGraph(State)

graph_builder.add_node('classifier' , classify_intent)
graph_builder.add_node('chat_agent' , prompt_llm_chat)
graph_builder.add_node('rag_agent' ,prompt_llm_rag)
graph_builder.add_node('coding_aget' , prompt_llm_code)

graph_builder.add_edge(START , 'classifier')
graph_builder.add_conditional_edges('classifier' , lambda state:state['message_intent'] , {
    'chat': 'chat_agent',
    'knowledge': 'rag_agent',
    'code': 'coding_aget'
    })


graph_builder.add_edge('chat_agent' , END)
graph_builder.add_edge('rag_agent' , END)
graph_builder.add_edge('coding_aget' , END)

checkpointer = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)

config = {'configurable' : {'thread_id' : str(uuid.uuid4())}}

while True :
    user_message = input("Enter message:")
    result = graph.invoke({'messages': [{'role':'user' , 'content' : user_message}]} , config=config)

    print(result['messages'][-1].content)