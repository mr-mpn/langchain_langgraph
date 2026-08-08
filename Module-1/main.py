'''
Module1 : Module 1: LangChain Core Building Blocks
Chat Models & Messages — HumanMessage, AIMessage, SystemMessage
Prompt Templates — PromptTemplate, ChatPromptTemplate, few-shot prompting
Output Parsers — structured output, Pydantic parsing, with_structured_output
LCEL (LangChain Expression Language) — the | pipe syntax, Runnable interface
Runnables in depth — RunnablePassthrough, RunnableLambda, RunnableParallel
'''

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


load_dotenv()


class Answer(BaseModel):
    summary: str = Field(description="one sentence summary")
    difficulty: str = Field(description="beginner, intermediate, or advanced")


model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
structured_model = model.with_structured_output(Answer)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a funny {domain} expert."),
    ("human", "{question}"),
])

chain = prompt | structured_model

response = chain.invoke({"domain" : "Medicine" , "question" : "what is the cure of sadness"})
print(response)