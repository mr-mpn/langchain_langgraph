'''
How to get Json Responses from OPENAI :
- use .with_structured_output(schema=None, method="json_mode")
- be sure that the prompt contains the word "json" or openai will refuse the api call 
'''

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

load_dotenv()

prompt= '''
what is your name and how old are you in json format 
'''

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
response_model = model.invoke(input=prompt)


structured_model = model.with_structured_output(schema=None, method="json_mode")
response_structured_model = structured_model.invoke(input=prompt)

class response(BaseModel):
    name:str = Field(description="name of the model")
    age: str = Field(description="age of the model")
schema_structured_model = model.with_structured_output(schema=response, method="json_mode")
response_schema_structured_model = schema_structured_model.invoke(input=prompt)


print(f'response_model : {response_model.content}')
print(f'response_structured_model : {response_structured_model}')
print(f'response_schema_structured_model : {response_schema_structured_model}')
