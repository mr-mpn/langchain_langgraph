'''
Module 4: Retrieval-Augmented Generation (RAG)
Document loaders & text splitters
Embeddings & vector stores (Chroma/FAISS)
Building a retriever
Retrieval chains with LCEL
Advanced retrieval (multi-query, re-ranking, self-query) — brief overview

'''


from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# --- Step 1: Load the document ---
notes_path = Path(__file__).parent / "notes.txt"
loader = TextLoader(notes_path)
docs = loader.load()

# --- Step 2: Split into chunks ---
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(docs)
print(f"Split into {len(chunks)} chunks\n")

# --- Step 3: Embed and store in a vector DB ---
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
)

# --- Step 4: Build a retriever ---
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# --- Step 5: Build the RAG chain with LCEL ---
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

rag_prompt = ChatPromptTemplate.from_template("""
Answer the question using only the following context.
If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}
""")

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | model
    | StrOutputParser()
)

# --- Step 6: Ask questions ---
questions = [
    "What did I write about generators?",
    "What is a decorator used for?",
    "What does the notes say about async/await?",  # not in notes.txt on purpose
]

for q in questions:
    print(f"Q: {q}")
    print(f"A: {rag_chain.invoke(q)}\n")