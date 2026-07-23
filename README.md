# tutorial_ai

A hands-on learning repo for LangChain / LangGraph fundamentals, working through core concepts module by module:

- **Module 1** — Basic chat model calls
- **Module 2** — Tools & function calling (`@tool`, `bind_tools`, manual tool-calling loop, Pydantic tool inputs)
- **Module 3** — Memory & state (conversation history patterns, `RunnableWithMessageHistory`)
- **Module 4** — Retrieval-Augmented Generation (RAG): document loaders, text splitters, embeddings, vector stores, retrieval chains with LCEL
- **Module 5** — Introduction to agents (`create_tool_calling_agent`, `AgentExecutor` from `langchain-classic`, since langchain 1.x moved these legacy agent APIs out of `langchain.agents`)

All modules share a single `uv`-managed environment and a root-level `.env` for API keys.

## Setup

```
uv init
uv add langchain langchain-openai langgraph python-dotenv pydantic langchain-chroma langchain-community pypdf langchain-classic
```

Copy `.env.example` to `.env` and fill in your OpenAI API key:

```
cp .env.example .env
```

```
OPENAI_API_KEY=sk-...
```

## Running the modules

```
uv run .\Module1\main.py
uv run .\Module2\main.py
uv run .\Module3\main.py
uv run .\Module4\main.py
uv run .\Module5\main.py
```