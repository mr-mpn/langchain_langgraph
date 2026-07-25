# tutorial_ai

A hands-on, learn-by-running repo for LangChain / LangGraph fundamentals. Each module is a single, runnable `main.py` that builds on the last — start at Module 1 and work down.

## Table of contents

- [Quickstart](#quickstart)
- [Modules](#modules)
- [Running a module](#running-a-module)
- [Notes & gotchas](#notes--gotchas)

## Quickstart

1. **Install [uv](https://docs.astral.sh/uv/)** if you don't have it yet.
2. **Install dependencies**
   ```
   uv sync
   ```
   (or, to set this up from scratch: `uv init` then `uv add langchain langchain-openai langgraph python-dotenv pydantic langchain-chroma langchain-community pypdf langchain-classic grandalf`)
3. **Set up your API key** — copy the example env file and fill in your own key:
   ```
   cp .env.example .env
   ```
   ```
   OPENAI_API_KEY=sk-...
   ```
   The `.env` lives at the repo root and is shared by every module (it's git-ignored, so your key never gets committed).
4. **Run any module**
   ```
   uv run .\Module1\main.py
   ```

## Modules

<details>
<summary><strong>Module 1 — Your first chat model call</strong></summary>

Basic `ChatOpenAI` invocation — the "hello world" of LangChain.

```
uv run .\Module1\main.py
```

**Extra:** `Module1\Extra\main.py` — structured/JSON output with `with_structured_output`.
</details>

<details>
<summary><strong>Module 2 — Tools & function calling</strong></summary>

Defining tools with `@tool`, binding them to a model with `bind_tools`, and running the manual tool-calling loop (append the tool call, run it, append a `ToolMessage`, ask again) before agents abstract it away. Also covers structured tool inputs with Pydantic.

```
uv run .\Module2\main.py
```

**Extra:** `Module2\Extra\main.py` — a currency-exchange tool worked through the same loop.
</details>

<details>
<summary><strong>Module 3 — Memory & state</strong></summary>

Why "memory" is really just state management: conversation history patterns and `RunnableWithMessageHistory`.

> This module intentionally shows a deprecated/non-best-practice pattern (`RunnableWithMessageHistory`) as a stepping stone — LangGraph's built-in persistence is the modern replacement.

```
uv run .\Module3\main.py
```
</details>

<details>
<summary><strong>Module 4 — Retrieval-Augmented Generation (RAG)</strong></summary>

Document loaders, text splitters, embeddings, vector stores (Chroma), building a retriever, and wiring it all into a retrieval chain with LCEL.

```
uv run .\Module4\main.py
```
</details>

<details>
<summary><strong>Module 5 — Introduction to agents</strong></summary>

What an "agent" is (LLM + tools + loop), using `create_tool_calling_agent` / `AgentExecutor` to get that loop for free, and where this approach hits its limits (why LangGraph exists).

> `create_tool_calling_agent` and `AgentExecutor` are deprecated in `langchain` 1.x and now live in the separate `langchain-classic` package.

```
uv run .\Module5\main.py
```
</details>

<details>
<summary><strong>Module 6 — LangGraph basics</strong></summary>

Graphs vs. chains and why control flow matters: `StateGraph`, nodes, edges, typed state (`TypedDict`), and building a simple linear graph.

```
uv run .\Module6\main.py
```

**Extra:** `Module6\Extra\main.py` — a small graph that converts a price to EUR and decides if it's in budget.
</details>

<details>
<summary><strong>Module 7 — LangGraph control flow</strong></summary>

Conditional edges (routing logic), cycles/loops (the key differentiator from plain LangChain chains), and parallel branches with fan-in.

```
uv run .\Module7\main.py
```
</details>

<details>
<summary><strong>Module 8 — coming soon</strong></summary>

Not started yet.
</details>

## Running a module

Every module is self-contained — just point `uv run` at it:

```
uv run .\Module1\main.py
uv run .\Module2\main.py
uv run .\Module3\main.py
uv run .\Module4\main.py
uv run .\Module5\main.py
uv run .\Module6\main.py
uv run .\Module7\main.py
```

`Extra` scripts (bonus/variant examples) run the same way, e.g. `uv run .\Module2\Extra\main.py`.

## Notes & gotchas

- **One shared `.env` at the repo root.** Every module calls `load_dotenv()`, which walks up from the script's own location to find it — so it works no matter which directory you run `uv run` from.
- **File paths should be relative to the script, not the cwd.** If a module reads a local file (e.g. Module 4's `notes.txt`), resolve it with `Path(__file__).parent / "notes.txt"` rather than a bare relative string.
- **`langchain` 1.x moved things around.** Legacy agent APIs (`create_tool_calling_agent`, `AgentExecutor`) live in `langchain-classic` now, not `langchain.agents`.
- **ASCII graph rendering** (`graph.get_graph().draw_ascii()`) needs the `grandalf` package installed — you don't import it directly, it's just a dependency LangGraph uses internally.
