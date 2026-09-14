# Deep Research Agent

A read-only CLI research agent. It takes a topic, plans multi-hop questions, searches the public web, fetches and grades sources, builds a citation graph, flags contradictions, and writes a fully cited Markdown report.

This is an MVP: one agent loop, no UI, no database, no write actions against the web.

## What it does

1. **Plans** 3 hops of sub-questions and search queries with an OpenAI-compatible LLM.
2. **Searches** DuckDuckGo (free) and **fetches** pages with `httpx` + `trafilatura`.
3. **Grades** each source on credibility, relevance, and recency.
4. **Extracts claims**, links them into a **citation graph** (support / contradict), and **detects contradictions** with pointers to both sides.
5. **Writes** a Markdown report with footnotes, a sources appendix, a graph summary, and optional Mermaid. The graph is also exported as JSON.

Defaults are human-safe: GET-only fetches, public `http(s)` URLs, timeouts, size caps, and no form submission or code execution from pages.

## Requirements

- Python 3.11+
- An OpenAI-compatible API key for live runs

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Copy `.env.example` to `.env` and set:

```bash
OPENAI_API_KEY=sk-...
# optional
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

`OPENAI_BASE_URL` can point at Azure OpenAI, Groq, Ollama, OpenRouter, or any `/v1/chat/completions` server.

## Run

```bash
python -m research_agent "GLP-1 drugs and cardiovascular outcomes" \
  --max-sources 6 \
  --output research_report.md \
  --model gpt-4o-mini
```

Equivalent entry point after install:

```bash
research-agent "GLP-1 drugs and cardiovascular outcomes" --max-sources 6
```

Flags:

| Flag | Meaning |
| --- | --- |
| `--max-sources` | Cap on graded sources kept in the report |
| `--output` | Markdown report path |
| `--model` | Per-run model override |
| `--graph-json` | Optional graph JSON path (defaults to `<output>.graph.json`) |
| `--no-mermaid` | Skip the Mermaid block |
| `-v` | Debug logs on stderr |

The CLI prints the report path, graph path, and source/claim/contradiction counts. Progress logs go to stderr.

## Report shape

- Executive summary and cited sections (`[^1]` footnotes)
- Research plan (hops and queries)
- Contradictions with side A / side B source pointers
- Citation graph summary, plus Mermaid
- Footnote list and a source appendix with grades and extracted claims

## Tests

Unit tests mock the LLM, search, and fetch layers. No network or API key is required.

```bash
pip install -e ".[dev]"
pytest
```

## Layout

```
research_agent/     package (planner, tools, graph, report, CLI)
tests/              mocked unit tests
pyproject.toml
.env.example
```

## Limits

- Search quality depends on DuckDuckGo HTML results.
- Grading is heuristic (domain + lexical overlap + date), not a trust oracle.
- The agent does not log in, click paywalls, or execute page JavaScript.
- Live quality tracks whatever model you point `OPENAI_BASE_URL` at.
