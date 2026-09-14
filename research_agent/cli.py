from __future__ import annotations

import argparse
import logging
import sys

from research_agent.agent import ResearchAgent, write_outputs
from research_agent.config import DEFAULT_MAX_SOURCES, load_settings
from research_agent.fetch import HttpFetcher
from research_agent.llm import OpenAICompatLLM
from research_agent.logging_setup import configure_logging
from research_agent.search import DuckDuckGoSearcher

log = logging.getLogger("research_agent.cli")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m research_agent",
        description=(
            "Read-only multi-hop research agent. Plans queries, fetches public pages, "
            "grades sources, builds a citation graph, and writes a cited Markdown report."
        ),
        epilog=(
            'Example: python -m research_agent "GLP-1 drugs and cardiovascular outcomes" '
            "--max-sources 6 --output report.md"
        ),
    )
    parser.add_argument("topic", help="Research topic or question")
    parser.add_argument(
        "--max-sources",
        type=int,
        default=DEFAULT_MAX_SOURCES,
        metavar="N",
        help=f"Maximum sources to keep after grading (default {DEFAULT_MAX_SOURCES})",
    )
    parser.add_argument(
        "--output",
        default="research_report.md",
        help="Markdown report path (default: research_report.md)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override OPENAI_MODEL for this run",
    )
    parser.add_argument(
        "--graph-json",
        default=None,
        help="Citation graph JSON path (default: alongside --output)",
    )
    parser.add_argument(
        "--no-mermaid",
        action="store_true",
        help="Omit the Mermaid diagram from the report",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Debug logging",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(verbose=args.verbose)
    if args.max_sources < 1:
        log.error("--max-sources must be >= 1")
        return 2
    settings = load_settings(
        model=args.model,
        max_sources=args.max_sources,
        include_mermaid=not args.no_mermaid,
    )
    try:
        llm = OpenAICompatLLM(settings)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    agent = ResearchAgent(
        settings=settings,
        llm=llm,
        searcher=DuckDuckGoSearcher(),
        fetcher=HttpFetcher(settings),
    )
    try:
        result = agent.run(args.topic)
    except Exception as exc:  # noqa: BLE001
        log.error("research failed: %s", exc)
        return 1
    report_path, graph_path = write_outputs(result, args.output, args.graph_json)
    print(f"Report: {report_path}")
    print(f"Graph:  {graph_path}")
    print(
        f"Sources: {len(result.sources)}  "
        f"Claims: {len(result.claims)}  "
        f"Contradictions: {len(result.contradictions)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
