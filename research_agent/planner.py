from __future__ import annotations

import logging

from research_agent.llm import LLMClient, parse_json_object
from research_agent.models import ResearchPlan, Source, SubQuestion

log = logging.getLogger("research_agent.planner")

PLANNER_SYSTEM = """You are a research planner for a read-only web research agent.
Produce a multi-hop plan that decomposes the topic into sequential sub-questions.
Each hop should depend on what the previous hop would typically uncover.
Return JSON only with this shape:
{
  "rationale": "why this sequence",
  "sub_questions": [
    {
      "question": "specific sub-question",
      "queries": ["web search query", "alternate query"],
      "hop": 1,
      "rationale": "what this hop unlocks"
    }
  ]
}
Rules:
- 3 hops. Hop 1 = definitions/background, hop 2 = evidence/mechanisms, hop 3 = debate/limits/recent change.
- 1-2 concise search queries per sub-question. Prefer queries that retrieve primary or high-quality sources.
- No instructions to buy, download binaries, log in, or take irreversible actions.
"""

FOLLOWUP_SYSTEM = """You refine a research plan after seeing early sources.
Return JSON only:
{
  "queries": ["new search query", "another query"],
  "focus": "what gap these queries close"
}
Propose 1-3 queries that fill missing evidence or hunt for dissenting sources.
Do not repeat queries already used.
"""


def plan_research(llm: LLMClient, topic: str) -> ResearchPlan:
    user = f"Topic: {topic}\nWrite a 3-hop research plan."
    raw = llm.complete(PLANNER_SYSTEM, user, json_mode=True)
    data = parse_json_object(raw)
    questions: list[SubQuestion] = []
    for item in data.get("sub_questions") or []:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question") or "").strip()
        queries = [
            str(q).strip()
            for q in (item.get("queries") or [])
            if str(q).strip()
        ]
        if not question or not queries:
            continue
        hop = int(item.get("hop") or (len(questions) + 1))
        questions.append(
            SubQuestion(
                question=question,
                queries=queries[:2],
                hop=max(1, hop),
                rationale=str(item.get("rationale") or ""),
            )
        )
    if not questions:
        questions = _fallback_plan(topic)
        log.warning("Planner JSON missing sub-questions; using heuristic plan")
    plan = ResearchPlan(
        topic=topic,
        sub_questions=questions,
        rationale=str(data.get("rationale") or ""),
    )
    log.info(
        "plan ready — %d sub-questions across hops %s",
        len(plan.sub_questions),
        sorted({q.hop for q in plan.sub_questions}),
    )
    return plan


def followup_queries(
    llm: LLMClient,
    topic: str,
    used_queries: list[str],
    sources: list[Source],
    hop: int,
) -> list[str]:
    notes = []
    for source in sources[:8]:
        excerpt = " ".join(source.text.split())[:280]
        notes.append(f"- {source.id} {source.title}: {excerpt}")
    user = (
        f"Topic: {topic}\nNext hop: {hop}\n"
        f"Already used queries:\n- "
        + "\n- ".join(used_queries or ["(none)"])
        + "\n\nSources so far:\n"
        + ("\n".join(notes) if notes else "(none yet)")
    )
    raw = llm.complete(FOLLOWUP_SYSTEM, user, json_mode=True)
    data = parse_json_object(raw)
    queries = [str(q).strip() for q in (data.get("queries") or []) if str(q).strip()]
    log.info("follow-up hop %s — %d new queries (%s)", hop, len(queries), data.get("focus"))
    return queries[:3]


def _fallback_plan(topic: str) -> list[SubQuestion]:
    return [
        SubQuestion(
            question=f"What is {topic} and why does it matter?",
            queries=[f"{topic} overview", f"{topic} definition"],
            hop=1,
            rationale="Establish the frame and vocabulary.",
        ),
        SubQuestion(
            question=f"What evidence exists about {topic}?",
            queries=[f"{topic} research evidence", f"{topic} systematic review"],
            hop=2,
            rationale="Collect empirical support.",
        ),
        SubQuestion(
            question=f"Where do experts disagree about {topic}?",
            queries=[f"{topic} controversy", f"{topic} limitations criticism"],
            hop=3,
            rationale="Surface contradictions and open questions.",
        ),
    ]
