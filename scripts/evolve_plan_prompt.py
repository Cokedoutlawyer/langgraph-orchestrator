"""
Darwinian evolver problem: evolve the langgraph-orchestrator PLAN_SYSTEM prompt.

Organism: the PLAN_SYSTEM string (the system prompt fed to the LLM in the plan node).
Evaluator: runs the plan node with the evolved prompt on 5 test queries, scores the
           resulting JSON plan against deterministic expected characteristics.
Mutator: LLM proposes an improved prompt given a failure case (query + expected + actual).

Uses the Blackbox LiteLLM proxy (api.blackbox.ai/v1, z-ai/glm-5.2) — the same LLM
the orchestrator uses — so the evolved prompt is optimized for the actual production model.

Run:
  cd ~/.hermes/cache/darwinian-evolver/darwinian_evolver
  EVOLVER_MODEL=z-ai/glm-5.2 uv run --with openai python \
    /home/peter/projects/langgraph-orchestrator/scripts/evolve_plan_prompt.py \
    --num_iterations 4 --num_parents_per_iteration 2 \
    --mutator_concurrency 2 --evaluator_concurrency 2 \
    --output_dir /tmp/evolve_plan
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from openai import OpenAI

from darwinian_evolver.cli_common import (
    build_hyperparameter_config_from_args,
    parse_learning_log_view_type,
    register_hyperparameter_args,
)
from darwinian_evolver.evolve_problem_loop import EvolveProblemLoop
from darwinian_evolver.learning_log import LearningLogEntry
from darwinian_evolver.problem import (
    EvaluationFailureCase,
    EvaluationResult,
    Evaluator,
    Mutator,
    Organism,
    Problem,
)

# --- LLM client (Blackbox LiteLLM proxy, OpenAI-compatible) ---
DEFAULT_MODEL = os.environ.get("EVOLVER_MODEL", "z-ai/glm-5.2")
BLACKBOX_KEY = os.environ.get("BLACKBOX_API_KEY", "")
BLACKBOX_BASE = "https://api.blackbox.ai/v1"


def _client() -> OpenAI:
    if not BLACKBOX_KEY:
        sys.exit("BLACKBOX_API_KEY is not set")
    return OpenAI(api_key=BLACKBOX_KEY, base_url=BLACKBOX_BASE)


def _prompt_llm(system: str, user: str, max_tokens: int = 800, temperature: float = 0.1) -> str:
    """Call the Blackbox proxy. Never raises — returns error sentinel on failure."""
    try:
        r = _client().chat.completions.create(
            model=DEFAULT_MODEL,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return r.choices[0].message.content or ""
    except Exception as e:
        return f"<LLM_ERROR: {type(e).__name__}: {e}>"


def _extract_json(text: str) -> dict | None:
    """Defensively extract a JSON object from LLM output."""
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        if t.endswith("```"):
            t = t[:-3]
    first, last = t.find("{"), t.rfind("}")
    if first == -1 or last == -1:
        return None
    try:
        return json.loads(t[first : last + 1])
    except json.JSONDecodeError:
        return None


# --------------------------------------------------------------------------- #
# 1. ORGANISM — the PLAN_SYSTEM prompt being evolved.
# --------------------------------------------------------------------------- #
class PlanPromptOrganism(Organism):
    prompt: str

    def run(self, query: str) -> dict | None:
        """Run the plan node with this prompt on a query. Returns parsed plan dict."""
        raw = _prompt_llm(self.prompt, query)
        return _extract_json(raw)


# --------------------------------------------------------------------------- #
# 2. EVALUATOR — score plans against expected characteristics.
# --------------------------------------------------------------------------- #
# (query, expected_provider, expected_tool_category, needs_rendering, geo)
TEST_CASES = [
    ("What is Hermes Agent by Nous Research?",
     "brightdata", "search", False, None),
    ("Go to example.com and tell me what the page says",
     "brightdata", "scrape", False, None),  # simple static page, scrape is fine
    ("Find the cheapest PS5 listings on Amazon in the US",
     "brightdata", "scrape", False, "US"),  # geo-targeted product scrape
    ("Compare PS5 prices across Amazon, Walmart, and Best Buy",
     "brightdata", "search", False, None),  # multi-step, search first
    ("Summarize this article: https://example.com/blog/post",
     "brightdata", "scrape", False, None),  # url given -> scrape
]


class PlanFailureCase(EvaluationFailureCase):
    query: str
    expected: str
    actual: str
    score_breakdown: str


class PlanEvaluator(Evaluator[PlanPromptOrganism, EvaluationResult, PlanFailureCase]):
    def _score_one(self, plan: dict | None, query: str, exp_provider: str,
                   exp_tool_cat: str, exp_render: bool, exp_geo: str | None) -> tuple[float, str]:
        if plan is None:
            return 0.0, "invalid JSON (0.0)"
        score = 0.0
        parts = []
        # valid JSON with required keys (0.25)
        first = plan.get("first_step", {})
        if isinstance(first, dict) and "provider" in first and "tool" in first:
            score += 0.25
            parts.append("valid JSON +keys +0.25")
        else:
            parts.append("missing first_step keys +0.0")
            return score, "; ".join(parts)
        # provider is valid + correct (0.25)
        provider = first.get("provider", "")
        if provider in ("brightdata", "oxylabs", "venice", "camofox"):
            score += 0.1
            parts.append(f"valid provider({provider}) +0.1")
            if provider == exp_provider:
                score += 0.15
                parts.append(f"correct provider +0.15")
        else:
            parts.append(f"invalid provider({provider}) +0.0")
        # tool category matches (0.2)
        tool = first.get("tool", "")
        if exp_tool_cat in tool:
            score += 0.2
            parts.append(f"tool({tool}) matches({exp_tool_cat}) +0.2")
        else:
            parts.append(f"tool({tool}) != expected({exp_tool_cat}) +0.0")
        # needs_rendering correct (0.15)
        got_render = bool(plan.get("needs_rendering", False))
        if got_render == exp_render:
            score += 0.15
            parts.append(f"render({got_render}) correct +0.15")
        else:
            parts.append(f"render({got_render}) != {exp_render} +0.0")
        # geo correct (0.15)
        got_geo = plan.get("geo")
        if exp_geo is None:
            if not got_geo:
                score += 0.15
                parts.append("geo=None correct +0.15")
            else:
                parts.append(f"geo({got_geo}) should be None +0.0")
        else:
            if str(got_geo).upper() == exp_geo.upper():
                score += 0.15
                parts.append(f"geo({got_geo}) correct +0.15")
            else:
                parts.append(f"geo({got_geo}) != {exp_geo} +0.0")
        return min(score, 1.0), "; ".join(parts)

    def evaluate(self, organism: PlanPromptOrganism) -> EvaluationResult:
        train_fails: list[PlanFailureCase] = []
        total = 0.0
        for i, (query, exp_p, exp_t, exp_r, exp_g) in enumerate(TEST_CASES):
            plan = organism.run(query)
            s, breakdown = self._score_one(plan, query, exp_p, exp_t, exp_r, exp_g)
            total += s
            if s < 0.9:  # anything not near-perfect is a failure case
                actual = json.dumps(plan)[:300] if plan else "None"
                train_fails.append(PlanFailureCase(
                    query=query,
                    expected=f"provider={exp_p}, tool_cat={exp_t}, render={exp_r}, geo={exp_g}",
                    actual=actual,
                    score_breakdown=breakdown,
                    data_point_id=f"train_{i}",
                ))
        score = total / len(TEST_CASES)
        return EvaluationResult(
            score=score,
            trainable_failure_cases=train_fails,
            holdout_failure_cases=[],
            is_viable=True,
        )


# --------------------------------------------------------------------------- #
# 3. MUTATOR — LLM proposes an improved prompt given a failure case.
# --------------------------------------------------------------------------- #
class PlanPromptMutator(Mutator[PlanPromptOrganism, PlanFailureCase]):
    PROMPT = """\
You are optimizing a planning prompt for an AI web-research agent.

CURRENT PROMPT:
\"\"\"
{artifact}
\"\"\"

This prompt produced a BAD plan on this query:
Query: {input}

Expected plan characteristics: {expected}
Actual plan produced: {actual}

Score breakdown: {breakdown}

Diagnose why the current prompt led to that bad plan, then propose an IMPROVED
version of the planning prompt that would produce the correct plan. Keep the
same JSON output schema (needs_rendering, needs_geo, geo, steps, first_step
with provider/tool/args). Keep it concise. Do not remove the tool priority list.

Put the ENTIRE new prompt in the LAST triple-backtick block of your response.
""".strip()

    def mutate(
        self,
        organism: PlanPromptOrganism,
        failure_cases: list[PlanFailureCase],
        learning_log_entries: list[LearningLogEntry],
    ) -> list[PlanPromptOrganism]:
        fc = failure_cases[0]
        prompt = self.PROMPT.format(
            artifact=organism.prompt,
            input=fc.query,
            expected=fc.expected,
            actual=fc.actual,
            breakdown=fc.score_breakdown,
        )
        resp = _prompt_llm("You are a prompt optimization assistant.", prompt, max_tokens=1500)
        parts = resp.split("```")
        if len(parts) < 3:
            return []
        new_prompt = parts[-2].strip()
        # Strip a language tag line if present
        if "\n" in new_prompt:
            first_line, rest = new_prompt.split("\n", 1)
            if first_line.strip() in ("", "text", "markdown", "python") or len(first_line) < 15:
                new_prompt = rest.strip() if first_line.strip() in ("", "text", "markdown") else new_prompt
        if len(new_prompt) < 50:  # too short to be a real prompt
            return []
        return [PlanPromptOrganism(prompt=new_prompt)]


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
SEED_PROMPT = """You are a web-research planning agent. Analyze the user query and produce a JSON plan.

Available tools (in priority order):
1. brightdata.scrape_as_markdown(url, geo) — single page, markdown (PRIMARY)
2. brightdata.search_engine(query, geo) — Google/Bing/Yandex SERP
3. brightdata.scraping_browser_navigate(url) — full browser rendering
4. oxylabs.ai_scraper(url, geo) — FAILOVER on brightdata block/CAPTCHA/5xx
5. oxylabs.ai_search(query, geo) — FAILOVER search
6. venice.chat_with_scraping(prompt, urls) — last resort, <3 pages, needs LLM answer
7. camofox.navigate+snapshot — local stealth browser for JS-heavy pages

Decide:
- needs_rendering: true if the target is a JS-heavy SPA (React/Vue) or requires login/interaction
- needs_geo: true if the query specifies a country/region
- geo: ISO country code (e.g. "US", "GB") or null
- steps: ordered list of sub-steps (1-5 steps)
- For the FIRST step, pick the best provider+tool+args

Respond with ONLY this JSON shape:
{
  "needs_rendering": false,
  "needs_geo": false,
  "geo": null,
  "steps": ["step 1", "step 2"],
  "first_step": {
    "provider": "brightdata",
    "tool": "search_engine",
    "args": {"query": "...", "geo": null}
  }
}"""


def make_problem() -> Problem:
    initial = PlanPromptOrganism(prompt=SEED_PROMPT)
    return Problem[PlanPromptOrganism, EvaluationResult, PlanFailureCase](
        evaluator=PlanEvaluator(),
        mutators=[PlanPromptMutator()],
        initial_organism=initial,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    register_hyperparameter_args(ap.add_argument_group("hyperparameters"))
    ap.add_argument("--num_iterations", type=int, default=4)
    ap.add_argument("--mutator_concurrency", type=int, default=2)
    ap.add_argument("--evaluator_concurrency", type=int, default=2)
    ap.add_argument("--output_dir", type=str, required=True)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "snapshots").mkdir(exist_ok=True)

    hp = build_hyperparameter_config_from_args(args)
    loop = EvolveProblemLoop(
        problem=make_problem(),
        learning_log_view_type=parse_learning_log_view_type(hp.learning_log_view_type),
        num_parents_per_iteration=hp.num_parents_per_iteration,
        mutator_concurrency=args.mutator_concurrency,
        evaluator_concurrency=args.evaluator_concurrency,
        fixed_midpoint_score=hp.fixed_midpoint_score,
        midpoint_score_percentile=hp.midpoint_score_percentile,
        sharpness=hp.sharpness,
        novelty_weight=hp.novelty_weight,
        batch_size=hp.batch_size,
        should_verify_mutations=hp.verify_mutations,
    )

    print("Evaluating seed prompt...")
    best_score = 0.0
    best_prompt = SEED_PROMPT
    for snap in loop.run(num_iterations=args.num_iterations):
        (out / "snapshots" / f"iteration_{snap.iteration}.pkl").write_bytes(snap.snapshot)
        org, res = snap.best_organism_result
        print(f"iter={snap.iteration} pop={snap.population_size} best_score={res.score:.3f}")
        if res.score > best_score:
            best_score = res.score
            best_prompt = org.prompt

    # Save the winning prompt
    (out / "best_prompt.txt").write_text(best_prompt, encoding="utf-8")
    (out / "seed_prompt.txt").write_text(SEED_PROMPT, encoding="utf-8")
    print(f"\nBest score: {best_score:.3f}")
    print(f"Best prompt saved: {out / 'best_prompt.txt'}")
    print(f"Seed prompt saved: {out / 'seed_prompt.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
