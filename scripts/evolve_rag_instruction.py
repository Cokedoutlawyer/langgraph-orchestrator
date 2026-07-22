"""
Darwinian evolver problem: evolve the RAG_INJECTION_PROMPT used by the summary node
to weave Engram-retrieved long-term memory into the final answer.

Organism: the RAG_INJECTION_PROMPT instruction string.
Evaluator: for each test case (memories + scraped_content + query + expected_keyword),
           run the summary LLM with the evolved instruction, score whether the output
           mentions the expected user-preference keyword AND stays grounded.
Mutator: LLM proposes an improved instruction given a failure case.

Uses the Blackbox LiteLLM proxy (z-ai/glm-5.2) — same model the orchestrator uses.
"""
from __future__ import annotations

import argparse
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

DEFAULT_MODEL = os.environ.get("EVOLVER_MODEL", "z-ai/glm-5.2")
BLACKBOX_KEY = os.environ.get("BLACKBOX_API_KEY", "")
BLACKBOX_BASE = "https://api.blackbox.ai/v1"


def _client() -> OpenAI:
    if not BLACKBOX_KEY:
        sys.exit("BLACKBOX_API_KEY is not set")
    return OpenAI(api_key=BLACKBOX_KEY, base_url=BLACKBOX_BASE)


def _llm(system: str, user: str, max_tokens: int = 600, temperature: float = 0.3) -> str:
    try:
        r = _client().chat.completions.create(
            model=DEFAULT_MODEL, max_tokens=max_tokens, temperature=temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return r.choices[0].message.content or ""
    except Exception as e:
        return f"<LLM_ERROR: {type(e).__name__}: {e}>"


# Fixed test cases: (query, scraped_content, rag_memory_block, strict_checks)
# Each check is (description, regex_or_substring, must_be_present_bool, weight)
# Seed must FAIL several of these so the evolver has trainable failure cases.
# (If the seed aces everything, sample_parents crashes with "No eligible organisms".)
TEST_CASES = [
    (
        "Where can I find good coffee in Berlin?",
        "Berlin coffee guide: The Barn (specialty roaster, Mitte), Five Elephant (roastery + cafe, Kreuzberg), "
        "Bonanza Coffee Heroes (specialty, Prenzlauer Berg), Cafe Kraft, No Fire No Glory. "
        "Also: Father Carpenter, Ben Rahim in Mitte. Note: Starbucks and Dunkin are also widely available.",
        "[1] (score=0.98) User prefers specialty coffee and is not interested in chain coffee shops.\n"
        "[2] (score=0.95) User's favorite coffee shop so far is Bonanza.",
        [
            ("mentions_bonanza", "bonanza", True, 0.2),
            ("mentions_specialty", "specialty", True, 0.15),
            ("excludes_chains", "starbucks", False, 0.2),  # must NOT recommend a chain given the preference
            ("excludes_dunkin", "dunkin", False, 0.15),
            ("flags_preference", "since you", True, 0.15),  # explicitly anchors recs to the stated preference
            ("grounded_tier", "barn", True, 0.15),          # pulls a real scraped rec
        ],
    ),
    (
        "Best places to sell trading cards online?",
        "Top card marketplaces: eBay (largest secondary market, auction + buy-it-now), "
        "CardMarket (Europe), TCGplayer (TCG focus — Magic/Yu-Gi-Oh/Pokemon), COMC (sports cards), "
        "StarStock (sports). eBay has the biggest buyer pool for sports cards. "
        "TCGplayer is the dominant TCG marketplace.",
        "[1] (score=0.97) User sells sports and wrestling cards on eBay (baseball, football, basketball, hockey, WWE, AEW). No TCG.",
        [
            ("mentions_ebay", "ebay", True, 0.2),
            ("mentions_sports", "sports", True, 0.15),
            ("excludes_tcgplayer", "tcgplayer", False, 0.2),   # must NOT push the TCG platform given "No TCG"
            ("excludes_pokemon", "pokemon", False, 0.15),       # must not recommend TCG categories
            ("flags_preference", "since you", True, 0.15),
            ("grounded_comc", "comc", True, 0.15),
        ],
    ),
    (
        "Recommend tools for a security assessment",
        "Security tools: Burp Suite (web app testing), Metasploit (exploitation), "
        "Nmap (discovery), Wireshark (traffic analysis), Cobalt Strike (red team C2), "
        "BloodHound (AD attack paths). Also: Nessus (vulnerability scanner), Acunetix.",
        "[1] (score=0.96) User is a security researcher and red-teamer. Prefers offensive/red-team tooling.",
        [
            ("mentions_red_team", "red", True, 0.2),
            ("mentions_cobalt", "cobalt", True, 0.15),          # red-team specific, not generic scanning
            ("excludes_acunetix", "acunetix", False, 0.2),       # must not lead with generic scanner
            ("excludes_nessus_lead", "nessus", False, 0.15),     # defensive scanner, not red-team
            ("flags_preference", "since you", True, 0.15),
            ("grounded_metasploit", "metasploit", True, 0.15),
        ],
    ),
]


SUMMARY_SYSTEM = "You are a research summarization agent. Produce a clear, grounded answer."


# --------------------------------------------------------------------------- #
# 1. ORGANISM — the RAG injection instruction.
# --------------------------------------------------------------------------- #
class RagInstructionOrganism(Organism):
    instruction: str

    def run(self, query: str, content: str, memory_block: str) -> str:
        """Run the summary LLM with this RAG injection instruction."""
        user = (
            f"User query: {query}\n\n"
            f"Relevant long-term memory (user context from prior sessions):\n{memory_block}\n---\n"
            f"{self.instruction}\n\n"
            f"Scraped content ({len(content)} chars):\n---\n{content}\n---\n\n"
            f"Produce a grounded answer to the user query."
        )
        return _llm(SUMMARY_SYSTEM, user)


# --------------------------------------------------------------------------- #
# 2. EVALUATOR — strict checks: required substrings present + forbidden absent.
# --------------------------------------------------------------------------- #
class RagFailureCase(EvaluationFailureCase):
    query: str
    expected: str
    actual: str
    failed_checks: str


class RagEvaluator(Evaluator[RagInstructionOrganism, EvaluationResult, RagFailureCase]):
    def evaluate(self, organism: RagInstructionOrganism) -> EvaluationResult:
        train_fails: list[RagFailureCase] = []
        total = 0.0
        for i, (query, content, mem, checks) in enumerate(TEST_CASES):
            out = organism.run(query, content, mem)
            out_low = out.lower()
            case_score = 0.0
            failed: list[str] = []
            for desc, needle, must_present, weight in checks:
                present = needle.lower() in out_low
                ok = present if must_present else (not present)
                if ok:
                    case_score += weight
                else:
                    direction = "should mention" if must_present else "should NOT mention"
                    failed.append(f"{desc}: {direction} '{needle}'")
            case_score = min(case_score, 1.0)
            total += case_score
            if case_score < 1.0:
                train_fails.append(RagFailureCase(
                    query=query,
                    expected="all strict checks pass",
                    actual=out[:280],
                    failed_checks="; ".join(failed),
                    data_point_id=f"rag_{i}",
                ))
        score = total / len(TEST_CASES)
        return EvaluationResult(
            score=score,
            trainable_failure_cases=train_fails,
            holdout_failure_cases=[],
            is_viable=True,
        )


# --------------------------------------------------------------------------- #
# 3. MUTATOR — LLM proposes an improved instruction.
# --------------------------------------------------------------------------- #
class RagInstructionMutator(Mutator[RagInstructionOrganism, RagFailureCase]):
    PROMPT = """\
You are optimizing the instruction that tells a summarizer how to use retrieved
long-term memory when writing its answer.

CURRENT INSTRUCTION:
\"\"\"
{artifact}
\"\"\"

This instruction produced a BAD summary on this case:
Query: {input}
Strict checks that FAILED: {failed_checks}
Actual summary produced: {actual}

The checks encode two goals: (a) the summary must weave in relevant user
preferences from memory and explicitly anchor recommendations to them, and
(b) it must NOT mention things the user explicitly dislikes or irrelevant
generic options, even if they appear in the scraped content.

Diagnose why the current instruction failed these checks, then propose an
IMPROVED instruction that reliably makes the summarizer (a) mention relevant
user preferences from memory and anchor recs to them, (b) avoid mentioning
things the user dislikes/irrelevant categories, (c) stay grounded in scraped
content. Keep it under 4 sentences.

Put the ENTIRE new instruction in the LAST triple-backtick block.
""".strip()

    def mutate(
        self,
        organism: RagInstructionOrganism,
        failure_cases: list[RagFailureCase],
        learning_log_entries: list[LearningLogEntry],
    ) -> list[RagInstructionOrganism]:
        fc = failure_cases[0]
        prompt = self.PROMPT.format(
            artifact=organism.instruction,
            input=fc.query,
            failed_checks=fc.failed_checks,
            actual=fc.actual,
        )
        resp = _llm("You are a prompt optimization assistant.", prompt, max_tokens=900)
        parts = resp.split("```")
        if len(parts) < 3:
            return []
        new = parts[-2].strip()
        if "\n" in new:
            first, rest = new.split("\n", 1)
            if first.strip() in ("", "text", "markdown") or len(first) < 12:
                new = rest.strip() if first.strip() in ("", "text", "markdown") else new
        if len(new) < 20:
            return []
        return [RagInstructionOrganism(instruction=new)]


# --------------------------------------------------------------------------- #
SEED_INSTRUCTION = (
    "Use this memory to personalize/ground the answer where relevant. "
    "Prefer scraped content for factual currency, but respect stated user "
    "preferences from memory."
)


def make_problem() -> Problem:
    initial = RagInstructionOrganism(instruction=SEED_INSTRUCTION)
    return Problem[RagInstructionOrganism, EvaluationResult, RagFailureCase](
        evaluator=RagEvaluator(),
        mutators=[RagInstructionMutator()],
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

    print("Evaluating seed RAG instruction...")
    best_score = 0.0
    best_instr = SEED_INSTRUCTION
    for snap in loop.run(num_iterations=args.num_iterations):
        org, res = snap.best_organism_result
        print(f"iter={snap.iteration} pop={snap.population_size} best_score={res.score:.3f}")
        if res.score > best_score:
            best_score = res.score
            best_instr = org.instruction

    (out / "best_instruction.txt").write_text(best_instr, encoding="utf-8")
    (out / "seed_instruction.txt").write_text(SEED_INSTRUCTION, encoding="utf-8")
    print(f"\nBest score: {best_score:.3f}")
    print(f"Best instruction saved: {out / 'best_instruction.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
