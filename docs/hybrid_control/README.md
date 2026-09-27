# Deterministic hybrid control plane 1.0.0

This package implements the twelve-step control contract in the existing
LangGraph Orchestrator repository. Python 3.11+; the control-plane runtime and
its gate use the standard library only.

## Run

```bash
python scripts/hybrid_gate.py --output gate-report.json
python scripts/hybrid_demo.py --db demo.db --release
python scripts/hybrid_demo.py --db demo.db --release
python scripts/hybrid_dashboard.py --db demo.db --run-id golden-v1 --output dashboard.html
```

Both demo executions end at `VERIFIED`; the durable local sink contains exactly
one effect. `--release` explicitly exercises the fixture review/controller path.
Without it, the verified artifact remains staged and unpublished.

## Runtime contract

`src/hybrid_control/core.py` owns the runtime ledger. Runs carry immutable
workflow/run identity, policy/config digests, owner, budget/eligibility snapshots,
approval state, evidence, artifacts, gaps, risk and transition history. SQLite
WAL with FULL synchronous commits and `BEGIN IMMEDIATE` serializes transitions,
owner compare-and-swap, and release deduplication. Connections are scoped and
closed. Runtime state does not depend on chat or the Coordinator board.

The exact transition chain is:

`VISIBLE → AUTHENTICATED → AUTHORIZED → ENTITLEMENT_EVALUATED → TARGET_RESOLVED → PREREQUISITES_SATISFIED → SCHEMA_RESOLVED → OPERATION_CAPABLE → EXECUTED → READ_BACK → VERIFIED`.

Terminal states are only `VERIFIED` and `FAILED_TRANSITION_WITH_EVIDENCE`.
Stop pauses persisted runs; it is not a third terminal outcome. Unknown
entitlement never becomes positive. `NOT_APPLICABLE` requires an explicit
controller fact that the entitlement does not apply.

A trusted `live_probe(workflow_id)` supplies fresh facts before each step and
release. Changed evidence re-enters the earliest affected transition, clears
staged output and invalidates approval. Every cross-attempt retry starts at
VISIBLE with a new immutable run ID and parent reference; cached authority is
not carried across attempts. Attempts are capped by versioned policy.

`EXECUTED` produces a staged canonical JSON artifact. `READ_BACK` reads that
artifact from disk. `VERIFIED` checks its digest. A failed read-back or final
verification removes the staged artifact in the same transaction as the
failure event. The release controller rechecks all prerequisite facts,
artifact/config-bound approval and current eligibility. An idempotency key
with different content is rejected.

## Ownership and authority

Workers receive bounded contracts and result/handoff packets. Disabled or
malformed workers are removed before routing. Tool/input validation happens
before execution. Handoffs include input references, completed transitions,
evidence, gaps, contract version and next transition. Concurrent owner changes
use compare-and-swap. Coordinator metadata grants no runtime authority.

Reviewer/controller names are **trusted in-process identities**, not network
authentication credentials. Do not expose these Python methods directly to
untrusted clients. A service integration must bind them to authenticated
principals outside the worker process. Workers must not receive the SQLite
file, controller instance, or writable policy directory.

## Laya and Jev

`SystemOneProvider` sends the actual `POST /v1/systemone` wire request. The
Laya adapter uses `answer_confidence`; Jev uses its native `confidence`.
Provider-native JSON remains in the receipt, with a SHA-256 digest. The
normalized decision does not conflate provider confidence semantics.

```python
import os
from hybrid_control.adapters import SystemOneProvider
from hybrid_control.core import Control

control = Control("ledger.db")
laya = SystemOneProvider("laya", "http://127.0.0.1:8000/v1/systemone")
jev = SystemOneProvider("jev", "https://api.typesafe.ai/v1/systemone",
                        api_key=os.environ["JEV_API_KEY"])
decision = control.decide("workflow", "run", {"text": "bounded evidence",
                          "classification": "public"}, laya, jev)
```

Policy is packaged in `src/hybrid_control/data/policy.json`. Laya's 0.92 and
Jev's 0.97 are independently versioned starting values, **not calibrated
production thresholds**. Low confidence, novelty, high cardinality, high
consequence, unsupported results and policy requirements trigger Jev.
Classification may prohibit cloud escalation. Low-confidence Jev results
become UNKNOWN. A semantic decision never grants execution permission.

The deterministic gate uses explicit `FixtureProvider` objects and a real
local HTTP server to test the adapter. It does not claim that a model checkpoint
or Jev account was deployed or calibrated. Deploy Laya and supply the Jev
credential, then replay labeled workload data before enabling a real external
sink. `external_effects_enabled` defaults to false; no network effect adapter
is included. The local fixture sink is deliberately bounded and reversible.

Wire reference inspected on 2026-09-27:
https://github.com/NandhaKishorM/laya#self-hosting-http-server-jev-compatible

## Twelve-step implementation map

| Step | Implementation / evidence |
|---|---|
| 1 | `fixtures/hybrid_control/golden.json`, canonical JSON staging workflow |
| 2 | SQLite runs/events/staged/effects/receipts/settings migration |
| 3 | `data/workers.json`, input validation and invalid-worker filtering |
| 4 | `eligible`, live probe, independent prerequisite toggle fixtures |
| 5 | `decide`, System One adapter, native receipt persistence and schema |
| 6 | Laya HTTP adapter and versioned boundary tests; live calibration pending |
| 7 | Jev escalation conditions and separate threshold tests |
| 8 | Transactional claim/handoff and concurrent ownership test |
| 9 | Separate `approve`/`release`; result digest and live-fact rechecks |
| 10 | Shadow matrix, provider disagreement/error counts, no staged effects |
| 11 | `hybrid-control.yml`, explicit CO-CI branch helper, rollback tests |
| 12 | Schemas, migration, policy, fixtures, dashboard, recovery, manifest |

## Coordinator integration

Codex Coordinator owns task boundaries only. The runtime ledger remains the
SQLite runs/events/staged/effects/receipts/settings database; Coordinator
claims must never contain runtime receipts, provider responses, or transcripts.

For starting, joining, claiming, completing or integrating work, read the
installed Coordinator skill's `references/operations.md`, then
`references/execution.md`, completely. Load messaging, installation, recovery,
maintenance or doctor references only when that operation applies.

1. Resolve the Git common repository and primary worktree. Read
   `.codex/coordination/project.yaml` before any other board state.
2. Respect `coordination_enabled: false`. An absent marker is not permission
   to initialize a board; use the skill's explicit enablement conditions.
3. Keep a coherent implementation in one task unless authorized parallel work
   has two or three substantial verticals. All writers share the same checkout
   and branch. Never invent a native thread UUID for a claim.
4. When the board is enabled, use its bundled state helper to list active
   claims and atomically claim the exact paths or exclusive action before work.
   A designated Coordinator claims only the bounded `goal-coordination` action
   unless it also edits source files.
5. Update claims at scope, blocked/unblocked and completion boundaries. Do not
   introduce polling, a heartbeat, a second task ledger, or automatic authority.
6. Stage explicit reviewed paths, preserve other work and serialize commands
   that write shared outputs. Path overlap alone is advisory; resolve actual
   conflicting edits before writing.
7. Release the current task's own claim at completion. Never infer that another
   claim is stale from elapsed time or silence.

Coordinator enablement does not change worker contracts, runtime ownership,
approval identities or release permissions. The CO-CI branch helper is an
explicit single-task setup command; do not invoke it while coordinated writers
are active.

## Acceptance evidence

`gate-report.json` records 16 executable gate outcomes and exact passing test
IDs. The suite includes a real process killed after EXECUTED, 24 concurrent
release attempts, concurrent owner claims, every negative transition,
threshold boundaries, disagreement recording and approval revocation.

The acceptance scope is the deterministic local workflow and HTTP protocol.
Production model quality, distributed multi-host storage, identity provider
integration and irreversible external sinks require separate integration gates.
