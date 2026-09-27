"""Transactional, bounded workflow runtime. Authority comes from controller probes.

The included operation stages canonical JSON and releases it to a local durable
sink. Network or other irreversible effects require a separately verified adapter.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

STAGES = ("VISIBLE", "AUTHENTICATED", "AUTHORIZED", "ENTITLEMENT_EVALUATED",
          "TARGET_RESOLVED", "PREREQUISITES_SATISFIED", "SCHEMA_RESOLVED",
          "OPERATION_CAPABLE", "EXECUTED", "READ_BACK", "VERIFIED")
FAILURES = dict(zip(STAGES, ("VISIBILITY_FAILURE", "AUTHENTICATION_FAILURE",
    "AUTHORIZATION_FAILURE", "ENTITLEMENT_FAILURE", "TARGET_RESOLUTION_FAILURE",
    "PREREQUISITE_FAILURE", "SCHEMA_FAILURE", "CAPABILITY_FAILURE",
    "EXECUTION_FAILURE", "READ_BACK_FAILURE", "VERIFICATION_FAILURE")))
TERMINAL = ("VERIFIED", "FAILED_TRANSITION_WITH_EVIDENCE")
OUTCOMES = ("POSITIVE", "NEGATIVE", "UNKNOWN", "NOT_APPLICABLE")
ROOT = Path(__file__).resolve().parents[2]


class ContractError(ValueError):
    """A controller contract was not satisfied; no action was performed."""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return "sha256:" + hashlib.sha256(canonical(value).encode()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


class FixtureProvider:
    """Explicit deterministic test adapter; never represents live inference."""
    def __init__(self, name, confidence=.99, outcome="POSITIVE"):
        self.name, self.confidence, self.outcome = name, confidence, outcome
        self.calls = []

    def evaluate(self, request):
        self.calls.append(request)
        return {"provider": self.name, "outcome": self.outcome,
                "confidence": self.confidence, "fixture": True,
                "request_digest": digest(request)}


class Control:
    def __init__(self, path, *, policy=None, workers=None, live_probe=None):
        self.path = str(path)
        self.policy = json.loads(canonical(policy)) if policy is not None else json.loads(
            (Path(__file__).parent / "data/policy.json").read_text())
        self.workers = json.loads(canonical(workers)) if workers is not None else json.loads(
            (Path(__file__).parent / "data/workers.json").read_text())
        self.live_probe = live_probe
        self._validate_policy()
        self.invalid_workers = [name for name, c in self.workers.items() if not self._valid_contract(c)]
        self.workers = {k: v for k, v in self.workers.items() if self._valid_contract(v)}
        with self._db() as db:
            db.executescript((Path(__file__).parent / "data/001_init.sql").read_text())

    def _validate_policy(self):
        try:
            assert self.policy["version"] and self.policy["contract_version"]
            for provider in ("laya", "jev"):
                c = self.policy["routing"][provider]
                assert c["policy_version"] and 0 <= c["confidence_min"] <= 1
            assert self.policy["max_attempts"] > 0
        except (KeyError, TypeError, AssertionError) as exc:
            raise ContractError("invalid versioned policy") from exc

    @staticmethod
    def _valid_contract(c):
        try:
            return bool(c["enabled"] is True and c["version"] and c["allowed_tools"]
                and c["input_schema"]["required"] and c["output_schema"]["required"]
                and c["timeout_seconds"] > 0 and c["max_attempts"] > 0
                and c["failure_classes"] and {"release", "approve", "change_policy"}.issubset(c["forbidden_actions"]))
        except (KeyError, TypeError):
            return False

    @contextlib.contextmanager
    def _db(self):
        db = sqlite3.connect(self.path, timeout=15, isolation_level=None)
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
        try:
            yield db
        finally:
            db.close()

    @contextlib.contextmanager
    def _tx(self):
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                yield db
                db.commit()
            except BaseException:
                db.rollback()
                raise

    @staticmethod
    def _get(db, run_id):
        row = db.execute("SELECT data FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise ContractError("unknown run")
        return json.loads(row[0])

    @staticmethod
    def _save(db, state):
        db.execute("UPDATE runs SET data=? WHERE run_id=?", (canonical(state), state["run_id"]))

    @staticmethod
    def _active(db):
        if db.execute("SELECT value FROM settings WHERE key='stopped'").fetchone()[0] == "true":
            raise ContractError("controller is stopped")

    def eligible(self, facts):
        if (facts.get("permission") is True and facts.get("available") is True
            and facts.get("classification") in self.policy["allowed_classifications"]
            and isinstance(facts.get("budget"), (int, float)) and facts["budget"] > 0
            and "stage" in facts.get("tools", [])):
            return ["stage"]
        return []

    def validate_worker(self, worker, request):
        c = self.workers.get(worker)
        if not c or request.get("tool") not in c["allowed_tools"]:
            raise ContractError("worker or tool is ineligible")
        data = request.get("input")
        if not isinstance(data, dict) or set(data) != {"value"} or not isinstance(data["value"], str):
            raise ContractError("invalid worker input")
        return c

    def create(self, workflow_id, run_id, inputs, *, owner, policy_version=None,
               parent_run_id=None, attempt=1, shadow=False):
        if not workflow_id or not run_id or owner not in self.workers:
            raise ContractError("identity or worker invalid")
        if policy_version is not None and policy_version != self.policy["version"]:
            raise ContractError("policy version mismatch")
        if not isinstance(inputs, dict) or attempt > self.policy["max_attempts"]:
            raise ContractError("input or retry budget invalid")
        state = dict(workflow_id=workflow_id, run_id=run_id, owner=owner,
            parent_run_id=parent_run_id, attempt=attempt, inputs=inputs,
            goal="Stage and verify a canonical JSON artifact", stage=None, status="RUNNING",
            policy_version=self.policy["version"], config_digest=digest(self.policy),
            contract_version=self.policy["contract_version"], budget_snapshot=inputs.get("budget"),
            eligibility_snapshot=self.eligible(inputs), approval="PENDING", approval_receipt=None,
            evidence=[], artifacts=[], gaps=[], risk="bounded_local_fixture", failure=None,
            trace=[], shadow=shadow, created_at=now())
        with self._tx() as db:
            self._active(db)
            try:
                db.execute("INSERT INTO runs VALUES (?,?)", (run_id, canonical(state)))
            except sqlite3.IntegrityError as exc:
                raise ContractError("run_id is immutable and already exists") from exc
        return state

    def inspect(self, run_id):
        with self._db() as db:
            state = self._get(db, run_id)
            state["events"] = [json.loads(r[0]) for r in db.execute(
                "SELECT data FROM events WHERE run_id=? ORDER BY seq", (run_id,))]
            return state

    def _facts(self, state):
        # A fixture snapshot is valid only for the local fixture operation. For
        # integrations the trusted controller injects a fresh probe each step.
        facts = self.live_probe(state["workflow_id"]) if self.live_probe else state["inputs"]
        if not isinstance(facts, dict):
            raise ContractError("invalid live probe result")
        return facts

    def _record(self, db, state, stage, success, observed):
        evidence = {"stage": stage, "observed": observed, "origin": "RUNTIME_OBSERVED",
                    "policy": state["config_digest"]}
        ev_id = digest(evidence)
        event = dict(stage=stage, success=success, evidence=[ev_id], gaps=[] if success else [stage],
                     observation=evidence, created_at=now())
        seq = db.execute("SELECT COUNT(*) FROM events WHERE run_id=?", (state["run_id"],)).fetchone()[0]
        db.execute("INSERT INTO events VALUES (?,?,?)", (state["run_id"], seq, canonical(event)))
        state["evidence"].append(ev_id)
        if success:
            state["stage"] = stage
            state["trace"].append(stage)
            if stage == "VERIFIED":
                state["status"] = "VERIFIED"
        else:
            state.update(status="FAILED_TRANSITION_WITH_EVIDENCE", failure=FAILURES[stage], gaps=[stage])
            db.execute("DELETE FROM staged WHERE run_id=?", (state["run_id"],))
            state["artifacts"] = []
        self._save(db, state)

    def step(self, run_id, owner):
        with self._tx() as db:
            self._active(db)
            s = self._get(db, run_id)
            if s["owner"] != owner or owner not in self.workers:
                raise ContractError("owner mismatch or invalid worker")
            if s["status"] in TERMINAL:
                return s["status"]
            f = self._facts(s)
            # Revalidate every previously established prerequisite before advancing.
            checks = [("visible",), ("authenticated",), ("authorized",),
                      ("entitlement", "entitlement_not_applicable"), ("target_exists",),
                      ("prerequisites",), ("schema_valid",),
                      ("capability_available", "permission", "tools", "classification", "budget", "available")]
            previous = s.get("evidence_facts", s["inputs"])
            for i, keys in enumerate(checks[:min(len(s["trace"]), 8)]):
                if any(previous.get(k) != f.get(k) for k in keys):
                    s["trace"] = s["trace"][:i]
                    s["stage"] = s["trace"][-1] if s["trace"] else None
                    s["approval"], s["approval_receipt"] = "PENDING", None
                    s["artifacts"] = []
                    db.execute("DELETE FROM staged WHERE run_id=?", (run_id,))
                    s.setdefault("invalidations", []).append({"reenter": STAGES[i], "at": now()})
                    break
            s["evidence_facts"] = json.loads(canonical(f))
            index = len(s["trace"])
            stage = STAGES[index]
            if s["config_digest"] != digest(self.policy):
                raise ContractError("policy changed; retry from fresh evidence")
            fields = dict(zip(STAGES[:8], ("visible", "authenticated", "authorized", "entitlement",
                          "target_exists", "prerequisites", "schema_valid", "capability_available")))
            if stage in fields:
                value = f.get(fields[stage])
                ok = value is True
                if stage == "ENTITLEMENT_EVALUATED":
                    value = value if value in OUTCOMES else "UNKNOWN"
                    ok = value == "POSITIVE" or (value == "NOT_APPLICABLE" and f.get("entitlement_not_applicable") is True)
                if stage == "SCHEMA_RESOLVED" and ok:
                    try:
                        self.validate_worker(owner, {"tool": "stage", "input": {"value": f.get("value", "golden")}})
                    except ContractError:
                        ok = False
                if stage == "OPERATION_CAPABLE":
                    ok = ok and bool(self.eligible(f))
                self._record(db, s, stage, ok, value)
            elif stage == "EXECUTED":
                self.validate_worker(owner, {"tool": "stage", "input": {"value": f.get("value", "golden")}})
                # Fresh authorization and eligibility before the first side effect.
                ok = (f.get("safe_operation") is True and f.get("authorized") is True
                      and f.get("authenticated") is True and bool(self.eligible(f)))
                if ok:
                    artifact = {"value": f.get("value", "golden"), "workflow_id": s["workflow_id"]}
                    db.execute("INSERT INTO staged VALUES (?,?)", (run_id, canonical(artifact)))
                    s["artifacts"] = [digest(artifact)]
                self._record(db, s, stage, ok, {"staged": ok})
            else:
                row = db.execute("SELECT artifact FROM staged WHERE run_id=?", (run_id,)).fetchone()
                matches = bool(row and digest(json.loads(row[0])) in s["artifacts"])
                ok = matches and f.get("read_back") is True if stage == "READ_BACK" else matches and f.get("verify", True) is True
                self._record(db, s, stage, ok, {"artifact_matches": matches})
            return s["status"]

    def run(self, run_id, owner):
        while self.step(run_id, owner) not in TERMINAL:
            pass
        return self.inspect(run_id)

    def claim(self, run_id, owner, *, expected_owner):
        if owner not in self.workers:
            raise ContractError("invalid worker")
        with self._tx() as db:
            self._active(db)
            s = self._get(db, run_id)
            if s["owner"] != expected_owner:
                raise ContractError("owner compare-and-swap rejected")
            s["owner"] = owner
            self._save(db, s)

    def handoff(self, run_id, owner, next_owner):
        with self._tx() as db:
            self._active(db)
            s = self._get(db, run_id)
            if s["owner"] != owner or next_owner not in self.workers or s["status"] in TERMINAL:
                raise ContractError("invalid handoff")
            packet = dict(run_id=run_id, input_refs=[digest(s["inputs"])],
                completed_transitions=s["trace"], evidence=s["evidence"], gaps=s["gaps"],
                contract_version=s["contract_version"], next_transition=STAGES[len(s["trace"])])
            s["owner"], s["handoff"] = next_owner, packet
            self._save(db, s)
            return packet

    def approve(self, run_id, reviewer, outcome):
        if reviewer not in self.policy["reviewers"] or reviewer in self.workers or outcome not in ("APPROVED", "REJECTED"):
            raise ContractError("invalid reviewer or decision")
        with self._tx() as db:
            self._active(db)
            s = self._get(db, run_id)
            if s["status"] != "VERIFIED" or s["shadow"]:
                raise ContractError("only verified non-shadow results can be approved")
            s["approval"] = outcome
            s["approval_receipt"] = {"reviewer": reviewer, "artifact_digest": s["artifacts"][0],
                                     "config_digest": s["config_digest"], "created_at": now()}
            self._save(db, s)

    def release(self, run_id, effect_key, *, controller):
        if controller not in self.policy["controllers"] or controller in self.workers or not effect_key:
            raise ContractError("invalid release controller or idempotency key")
        with self._tx() as db:
            self._active(db)
            s = self._get(db, run_id)
            f = self._facts(s)
            row = db.execute("SELECT artifact FROM staged WHERE run_id=?", (run_id,)).fetchone()
            a = s["approval_receipt"]
            if not (s["status"] == "VERIFIED" and not s["shadow"] and s["approval"] == "APPROVED"
                and row and a and digest(json.loads(row[0])) == a["artifact_digest"]
                and a["config_digest"] == digest(self.policy) and self.eligible(f)
                and f.get("authorized") is True and f.get("authenticated") is True
                and f.get("visible") is True and f.get("target_exists") is True
                and f.get("prerequisites") is True and f.get("schema_valid") is True
                and f.get("capability_available") is True
                and (f.get("entitlement") == "POSITIVE" or
                     (f.get("entitlement") == "NOT_APPLICABLE" and f.get("entitlement_not_applicable") is True))
                and s["evidence_facts"] == f
                and self.policy["fixture_effects_enabled"]):
                raise ContractError("release gate rejected")
            old = db.execute("SELECT artifact,receipt FROM effects WHERE workflow_id=? AND effect_key=?",
                             (s["workflow_id"], effect_key)).fetchone()
            if old:
                if old[0] != row[0]:
                    raise ContractError("idempotency key reused for different content")
                return json.loads(old[1])
            receipt = dict(workflow_id=s["workflow_id"], effect_key=effect_key,
                           artifact_digest=digest(json.loads(row[0])), sink="local-sqlite-fixture")
            db.execute("INSERT INTO effects VALUES (?,?,?,?)", (s["workflow_id"], effect_key, row[0], canonical(receipt)))
            return receipt

    def effect_count(self, workflow_id):
        with self._db() as db:
            return db.execute("SELECT COUNT(*) FROM effects WHERE workflow_id=?", (workflow_id,)).fetchone()[0]

    def staged_count(self, run_id):
        with self._db() as db:
            return db.execute("SELECT COUNT(*) FROM staged WHERE run_id=?", (run_id,)).fetchone()[0]

    def retry(self, parent, run_id, inputs, *, owner):
        s = self.inspect(parent)
        if s["status"] not in TERMINAL:
            raise ContractError("parent is not terminal")
        # Re-enter VISIBLE: cross-attempt authority is deliberately not cached.
        return self.create(s["workflow_id"], run_id, inputs, owner=owner,
                           parent_run_id=parent, attempt=s["attempt"] + 1)

    def stop(self):
        with self._tx() as db:
            db.execute("UPDATE settings SET value='true' WHERE key='stopped'")

    def start(self):
        with self._tx() as db:
            db.execute("UPDATE settings SET value='false' WHERE key='stopped'")

    def _decision(self, workflow_id, run_id, request, provider):
        raw = provider.evaluate(request)
        confidence = raw.get("confidence")
        outcome = raw.get("outcome", "UNKNOWN")
        if (outcome not in OUTCOMES or type(confidence) not in (float, int)
                or not math.isfinite(confidence) or not 0 <= confidence <= 1):
            raise ContractError("provider response violates decision schema")
        raw_id = digest(raw)
        with self._tx() as db:
            self._active(db)
            db.execute("INSERT OR IGNORE INTO receipts VALUES (?,?)", (raw_id, canonical(raw)))
        return dict(decision_id=digest([workflow_id, run_id, request, raw]), workflow_id=workflow_id,
            run_id=run_id, transition="ENTITLEMENT_EVALUATED", outcome=outcome, confidence=confidence,
            provider=provider.name, provider_policy_version=self.policy["routing"][provider.name]["policy_version"],
            contract_version=self.policy["contract_version"], inputs_digest=digest(request),
            evidence=[raw_id], receipt={"provider_receipt_id": raw_id, "raw_digest": raw_id}, gaps=[], created_at=now())

    def decide(self, workflow_id, run_id, request, laya, jev):
        if laya.name != "laya" or jev.name != "jev":
            raise ContractError("provider identity mismatch")
        decision = self._decision(workflow_id, run_id, request, laya)
        reasons = [k for k in self.policy["routing"]["jev"]["required_for"] if request.get(k)]
        if decision["confidence"] < self.policy["routing"]["laya"]["confidence_min"]:
            reasons.append("low_confidence")
        if decision["outcome"] == "UNKNOWN":
            reasons.append("unsupported_result")
        if reasons:
            if request.get("classification", "public") not in self.policy["cloud_classifications"]:
                raise ContractError("cloud escalation disallowed by classification")
            decision = self._decision(workflow_id, run_id, request, jev)
            if decision["confidence"] < self.policy["routing"]["jev"]["confidence_min"]:
                decision["outcome"] = "UNKNOWN"
                decision["gaps"] = ["jev_below_threshold"]
        decision["escalation_reasons"] = reasons
        return decision

    def shadow(self, run_id, owner, laya, jev, *, expected):
        with self._tx() as db:
            self._active(db)
            s = self._get(db, run_id)
            if s["trace"] or s["owner"] != owner:
                raise ContractError("shadow requires a fresh owned run")
            s["shadow"] = True
            self._save(db, s)
        request = {"classification": s["inputs"].get("classification", "public")}
        if request["classification"] not in self.policy["cloud_classifications"]:
            raise ContractError("shadow cross-check disallowed")
        a = self._decision(s["workflow_id"], run_id, request, laya)
        b = self._decision(s["workflow_id"], run_id, request, jev)
        report = {"laya": a, "jev": b, "expected": expected,
                  "disagreements": int(a["outcome"] != b["outcome"]),
                  "errors": int(a["outcome"] != expected) + int(b["outcome"] != expected),
                  "matrix": {a["outcome"] + "/" + b["outcome"]: 1}}
        with self._tx() as db:
            s = self._get(db, run_id)
            s["shadow_report"] = report
            self._save(db, s)
        return report
