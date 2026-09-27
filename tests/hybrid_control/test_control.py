import json
import tempfile
import unittest
from pathlib import Path

from hybrid_control.core import Control, FixtureProvider, STAGES, FAILURES, ContractError


class EndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "ledger.sqlite"
        self.c = Control(self.path)
        self.good = dict(authenticated=True, authorized=True, entitlement="POSITIVE",
                         target_exists=True, prerequisites=True, schema_valid=True,
                         capability_available=True, read_back=True, visible=True,
                         safe_operation=True, permission=True, tools=["stage"],
                         classification="public", budget=1, available=True,
                         approval_required=True)

    def create(self, **changes):
        inputs = dict(self.good, **changes)
        self.c.create("wf", "run", inputs, owner="codex", policy_version="2026-09-27.1")
        return inputs

    def test_golden_recovery_approval_and_duplicate_release(self):
        self.create()
        self.c.step("run", "codex")
        other = Control(self.path)
        self.assertEqual(other.inspect("run")["trace"], ["VISIBLE"])
        with self.assertRaises(ContractError):
            other.claim("run", "jev", expected_owner=None)
        other.run("run", "codex")
        state = other.inspect("run")
        self.assertEqual(state["trace"], list(STAGES))
        self.assertEqual(state["status"], "VERIFIED")
        self.assertEqual(other.effect_count("wf"), 0)
        with self.assertRaises(ContractError):
            other.release("run", "effect-1", controller="release-controller")
        other.approve("run", "reviewer", "APPROVED")
        first = other.release("run", "effect-1", controller="release-controller")
        second = other.release("run", "effect-1", controller="release-controller")
        self.assertEqual(first, second)
        self.assertEqual(other.effect_count("wf"), 1)

    def test_each_first_failure_blocks_effect(self):
        faults = {"VISIBLE": ("visible", False), "AUTHENTICATED": ("authenticated", False),
                  "AUTHORIZED": ("authorized", False), "ENTITLEMENT_EVALUATED": ("entitlement", "UNKNOWN"),
                  "TARGET_RESOLVED": ("target_exists", False),
                  "PREREQUISITES_SATISFIED": ("prerequisites", False),
                  "SCHEMA_RESOLVED": ("schema_valid", False),
                  "OPERATION_CAPABLE": ("capability_available", False),
                  "EXECUTED": ("safe_operation", False), "READ_BACK": ("read_back", False)}
        for index, (stage, (key, value)) in enumerate(faults.items()):
            with self.subTest(stage=stage):
                run_id = f"bad-{index}"
                self.c.create(f"wf-{index}", run_id, dict(self.good, **{key: value}), owner="codex")
                self.c.run(run_id, "codex")
                state = self.c.inspect(run_id)
                self.assertEqual(state["status"], "FAILED_TRANSITION_WITH_EVIDENCE")
                self.assertEqual(state["failure"], FAILURES[stage])
                self.assertEqual(state["trace"], list(STAGES[:index]))
                self.assertEqual(self.c.effect_count(f"wf-{index}"), 0)
                self.assertTrue(state["events"][-1]["evidence"])
                self.assertIsInstance(state["events"][-1]["gaps"], list)

    def test_eligibility_and_worker_contract_fail_closed(self):
        self.assertEqual(self.c.eligible(self.good), ["stage"])
        for field, value in [("permission", False), ("tools", []), ("classification", "restricted"),
                             ("budget", 0), ("available", False)]:
            with self.subTest(field=field):
                self.assertEqual(self.c.eligible(dict(self.good, **{field: value})), [])
        self.create(tools=[])
        self.c.run("run", "codex")
        self.assertEqual(self.c.inspect("run")["failure"], "CAPABILITY_FAILURE")
        with self.assertRaises(ContractError):
            self.c.validate_worker("bad", {"tool": "delete", "input": {"value": "a"}})
        with self.assertRaises(ContractError):
            self.c.validate_worker("codex", {"tool": "stage", "input": {"wrong": "a"}})

    def test_provider_receipts_escalation_threshold_and_shadow(self):
        laya = FixtureProvider("laya", confidence=.92, outcome="POSITIVE")
        jev = FixtureProvider("jev", confidence=.98, outcome="POSITIVE")
        result = self.c.decide("wf", "run", {"case": "ordinary"}, laya, jev)
        self.assertEqual(result["provider"], "laya")
        self.assertEqual(result["outcome"], "POSITIVE")
        self.assertEqual(len(jev.calls), 0)
        laya.confidence = .919
        result2 = self.c.decide("wf", "run", {"case": "ordinary"}, laya, jev)
        self.assertEqual(result2["provider"], "jev")
        self.assertNotEqual(result2["receipt"]["raw_digest"], result["receipt"]["raw_digest"])
        self.assertEqual(result2["outcome"], result["outcome"])
        self.create()
        report = self.c.shadow("run", "codex", laya, jev, expected="POSITIVE")
        self.assertEqual(report["disagreements"], 0)
        self.assertEqual(self.c.effect_count("wf"), 0)
        with self.assertRaises(ContractError):
            self.c.release("run", "effect-1", controller="release-controller")

    def test_rollback_shutdown_resume_and_handoff(self):
        self.create(read_back=False)
        self.c.step("run", "codex")
        packet = self.c.handoff("run", "codex", "jev")
        self.assertEqual(packet["next_transition"], "AUTHENTICATED")
        with self.assertRaises(ContractError):
            self.c.step("run", "codex")
        self.c.stop()
        with self.assertRaises(ContractError):
            self.c.step("run", "jev")
        resumed = Control(self.path)
        resumed.start()
        resumed.run("run", "jev")
        self.assertEqual(resumed.inspect("run")["failure"], "READ_BACK_FAILURE")
        self.assertEqual(resumed.staged_count("run"), 0)
        self.assertEqual(resumed.effect_count("wf"), 0)
        self.assertEqual(resumed.inspect("run")["owner"], "jev")

    def test_retry_new_run_rechecks_authority_and_idempotency(self):
        self.create(authorized=False)
        self.c.run("run", "codex")
        self.c.retry("run", "retry", dict(self.good), owner="codex")
        self.c.run("retry", "codex")
        self.assertEqual(self.c.inspect("retry")["trace"], list(STAGES))
        self.assertEqual(self.c.inspect("retry")["parent_run_id"], "run")
        with self.assertRaises(ContractError):
            self.c.retry("run", "retry", dict(self.good), owner="codex")


if __name__ == "__main__":
    unittest.main()
