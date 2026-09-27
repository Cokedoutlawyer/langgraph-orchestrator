"""Release gates exercise disk, processes, threads and controller boundaries."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
import test_control
from hybrid_control.core import Control, ContractError, FixtureProvider


class Gates(unittest.TestCase):
    setUp = test_control.EndToEnd.setUp
    create = test_control.EndToEnd.create
    def test_invalid_workers_removed(self):
        workers = dict(self.c.workers, broken={"enabled": True})
        c = Control(self.path, workers=workers)
        self.assertEqual(c.invalid_workers, ["broken"])
        with self.assertRaises(ContractError):
            c.create("bad", "bad", self.good, owner="broken")

    def test_all_workers_have_contracts(self):
        for owner in self.c.workers:
            with self.subTest(owner=owner):
                for tool in ["release", "approve", "change_policy", "delete"]:
                    with self.assertRaises(ContractError):
                        self.c.validate_worker(owner, {"tool": tool, "input": {"value": "ok"}})
                self.c.validate_worker(owner, {"tool": "stage", "input": {"value": "ok"}})

    def test_laya_threshold_versioned(self):
        for confidence, expected in [(.919999,"jev"),(.92,"laya"),(.920001,"laya")]:
            r = self.c.decide("wf","run",{},FixtureProvider("laya",confidence),FixtureProvider("jev"))
            self.assertEqual(r["provider"], expected)
            self.assertTrue(r["provider_policy_version"])
        for flag in ["novelty","high_cardinality","high_consequence","requires_escalation","unsupported_result","policy_override"]:
            r = self.c.decide("wf","run",{flag:True},FixtureProvider("laya"),FixtureProvider("jev"))
            self.assertEqual(r["provider"], "jev")

    def test_jev_threshold_versioned(self):
        for confidence, expected in [(.969999,"UNKNOWN"),(.97,"POSITIVE"),(.970001,"POSITIVE")]:
            r = self.c.decide("wf","run",{"novelty":True},FixtureProvider("laya"),FixtureProvider("jev",confidence))
            self.assertEqual(r["outcome"], expected)
            self.assertTrue(r["provider_policy_version"])
        with self.assertRaises(ContractError):
            self.c.decide("wf","run",{"novelty":True,"classification":"restricted"},FixtureProvider("laya"),FixtureProvider("jev"))

    def test_evidence_and_gaps_required(self):
        self.create(entitlement="made-up")
        self.c.run("run","codex")
        s = self.c.inspect("run")
        self.assertEqual(s["failure"],"ENTITLEMENT_FAILURE")
        self.assertEqual(s["events"][-1]["observation"]["observed"],"UNKNOWN")
        self.assertTrue(s["gaps"])
        self.assertTrue(all(e["evidence"] for e in s["events"]))
        with self.assertRaises(ContractError):
            self.c.decide("wf","run",{},FixtureProvider("laya",float("nan")),FixtureProvider("jev"))

    def test_owner_uniqueness_concurrent(self):
        self.create()
        def claim(owner):
            try:
                Control(self.path).claim("run",owner,expected_owner="codex")
                return True
            except ContractError:
                return False
        # Both contenders replace the same observed owner. Exactly one CAS wins.
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            results = list(pool.map(claim,["jev","jev"]))
        self.assertEqual(sum(results),1)

    def test_duplicate_effect_concurrent(self):
        self.create()
        self.c.run("run","codex")
        self.c.approve("run","reviewer","APPROVED")
        def release(_):
            return Control(self.path).release("run","one",controller="release-controller")
        with concurrent.futures.ThreadPoolExecutor(8) as pool:
            receipts = list(pool.map(release,range(24)))
        self.assertEqual(self.c.effect_count("wf"),1)
        self.assertTrue(all(r == receipts[0] for r in receipts))

    def test_verification_failure_rolls_back(self):
        self.create(verify=False)
        self.c.run("run","codex")
        s = self.c.inspect("run")
        self.assertEqual(s["failure"],"VERIFICATION_FAILURE")
        self.assertEqual(s["trace"][-1],"READ_BACK")
        self.assertEqual(self.c.staged_count("run"),0)
        with self.assertRaises(ContractError):
            self.c.approve("run","reviewer","APPROVED")

    def test_recovery_from_killed_process(self):
        self.create()
        code = '''import sys,time
from hybrid_control.core import Control
c=Control(sys.argv[1])
for _ in range(9): c.step("run","codex")
print("checkpoint",flush=True)
time.sleep(60)
'''
        proc = subprocess.Popen([sys.executable,"-c",code,str(self.path)],stdout=subprocess.PIPE,text=True,env=dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2]/"src")))
        self.addCleanup(lambda: proc.poll() is None and proc.kill())
        self.assertEqual(proc.stdout.readline().strip(),"checkpoint")
        proc.kill()
        proc.wait(timeout=5)
        proc.stdout.close()
        c = Control(self.path)
        self.assertEqual(c.staged_count("run"),1)
        c.run("run","codex")
        s = c.inspect("run")
        self.assertEqual(s["status"],"VERIFIED")
        self.assertEqual(len(s["events"]),11)
        self.assertEqual(c.staged_count("run"),1)

    def test_live_revocation_reenters_first_invalid_transition(self):
        live = dict(self.good)
        c = Control(self.path,live_probe=lambda _: dict(live))
        self.create()
        for _ in range(8): c.step("run","codex")
        live["authorized"] = False
        c.run("run","codex")
        self.assertEqual(c.inspect("run")["failure"],"AUTHORIZATION_FAILURE")
        self.assertEqual(c.inspect("run")["trace"],["VISIBLE","AUTHENTICATED"])
        self.assertEqual(c.staged_count("run"),0)

    def test_release_rechecks_entitlement_and_evidence(self):
        live = dict(self.good)
        c = Control(self.path,live_probe=lambda _: dict(live))
        self.create()
        c.run("run","codex")
        c.approve("run","reviewer","APPROVED")
        live["entitlement"] = "NEGATIVE"
        with self.assertRaises(ContractError):
            c.release("run","one",controller="release-controller")
        self.assertEqual(c.effect_count("wf"),0)

    def test_shadow_disagreement_recorded(self):
        self.create()
        r = self.c.shadow("run","codex",FixtureProvider("laya"),FixtureProvider("jev",outcome="NEGATIVE"),expected="POSITIVE")
        self.assertEqual(r["matrix"],{"POSITIVE/NEGATIVE":1})
        self.assertEqual(r["disagreements"],1)
        self.assertEqual(r["errors"],1)
        self.assertEqual(self.c.inspect("run")["trace"],[])
        self.assertEqual(self.c.staged_count("run"),0)

if __name__ == "__main__": unittest.main()
