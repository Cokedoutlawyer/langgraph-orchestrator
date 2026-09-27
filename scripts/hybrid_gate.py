#!/usr/bin/env python3
"""Execute tests and emit the requested 16 gates. Any failure exits nonzero."""
import argparse
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests/hybrid_control')]
GATES = {
 'durable_state':['Gates.test_recovery_from_killed_process'],
 'chat_state_is_non_authoritative':['EndToEnd.test_golden_recovery_approval_and_duplicate_release'],
 'laya_threshold_versioned':['Gates.test_laya_threshold_versioned'],
 'jev_threshold_versioned':['Gates.test_jev_threshold_versioned'],
 'all_workers_have_contracts':['Gates.test_all_workers_have_contracts'],
 'invalid_workers_removed':['Gates.test_invalid_workers_removed'],
 'evidence_and_gaps_required':['Gates.test_evidence_and_gaps_required','EndToEnd.test_each_first_failure_blocks_effect'],
 'external_effect_requires_approval':['EndToEnd.test_golden_recovery_approval_and_duplicate_release','Gates.test_release_rechecks_entitlement_and_evidence'],
 'retries_idempotent':['EndToEnd.test_retry_new_run_rechecks_authority_and_idempotency'],
 'duplicate_effect_test':['Gates.test_duplicate_effect_concurrent'],
 'owner_uniqueness':['Gates.test_owner_uniqueness_concurrent'],
 'completion_separate_from_release':['EndToEnd.test_golden_recovery_approval_and_duplicate_release'],
 'shadow_mode_passed':['Gates.test_shadow_disagreement_recorded'],
 'rollback_verified':['Gates.test_verification_failure_rolls_back'],
 'graceful_stop_verified':['EndToEnd.test_rollback_shutdown_resume_and_handoff'],
 'recovery_from_interruption_verified':['Gates.test_recovery_from_killed_process','Gates.test_live_revocation_reenters_first_invalid_transition'],
}

class EvidenceResult(unittest.TextTestResult):
    def __init__(self,*a,**k): super().__init__(*a,**k); self.passed=[]
    def addSuccess(self,test): super().addSuccess(test); self.passed.append(test.id())


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path); args=p.parse_args()
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests/hybrid_control'))
    result=unittest.TextTestRunner(verbosity=1,resultclass=EvidenceResult).run(suite)
    gates={name: all(any(t.endswith(required) for t in result.passed) for required in cases) for name,cases in GATES.items()}
    for name,passed in gates.items(): print(('PASS' if passed else 'FAIL')+' '+name)
    report={'scope':'deterministic local fixture and HTTP adapter contract',
            'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
            'gates':gates,'test_evidence':result.passed,
            'live_provider_calibration':'NOT_RUN: requires deployed checkpoint, labeled workload and Jev credential',
            'external_effect_sink':'local-sqlite-fixture'}
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    return 0 if result.wasSuccessful() and all(gates.values()) else 1

if __name__=='__main__': raise SystemExit(main())
