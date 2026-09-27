#!/usr/bin/env python3
"""Reproducible golden execution and separate fixture release."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from hybrid_control.core import Control

p=argparse.ArgumentParser(); p.add_argument('--db',default='hybrid-demo.db'); p.add_argument('--run-id',default='golden-v1'); p.add_argument('--release',action='store_true'); args=p.parse_args()
c=Control(args.db)
f=json.loads((Path(__file__).resolve().parents[1]/'fixtures/hybrid_control/golden.json').read_text())
try: c.inspect(args.run_id)
except ValueError: c.create('golden-workflow',args.run_id,f,owner='codex')
s=c.run(args.run_id,'codex')
if args.release:
    c.approve(args.run_id,'reviewer','APPROVED')
    receipt=c.release(args.run_id,'golden-operation-v1',controller='release-controller')
    print(json.dumps(receipt,indent=2))
print(json.dumps({'status':s['status'],'trace':s['trace'],'effects':c.effect_count('golden-workflow')},indent=2))
raise SystemExit(0 if s['status']=='VERIFIED' else 1)
