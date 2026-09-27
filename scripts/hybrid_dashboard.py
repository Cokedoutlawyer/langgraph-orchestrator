#!/usr/bin/env python3
"""Render a read-only, escaped dashboard from the authoritative ledger."""
import argparse
import html
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from hybrid_control.core import Control
p=argparse.ArgumentParser();p.add_argument('--db',required=True);p.add_argument('--run-id',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
s=Control(a.db).inspect(a.run_id)
rows=''.join('<tr><td>'+html.escape(e['stage'])+'</td><td>'+('PASS' if e['success'] else 'FAIL')+'</td><td>'+html.escape(', '.join(e['evidence']))+'</td></tr>' for e in s['events'])
a.output.write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Hybrid control evidence</title><style>body{font:16px system-ui;margin:3rem;max-width:1100px}table{border-collapse:collapse}td,th{padding:12px;border:1px solid #ddd;text-align:left}td:last-child{font:12px monospace;overflow-wrap:anywhere}pre{white-space:pre-wrap}</style><h1>Hybrid control evidence</h1><p>'+html.escape(s['run_id']+' — '+s['status'])+'</p><table><tr><th>Transition</th><th>Result</th><th>Evidence</th></tr>'+rows+'</table><h2>Result packet</h2><pre>'+html.escape(json.dumps({k:v for k,v in s.items() if k!='events'},indent=2))+'</pre></html>')
