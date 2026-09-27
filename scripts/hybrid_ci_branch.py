#!/usr/bin/env python3
"""Explicitly create a CO-CI branch; never run as a background coordinator."""
import argparse
import re
import subprocess
p=argparse.ArgumentParser();p.add_argument('name');a=p.parse_args()
if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,60}',a.name): p.error('use a short lowercase branch suffix')
if subprocess.check_output(['git','status','--porcelain'],text=True).strip(): p.error('commit or preserve outstanding work before starting another branch')
subprocess.run(['git','switch','-c','co-ci/'+a.name],check=True)
