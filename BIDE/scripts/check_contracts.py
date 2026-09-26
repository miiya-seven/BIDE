#!/usr/bin/env python3
"""Validate the file contracts between completed stages of one run."""
import argparse, json
from pathlib import Path

CONTRACTS={
 'memory':['MEMORY_OUTPUT/wire'],
 'contextual_memory':['CONTEXTUAL_V2.jsonl'],
 'contextual_merge':['CONTEXTUAL_MEMORY_V41_SIDECAR.jsonl'],
 'query':['QUERY_OUTPUT/QUERY_KEYS.jsonl'],
 'query_merge':['query/V41_QUERIES_REBUILT.jsonl'],
 'corpus':['V41_RAW_VIEWS.jsonl','V41_QUERIES_REBUILT.jsonl'],
 'fusion':['retrieval/CANDIDATE128.jsonl'],
 'rerank':['RANKINGS.json'],
 'answer':['full/FROZEN_FINAL.json'],
}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('run_dir'); ap.add_argument('--stage',required=True); a=ap.parse_args(); root=Path(a.run_dir)
 missing=[]
 for rel in CONTRACTS[a.stage]:
  if not any((root/rel).exists() for _ in [0]): missing.append(rel)
 result={'stage':a.stage,'run_dir':str(root),'required':CONTRACTS[a.stage],'missing':missing,'ok':not missing}
 print(json.dumps(result,ensure_ascii=False,indent=2)); return 0 if not missing else 1
if __name__=='__main__': raise SystemExit(main())
