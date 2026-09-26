#!/usr/bin/env python3
"""Publish the rebuilt Query view separately from memory corpus compilation."""
import json, os
from pathlib import Path
run=Path(os.environ.get('BEST276_RUN_ROOT',Path(__file__).resolve().parents[4]/'runs/default'))
src=Path(os.environ.get('QUERY_REBUILT',run/'query/V41_QUERIES_REBUILT.jsonl'))
out=Path(os.environ.get('CORPUS_OUTPUT_DIR',run/'retrieval'))/'V41_QUERIES_REBUILT.jsonl'
out.parent.mkdir(parents=True,exist_ok=True)
if not src.exists(): raise SystemExit(f'missing rebuilt query: {src}')
rows=[json.loads(x) for x in src.read_text(encoding='utf8').splitlines() if x.strip()]
out.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),encoding='utf8')
print(json.dumps({'rows':len(rows),'output':str(out),'gold_used':False}))
