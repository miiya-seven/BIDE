#!/usr/bin/env python3
"""Index an interrupted LongMemEval turn run without calling the API."""
from __future__ import annotations
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run-root',required=True); args=ap.parse_args()
    root=Path(args.run_root); req=root/'memory/MEMORY_REQUESTS.jsonl'; out=root/'migration'; out.mkdir(exist_ok=True)
    wires=root/'memory/MEMORY_OUTPUT/wire'; rejected=root/'memory/MEMORY_OUTPUT/rejected'
    rows=[]; status=Counter(); sessions=Counter(); question_records=defaultdict(int)
    with req.open(encoding='utf8') as fh:
      for idx,line in enumerate(fh):
        if not line.strip(): continue
        item=json.loads(line); center=item.get('center',{}); raw_id=center.get('raw_id',f'index:{idx}')
        wire=wires/f'{idx:04d}.json'; rej=rejected/f'{idx:04d}.json'
        st='SUCCESS' if wire.exists() else ('REJECTED' if rej.exists() else 'MISSING')
        row={'request_index':idx,'raw_id':raw_id,'source_session_id':center.get('source_session_id'),'speaker':center.get('speaker'),'timestamp':center.get('timestamp'),'text':center.get('text'),'wire_path':str(wire) if wire.exists() else None,'rejected_path':str(rej) if rej.exists() else None,'status':st}
        rows.append(row); status[st]+=1; sessions[str(row['source_session_id'])]+=1
    manifest=out/'RAW_CACHE_MANIFEST.jsonl'; manifest.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf8')
    summary={'status':'COMPLETE','requests':len(rows),'status_counts':dict(status),'unique_raw_ids':len({r['raw_id'] for r in rows}),'unique_sessions':len(sessions),'successful_sessions':len({r['source_session_id'] for r in rows if r['status']=='SUCCESS'}),'output':str(manifest),'source_run':str(root)}
    (out/'CACHE_MIGRATION_SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__': main()
