#!/usr/bin/env python3
"""Build question-level views over the canonical LongMemEval turn requests."""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset',required=True); ap.add_argument('--requests',required=True); ap.add_argument('--wire-dir',required=True); ap.add_argument('--rejected-dir',required=True); ap.add_argument('--out-root',required=True); args=ap.parse_args()
    data=json.loads(Path(args.dataset).read_text(encoding='utf8')); cache=[]
    wire_dir=Path(args.wire_dir); rejected_dir=Path(args.rejected_dir)
    for idx,line in enumerate(Path(args.requests).open(encoding='utf8')):
        if not line.strip(): continue
        item=json.loads(line); center=item.get('center',{}); raw_id=center.get('raw_id',f'index:{idx}')
        wire=wire_dir/f'{idx:04d}.json'; rej=rejected_dir/f'{idx:04d}.json'
        cache.append({'request_index':idx,'raw_id':raw_id,'source_session_id':center.get('source_session_id'),'speaker':center.get('speaker'),'timestamp':center.get('timestamp'),'text':center.get('text'),'status':'SUCCESS' if wire.exists() else ('REJECTED' if rej.exists() else 'MISSING')})
    by_session=defaultdict(list)
    for row in cache: by_session[str(row.get('source_session_id'))].append(row)
    out=Path(args.out_root); (out/'questions').mkdir(parents=True,exist_ok=True)
    manifest_rows=[]
    for i,item in enumerate(data):
        qid=str(item.get('question_id') or f'record_{i:04d}')
        sessions=[str(x) for x in item.get('haystack_session_ids',[])]
        raws=[r for sid in sessions for r in by_session.get(sid,[])]
        row={'question_index':i,'question_id':qid,'question':item.get('question'),'question_type':item.get('question_type'),'question_date':item.get('question_date'),'answer_session_ids':item.get('answer_session_ids',[]),'haystack_session_ids':sessions,'raw_count':len(raws),'cached_raw_count':sum(r.get('status')=='SUCCESS' for r in raws),'missing_raw_count':sum(r.get('status')=='MISSING' for r in raws),'rejected_raw_count':sum(r.get('status')=='REJECTED' for r in raws),'raw_ids':[r['raw_id'] for r in raws]}
        manifest_rows.append(row)
        (out/'questions'/f'{i:04d}.json').write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    (out/'QUESTION_MANIFEST.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in manifest_rows),encoding='utf8')
    summary={'status':'PREPARED','questions':len(manifest_rows),'raw_cache_entries':len(cache),'questions_with_cached_raw':sum(x['cached_raw_count']>0 for x in manifest_rows),'total_question_raw_slots':sum(x['raw_count'] for x in manifest_rows),'total_cached_slots':sum(x['cached_raw_count'] for x in manifest_rows),'total_missing_slots':sum(x['missing_raw_count'] for x in manifest_rows),'output':str(out)}
    (out/'QUESTION_MANIFEST_SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8'); print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__': main()
