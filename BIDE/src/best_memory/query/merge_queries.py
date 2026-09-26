#!/usr/bin/env python3
import json, os
from pathlib import Path
H=Path(__file__).resolve().parent;RUN=Path(os.environ.get('BEST276_RUN_ROOT',H/'../../../../runs/default'));SRC=Path(os.environ.get('QUERY_FORMAL',RUN/'query/QUERY_KEYS.jsonl'));OUT=RUN/'query/V41_QUERIES_REBUILT.jsonl';OUT.parent.mkdir(parents=True,exist_ok=True)
def rows(p):return [json.loads(x) for x in p.open() if x.strip()]
primary_path=RUN/'query/V41_QUERY_PLANS_1540.jsonl'; retry_path=RUN/'query/V41_QUERY_RETRY.jsonl'
latest={x['sample_id']:x for x in rows(primary_path)} if primary_path.exists() else {}
latest.update({x['sample_id']:x for x in rows(retry_path) if x['status']=='VALIDATED' and x.get('protocol')=='query-v4.1-complete-20260916'} if retry_path.exists() else {})
output=[]
for base in rows(SRC):
 record=latest.get(base['sample_id'])
 if not record: raise RuntimeError('missing validated complete query plan: '+base['sample_id'])
 plan=record['query_v41']
 parts=[base['question'],*plan['target_entities'],*plan['relation_queries'],*plan['value_queries'],*plan['event_queries'],*plan['temporal_queries'],*plan['lexical_expansions'],*plan['owner_roles'],*plan['required_evidence_acts']]
 # Runtime retrieval records intentionally exclude answers/evidence.
 output.append({'sample_id':base['sample_id'],'conversation_id':base.get('conversation_id') or base['sample_id'].split('::')[0],'question':base['question'],'query_text':' | '.join(str(x) for x in parts if x),'query_v41':plan,'schema_version':'v41-query-rebuilt-v2','gold_visible':False})
assert len(output)==len(rows(SRC)) and len({x['sample_id'] for x in output})==len(output)
with OUT.open('w') as f:
 for x in output:f.write(json.dumps(x,ensure_ascii=False)+'\n')
print(json.dumps({'rows':len(output),'validated':len(latest),'gold_used':False}))
