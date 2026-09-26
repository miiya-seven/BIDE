#!/usr/bin/env python3
import concurrent.futures,hashlib,json,os,time,urllib.request,re,subprocess
from collections import Counter
from pathlib import Path
H=Path(__file__).resolve().parent;RUN=Path(os.environ.get('BEST276_RUN_ROOT',H/'../../../../runs/default'));FORMAL=Path(os.environ.get('QUERY_FORMAL',RUN/'query/QUERY_KEYS.jsonl'));PRIMARY=RUN/'query/V41_QUERY_PLANS_1540.jsonl';OUT=RUN/'query/V41_QUERY_RETRY.jsonl';WIRE=RUN/'query/WIRE';WIRE.mkdir(parents=True,exist_ok=True);PROTOCOL='query-v4.1-complete-20260916'
URL=os.environ.get('LLM_BASE_URL','').rstrip('/')+'/chat/completions';MODEL=os.environ.get('LLM_MODEL','gpt-5.4-mini');KEY=os.environ.get(os.environ.get('LLM_AUTH_ENV','GENERATION_API_KEY')) or os.environ.get('API_KEY') or os.environ.get('OPENAI_API_KEY','')
TIMEOUT=int(os.environ.get('QUERY_API_TIMEOUT','300'));ATTEMPTS=int(os.environ.get('QUERY_API_ATTEMPTS','3'));NETWORK_MODE=os.environ.get('LLM_NETWORK_MODE','direct');PROXY_URL=os.environ.get('LLM_PROXY_URL','')
if PROXY_URL:
 OPENER=urllib.request.build_opener(urllib.request.ProxyHandler({'http':PROXY_URL,'https':PROXY_URL}))
elif NETWORK_MODE=='direct':
 OPENER=urllib.request.build_opener(urllib.request.ProxyHandler({}))
else:
 OPENER=urllib.request.build_opener()
SYSTEM='''Create a retrieval Query Plan for the question only. Never answer the question and never invent conversation facts. Return JSON with exactly:
query_kind: SINGLE_FACT|SET_COLLECTION|TEMPORAL|COMPARISON|CAUSE_RESULT|MULTIHOP
target_entities: list of explicit or resolved person/object names from the question
owner_roles: list from SUBJECT|AGENT|EXPERIENCER|OWNER|OBSERVER|PARTICIPANT|TIME_OWNER
relation_queries: 2-6 short normalized relation paraphrases
value_queries: list of value/category cues requested by the question
event_queries: list of event or activity anchors
temporal_queries: list of explicit/relative time constraints
polarity: POSITIVE|NEGATIVE|UNKNOWN
modality: ACTUAL|PLANNED|DESIRED|HYPOTHETICAL|UNKNOWN
required_evidence_acts: list from ASSERTION|CONFIRMATION|DENIAL|REACTION|QUESTION
graph_operations: list from ANTECEDENT|OCCURRENCE_MEMBERS|PROPOSITION_SOURCE|TEMPORAL_NEIGHBORS|SAME_ENTITY_EVENTS
lexical_expansions: 4-12 short generic synonyms/paraphrases grounded in the question
Rules: Use only the question. Generic and dataset-independent. JSON only.
- relation_queries and lexical_expansions must be non-empty.
- value_queries must describe the requested answer slot (for example time/date, place, person, activity, reason, count, preference, or state).
- TEMPORAL requires temporal_queries and a time/date value cue.
- required_evidence_acts must be non-empty; include CONFIRMATION when a confirming reply could carry the answer.
- graph_operations must be non-empty. Select operations that can recover linked evidence; temporal questions normally need TEMPORAL_NEIGHBORS, and event questions normally need SAME_ENTITY_EVENTS or OCCURRENCE_MEMBERS.
- Empty event_queries or temporal_queries are allowed only when irrelevant to the question. JSON only.'''
def rows(p):return [json.loads(x) for x in p.open() if x.strip()] if p.exists() else []
allowed_kinds={'SINGLE_FACT','SET_COLLECTION','TEMPORAL','COMPARISON','CAUSE_RESULT','MULTIHOP'}
primary=rows(PRIMARY)
bad_ids={x['sample_id'] for x in primary if x.get('status')!='VALIDATED' or (x.get('query_v41') or {}).get('query_kind') not in allowed_kinds}
if not primary: bad_ids={x['sample_id'] for x in rows(FORMAL)}
questions=[{'sample_id':x['sample_id'],'question':x['question']} for x in rows(FORMAL) if x['sample_id'] in bad_ids]
existing={x['sample_id']:x for x in rows(OUT)}
done={sid for sid,x in existing.items() if x.get('status')=='VALIDATED' and x.get('protocol')==PROTOCOL};todo=[x for x in questions if x['sample_id'] not in done]
required={'query_kind','target_entities','owner_roles','relation_queries','value_queries','event_queries','temporal_queries','polarity','modality','required_evidence_acts','graph_operations','lexical_expansions'}


def _deterministic_lme_plan(item):
 """Build a gold-blind query plan from question wording only.

 LongMemEval's question text is already a valid retrieval query.  This
 adapter avoids 500 extra provider calls in the full benchmark while keeping
 the v4.1 closed shape and never reading answers or evidence.
 """
 q=item['question'].strip(); lower=q.lower()
 # Questions can contain a temporal clause while asking for a non-temporal
 # value (e.g. "what amount ... when I got ...").  Classify the requested
 # answer slot before looking at subordinate time clauses; otherwise the
 # temporal lane overweights message/event time and weakens value retrieval.
 if any(x in lower for x in ('how much', 'what amount', 'the amount', 'amount was', 'price', 'cost', 'dollar', '$')):
  kind='SINGLE_FACT'; temporal=[]
 elif any(x in lower for x in ('when ', 'what date', 'which date', 'how long', 'what time')):
  kind='TEMPORAL'; temporal=['when/date/time']
 elif any(x in lower for x in ('how many', 'which of', 'what are ', 'list ')):
  kind='SET_COLLECTION'; temporal=[]
 elif lower.startswith('why ') or ' why ' in lower:
  kind='CAUSE_RESULT'; temporal=[]
 elif any(x in lower for x in ('compare', 'difference', 'both ', 'versus', ' vs ')):
  kind='COMPARISON'; temporal=[]
 elif ' and ' in lower and any(x in lower for x in ('who', 'what', 'how')):
  kind='MULTIHOP'; temporal=[]
 else:
  kind='SINGLE_FACT'; temporal=[]
 words=re.findall(r"[A-Za-z0-9][A-Za-z0-9'_-]*",q)
 lexical=[]
 for word in words:
  if len(word)>2 and word.lower() not in {'what','when','where','which','who','does','did','how','the','and','for','with'}:
   lexical.append(word.lower())
 lexical=list(dict.fromkeys(lexical))[:12] or ['evidence']
 plan={'query_kind':kind,'target_entities':[],
       'owner_roles':['SUBJECT'],'relation_queries':[q[:240]],
       'value_queries':[q[:240]],'event_queries':[],'temporal_queries':temporal,
       'polarity':'UNKNOWN','modality':'ACTUAL','required_evidence_acts':['ASSERTION'],
       'graph_operations':['SAME_ENTITY_EVENTS'],'lexical_expansions':lexical}
 return {**item,'status':'VALIDATED','validation_errors':[],'query_v41':plan,
         'model':'deterministic_lme_adapter','protocol':PROTOCOL,
         'request_sha256':hashlib.sha256(canon_json(plan=item['question']).encode()).hexdigest(),
         'gold_visible':False}


def canon_json(plan):
 return json.dumps(plan,ensure_ascii=False,sort_keys=True,separators=(',',':'))


if os.environ.get('BEST276_DATASET') == 'longmemeval' and os.environ.get('BEST276_LME_DETERMINISTIC_QUERY','0') == '1':
 deterministic=[_deterministic_lme_plan(item) for item in questions]
 latest={x['sample_id']:x for x in existing.values()}
 latest.update({x['sample_id']:x for x in deterministic})
 OUT.write_text(''.join(json.dumps(latest[sid],ensure_ascii=False)+'\n' for sid in sorted(latest)))
 receipt={'questions':len(questions),'rows':len(latest),'status':{'VALIDATED':len(deterministic)},
          'model':'deterministic_lme_adapter','protocol':PROTOCOL,'gold_used':False}
 (RUN/'query/QUERY_V41_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(receipt)); raise SystemExit(0)

def call(item):
 # Keep the request within the OpenAI-compatible subset supported by the configured service.
 # In particular, seed/reasoning_effort are not accepted by every direct model.
 body={'model':MODEL,'messages':[{'role':'system','content':SYSTEM},{'role':'user','content':item['question']}],'temperature':0,'max_tokens':1100,'response_format':{'type':'json_object'}};canon=json.dumps(body,sort_keys=True,separators=(',',':'));sha=hashlib.sha256(canon.encode()).hexdigest()
 for attempt in range(ATTEMPTS):
  try:
   if NETWORK_MODE=='direct':
    command=['curl','-sS','--fail-with-body','--max-time',str(TIMEOUT),'-H','Content-Type: application/json','-H','Authorization: Bearer '+KEY,'--data-binary','@-',URL]
    try:
     completed=subprocess.run(command,input=canon.encode(),stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,timeout=TIMEOUT+10,check=True)
    except Exception as exc:
     raise RuntimeError(f'curl_transport_failed:{type(exc).__name__}') from None
    wire=json.loads(completed.stdout)
   else:
    request=urllib.request.Request(URL,data=canon.encode(),headers={'Authorization':'Bearer '+KEY,'Content-Type':'application/json'})
    wire=json.loads(OPENER.open(request,timeout=TIMEOUT).read())
   (WIRE/f'{sha}.json').write_text(json.dumps(wire)+'\n');plan=json.loads(wire['choices'][0]['message']['content']);errors=[]
   if set(plan)!=required:errors.append('KEY_CONTRACT')
   if plan.get('query_kind') not in allowed_kinds:errors.append('QUERY_KIND_ENUM')
   for key in ('target_entities','owner_roles','relation_queries','value_queries','event_queries','temporal_queries','required_evidence_acts','graph_operations','lexical_expansions'):
    if not isinstance(plan.get(key),list):errors.append('TYPE_'+key.upper())
   for key in ('relation_queries','value_queries','required_evidence_acts','graph_operations','lexical_expansions'):
    if not plan.get(key):errors.append('EMPTY_'+key.upper())
   if plan.get('query_kind')=='TEMPORAL' and not plan.get('temporal_queries'):errors.append('EMPTY_TEMPORAL_QUERIES')
   return {**item,'status':'VALIDATED' if not errors else 'UNCERTAIN','validation_errors':sorted(set(errors)),'query_v41':plan,'model':MODEL,'protocol':PROTOCOL,'request_sha256':sha,'gold_visible':False}
  except Exception as e:
   if attempt==ATTEMPTS-1:
    error='API_TIMEOUT' if isinstance(e,TimeoutError) else f'{type(e).__name__}:{str(e)[:300]}'
    return {**item,'status':'ERROR','validation_errors':[error],'model':MODEL,'protocol':PROTOCOL,'request_sha256':sha,'gold_visible':False}
   time.sleep(2**attempt)
with OUT.open('a') as f:
 with concurrent.futures.ThreadPoolExecutor(max_workers=int(os.environ.get('QUERY_WORKERS','2'))) as pool:
  for i,x in enumerate(pool.map(call,todo),1):f.write(json.dumps(x,ensure_ascii=False)+'\n');f.flush();print(json.dumps({'completed':i,'total':len(todo)}),flush=True)
latest={x['sample_id']:x for x in rows(OUT)}
with OUT.open('w') as f:
 for sid in sorted(latest):f.write(json.dumps(latest[sid],ensure_ascii=False)+'\n')
final=list(latest.values());receipt={'questions':len(questions),'rows':len(final),'status':dict(Counter(x['status'] for x in final)),'model':MODEL,'protocol':PROTOCOL,'gold_used':False};(RUN/'query/QUERY_V41_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
