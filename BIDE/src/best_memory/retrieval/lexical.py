#!/usr/bin/env python3
import json,math,re,os
from collections import Counter,defaultdict
from pathlib import Path
H=Path(__file__).resolve().parent;INPUT=Path(os.environ.get('LEXICAL_INPUT_DIR',H));OUTPUT=Path(os.environ.get('LEXICAL_OUTPUT_DIR',H));PREFIX=os.environ.get('LEXICAL_PREFIX','V41');QUERY=Path(os.environ.get('LEXICAL_QUERY',H/'V41_QUERIES_REBUILT.jsonl'))
STOP=set('a an the and or of to in on at for from with by is are was were be been being do did does have has had what when where who why how which that this these those his her their its he she they it about according'.split())
def rows(p):return [json.loads(x) for x in p.open(encoding='utf8') if x.strip()]
def toks(v):return [x for x in re.findall(r"[a-z0-9']+",str(v).lower()) if len(x)>2 and x not in STOP]
def make_index(items,id_key):
 groups=defaultdict(list)
 for x in items:groups[x['conversation_id']].append((x[id_key],x['raw_id'],toks(x.get('retrieval_text',''))))
 out={}
 for cid,docs in groups.items():
  df=Counter(t for _,_,d in docs for t in set(d));avg=sum(len(d) for _,_,d in docs)/max(1,len(docs));out[cid]=(docs,df,avg)
 return out
def bm25(index,cid,query):
 # A valid LongMemEval session may have no extracted assertions (for
 # example, the self-contained/raw smoke backend or a failed L2 response).
 # Raw-view retrieval must still remain usable; an absent assertion index is
 # an empty lane, not a fatal KeyError.
 if cid not in index:
  return [], {}
 docs,df,avg=index[cid];qt=toks(query);raw_score=defaultdict(float)
 for _,rid,doc in docs:
  tf=Counter(doc);s=0.0
  for t in qt:
   if tf[t]:
    idf=math.log(1+(len(docs)-df[t]+.5)/(df[t]+.5));s+=idf*(tf[t]*2.2)/(tf[t]+1.2*(.25+.75*len(doc)/max(1,avg)))
  raw_score[rid]=max(raw_score[rid],s)
 return sorted(raw_score,key=lambda r:(-raw_score[r],r)),raw_score
raws=rows(INPUT/f'{PREFIX}_RAW_VIEWS.jsonl');assertions=rows(INPUT/f'{PREFIX}_ASSERTIONS.jsonl');props=rows(INPUT/f'{PREFIX}_PROPOSITIONS.jsonl');queries=rows(QUERY);postings=rows(INPUT/f'{PREFIX}_ENTITY_POSTINGS.jsonl');edges=rows(INPUT/f'{PREFIX}_GRAPH_EDGES.jsonl')
if os.environ.get('BEST276_DISABLE_L3')=='1':
 postings=[];edges=[]
raw_idx=make_index(raws,'raw_id');assert_idx=make_index(assertions,'assertion_id');prop_idx=make_index(props,'proposition_id')
entity=defaultdict(lambda:defaultdict(set))
for x in postings:entity[x['raw_id'].split('::')[0]][str(x['entity']).lower()].add(x['raw_id'])
adj=defaultdict(set)
for x in edges:adj[x['source_raw_id']].add(x['target_raw_id']);adj[x['target_raw_id']].add(x['source_raw_id'])
def entity_rank(q,cid,fallback):
 targets=[str(x).lower() for x in q.get('target_entities',[])];score=Counter()
 for name,rids in entity[cid].items():
  for target in targets:
   if target==name or target in name or name in target:
    for rid in rids:score[rid]+=3 if target==name else 1
 return sorted(score,key=lambda r:(-score[r],r))+[r for r in fallback if r not in score]
def graph_rank(seed,base_scores):
 score=defaultdict(float)
 for pos,rid in enumerate(seed[:96],1):
  score[rid]+=1/(20+pos)
  for n in adj.get(rid,()):score[n]+=.72/(20+pos)
 return sorted(score,key=lambda r:(-score[r],-base_scores.get(r,0),r))+[r for r in seed if r not in score]
def quota(lanes,amounts,limit=128):
 out=[];seen=set()
 for name,n in amounts:
  added=0
  for rid in lanes[name]:
   if rid in seen:continue
   out.append(rid);seen.add(rid);added+=1
   if added==n or len(out)==limit:break
 while len(out)<limit:
  changed=False
  for ranking in lanes.values():
   for rid in ranking:
    if rid not in seen:out.append(rid);seen.add(rid);changed=True;break
   if len(out)==limit:break
  if not changed:break
 return out
output=[];lane_rows=[]
for q in queries:
 sid=q['sample_id'];cid=q['conversation_id'];text=q['query_text'];raw_rank,raw_score=bm25(raw_idx,cid,text);a_rank,a_score=bm25(assert_idx,cid,text);p_rank,p_score=bm25(prop_idx,cid,text) if cid in prop_idx else (raw_rank,{})
 limit=min(128,len(raw_idx.get(cid,([],{},0))[0]))
 e_rank=entity_rank(q,cid,a_rank);combined={r:max(raw_score.get(r,0),a_score.get(r,0),p_score.get(r,0)) for r in set(raw_rank)|set(a_rank)|set(p_rank)};seed=sorted(combined,key=lambda r:(-combined[r],r));g_rank=graph_rank(seed,combined)
 lanes={'raw':raw_rank,'assertion':a_rank,'proposition':p_rank,'entity':e_rank,'graph':g_rank}
 arms={
  'LEX_BALANCED':quota(lanes,[('raw',28),('assertion',36),('proposition',16),('entity',16),('graph',32)],limit),
  'ASSERT_GRAPH':quota(lanes,[('assertion',44),('graph',40),('raw',20),('entity',16),('proposition',8)],limit),
  'RAW_GRAPH':quota(lanes,[('raw',40),('graph',40),('assertion',24),('entity',16),('proposition',8)],limit),
  'SET_HEAVY':quota(lanes,[('graph',48),('assertion',32),('entity',20),('raw',20),('proposition',8)],limit)}
 # LongMemEval has roughly 48 candidates per question, while LoCoMo uses the
 # historical 128 contract.  Preserve 128 where available and use the full
 # available domain otherwise; never fabricate duplicate candidate IDs.
 if any(len(v)!=limit or len(set(v))!=limit for v in arms.values()):raise RuntimeError(sid)
 output.append({'sample_id':sid,'rankings':arms,'candidate_size':limit,'gold_visible':False});lane_rows.append({'sample_id':sid,'rankings':{k:v[:limit] for k,v in lanes.items()},'gold_visible':False})
for name,data in [('LEXICAL_CANDIDATE128.jsonl',output),('LEXICAL_LANES.jsonl',lane_rows)]:
 with (OUTPUT/name).open('w') as f:
  for x in data:f.write(json.dumps(x)+'\n')
print(json.dumps({'questions':len(output),'candidate_size':128,'gold_used':False}))
