#!/usr/bin/env python3
import json,time,urllib.request,os
from collections import defaultdict
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;INPUT=Path(os.environ.get('DENSE_INPUT_DIR',H));OUTPUT=Path(os.environ.get('DENSE_OUTPUT_DIR',H));PREFIX=os.environ.get('DENSE_PREFIX','V41');QUERY=Path(os.environ.get('DENSE_QUERY',H/'V41_QUERIES_REBUILT.jsonl'));URL=os.environ.get('EMBEDDING_URL',os.environ.get('EMBEDDING_BASE_URL','http://127.0.0.1:8005/v1').rstrip('/')+'/embeddings');MODEL=os.environ.get('EMBEDDING_MODEL','BAAI/bge-m3')
def rows(p):return [json.loads(x) for x in p.open(encoding='utf8') if x.strip()]
def embed(texts):
 if not texts:return np.empty((0,0),dtype=np.float32)
 out=[]
 batches=[];current=[];chars=0
 for text in texts:
  text=str(text)
  if current and (len(current)>=8 or chars+len(text)>45000):
   batches.append(current);current=[];chars=0
  current.append(text);chars+=len(text)
 if current:batches.append(current)
 def request_batch(batch):
  req=urllib.request.Request(URL,data=json.dumps({'model':MODEL,'input':batch},ensure_ascii=False).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer dummy'})
  try:
   with urllib.request.urlopen(req,timeout=300) as r:
    return [x['embedding'] for x in sorted(json.loads(r.read())['data'],key=lambda x:x['index'])]
  except Exception:
   # LongMemEval sessions vary sharply in length; split an unlucky request
   # rather than failing the complete retrieval stage on one provider 500.
   if len(batch)>1:
    mid=len(batch)//2;return request_batch(batch[:mid])+request_batch(batch[mid:])
   raise
 for batch in batches:out.extend(request_batch(batch))
 a=np.asarray(out,dtype=np.float32);return a/np.linalg.norm(a,axis=1,keepdims=True).clip(min=1e-8)
raw=rows(INPUT/f'{PREFIX}_RAW_VIEWS.jsonl');facts=rows(INPUT/f'{PREFIX}_ASSERTIONS.jsonl');props=rows(INPUT/f'{PREFIX}_PROPOSITIONS.jsonl');queries=rows(QUERY);start=time.time()
RV=embed([x['retrieval_text'] for x in raw]);FV=embed([x['retrieval_text'] for x in facts]);PV=embed([x['retrieval_text'] for x in props]);QQ=embed([x['question'] for x in queries]);QS=embed([x['query_text'] for x in queries])
raw_by=defaultdict(list);fact_by=defaultdict(list);prop_by=defaultdict(list)
for i,x in enumerate(raw):raw_by[x['conversation_id']].append(i)
for i,x in enumerate(facts):fact_by[x['conversation_id']].append(i)
for i,x in enumerate(props):prop_by[x['conversation_id']].append(i)
def raw_rank(vec,cid):
 data=[(raw[i]['raw_id'],float(vec@RV[i])) for i in raw_by[cid]];return sorted(data,key=lambda x:(-x[1],x[0]))
def aggregate(vec,cid,items,matrix,index):
 scores=defaultdict(lambda:-2.0)
 for i in index.get(cid,[]):scores[items[i]['raw_id']]=max(scores[items[i]['raw_id']],float(vec@matrix[i]))
 return sorted(scores.items(),key=lambda x:(-x[1],x[0]))
outputs=[]
for qi,q in enumerate(queries):
 cid=q['conversation_id'];lanes={
  'raw_question':raw_rank(QQ[qi],cid),'raw_structured':raw_rank(QS[qi],cid),
  'assertion_question':aggregate(QQ[qi],cid,facts,FV,fact_by),'assertion_structured':aggregate(QS[qi],cid,facts,FV,fact_by),
  'proposition':aggregate(QS[qi],cid,props,PV,prop_by)}
 outputs.append({'sample_id':q['sample_id'],'rankings':{k:[rid for rid,_ in v[:128]] for k,v in lanes.items()},'gold_visible':False})
with (OUTPUT/'DENSE_LANES.jsonl').open('w') as f:
 for x in outputs:f.write(json.dumps(x)+'\n')
(OUTPUT/'DENSE_RECEIPT.json').write_text(json.dumps({'questions':len(queries),'raws':len(raw),'assertions':len(facts),'propositions':len(props),'model':MODEL,'elapsed_seconds':time.time()-start,'gold_used':False},indent=2)+'\n')
print((OUTPUT/'DENSE_RECEIPT.json').read_text())
