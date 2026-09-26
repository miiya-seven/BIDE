#!/usr/bin/env python3
import json,time,os,re
from collections import defaultdict
from pathlib import Path
import torch
from transformers import AutoModelForMaskedLM,AutoTokenizer
H=Path(__file__).resolve().parent;INPUT=Path(os.environ.get('SPLADE_INPUT_DIR',H));OUTPUT=Path(os.environ.get('SPLADE_OUTPUT_DIR',H));OUTPUT.mkdir(parents=True,exist_ok=True);M=Path(os.environ.get('SPLADE_MODEL',''))
def rows(p):return [json.loads(x) for x in p.open() if x.strip()]


def _degraded_lane(reason, raw, queries):
 """Write a transparent lexical fallback for hosts without local CUDA."""
 token_re=re.compile(r"[A-Za-z0-9_]+")
 def toks(text):return set(token_re.findall(str(text).lower()))
 by=defaultdict(list)
 for item in raw:by[item['conversation_id']].append(item)
 out=[]
 for q in queries:
  qtok=toks(q.get('query_text',''));rank=[]
  for item in by[q['conversation_id']]:
   dtok=toks(item.get('retrieval_text',''));score=float(len(qtok&dtok));rank.append((item['raw_id'],score))
  rank.sort(key=lambda x:(-x[1],x[0]))
  out.append({'sample_id':q['sample_id'],'rankings':{'splade':[rid for rid,_ in rank[:128]]},'gold_visible':False})
 with (OUTPUT/'SPLADE_LANE.jsonl').open('w') as f:
  for x in out:f.write(json.dumps(x,ensure_ascii=False)+'\n')
 receipt={'questions':len(queries),'raws':len(raw),'model':str(M),'backend':'degraded_lexical','score':'lexical_overlap_fallback','degraded_reason':reason,'elapsed_seconds':time.time()-start,'gold_used':False}
 (OUTPUT/'SPLADE_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))


raw=rows(INPUT/'V41_RAW_VIEWS.jsonl');queries=rows(INPUT/'V41_QUERIES_REBUILT.jsonl');start=time.time()
allow_degraded=os.environ.get('BEST276_ALLOW_DEGRADED_LANES','0')=='1'
if allow_degraded and not torch.cuda.is_available():
 _degraded_lane('cuda_unavailable',raw,queries);raise SystemExit(0)
try:
 tok=AutoTokenizer.from_pretrained(M,local_files_only=True);model=AutoModelForMaskedLM.from_pretrained(M,local_files_only=True,torch_dtype=torch.float16).cuda().eval()
except Exception as exc:
 if allow_degraded:
  _degraded_lane(f'local_model_unavailable:{type(exc).__name__}',raw,queries);raise SystemExit(0)
 raise
@torch.inference_mode()
def enc(texts,batch,maxlen,topk):
 out=[]
 for i in range(0,len(texts),batch):
  z=tok(texts[i:i+batch],padding=True,truncation=True,max_length=maxlen,return_tensors='pt').to('cuda');v=torch.log1p(torch.relu(model(**z).logits))*z['attention_mask'].unsqueeze(-1);p=v.max(1).values;w,ids=torch.topk(p,k=topk,dim=1)
  for a,b in zip(w,ids):out.append({int(j):float(k) for j,k in zip(b[a>0].tolist(),a[a>0].float().tolist())})
 return out
D=enc([x['retrieval_text'] for x in raw],16,320,128);Q=enc([x['query_text'] for x in queries],16,160,64);by=defaultdict(list)
for i,x in enumerate(raw):by[x['conversation_id']].append(i)
out=[]
for qi,q in enumerate(queries):
 rank=[]
 for i in by[q['conversation_id']]:rank.append((raw[i]['raw_id'],sum(w*D[i].get(t,0) for t,w in Q[qi].items())))
 rank.sort(key=lambda x:(-x[1],x[0]));out.append({'sample_id':q['sample_id'],'rankings':{'splade':[rid for rid,_ in rank[:128]]},'gold_visible':False})
with (OUTPUT/'SPLADE_LANE.jsonl').open('w') as f:
 for x in out:f.write(json.dumps(x)+'\n')
(OUTPUT/'SPLADE_RECEIPT.json').write_text(json.dumps({'questions':len(queries),'raws':len(raw),'model':str(M),'elapsed_seconds':time.time()-start,'gold_used':False},indent=2)+'\n');print((OUTPUT/'SPLADE_RECEIPT.json').read_text())
