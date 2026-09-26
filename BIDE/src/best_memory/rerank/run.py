"""Full-domain frozen RRF6 / structured Qwen8B reranking, resumable pair cache."""
import argparse
import hashlib
import json
import time
import os
import urllib.request
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

H = Path(os.environ.get('BEST276_RUN_ROOT', Path(__file__).resolve().parents[4] / 'runs/default')) / 'rerank'
H.mkdir(parents=True, exist_ok=True)
B = Path(os.environ.get('BEST276_ROOT', Path(__file__).resolve().parents[4]))/'runs'
def rows(p): return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def save(name, obj): (H/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2))
a=argparse.ArgumentParser()
a.add_argument('--batch-size',type=int,default=64)
args=a.parse_args()
C=Path(os.environ.get('BEST276_INPUT_ROOT',B/'default'))/'retrieval'
qpath=Path(os.environ.get('BEST276_RERANK_QUERY',C/'V41_QUERIES_REBUILT.jsonl'))
rpath=Path(os.environ.get('BEST276_RERANK_RAW',C/'V41_RAW_VIEWS.jsonl'))
ppath=Path(os.environ.get('BEST276_FUSION_CANDIDATE',Path(os.environ.get('BEST276_RUN_ROOT',B/'default'))/'retrieval/CANDIDATE128.jsonl'))
qs={x['sample_id']:x for x in rows(qpath)}
raw={x['raw_id']:x for x in rows(rpath)}
pools={x['sample_id']:x['rankings']['RRF_BASELINE'] for x in rows(ppath)}
ids=sorted(qs)
assert set(ids)==set(pools)
for s in ids:
    expected=min(128,len(pools[s])) if os.environ.get('BEST276_DATASET')=='longmemeval' else 128
    assert len(pools[s])==expected and len(set(pools[s]))==expected
    assert all(raw[r]['conversation_id']==qs[s]['conversation_id'] for r in pools[s])
modelpath=os.environ.get('QWEN_RERANKER_MODEL','')
prefix='<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
suffix='<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
instruction='Given a web search query, retrieve relevant passages that answer the query'
protocol={'ids':ids,'pool':'current fusion Candidate128','input':'query_text / retrieval_text',
              'model':modelpath,'prefix':prefix,'suffix':suffix,'instruction':instruction,'max_length':2048,
          'dtype':'bf16','batch_size':args.batch_size,'candidate_policy':'min(128, available unique raw views)',
          'batching':'ascending token length; numerical batching differences possible',
          'source_sha256':{str(p.relative_to(B)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [qpath,rpath,ppath]}}
fingerprint=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
if (H/'PROTOCOL.json').exists():
    assert json.loads((H/'PROTOCOL.json').read_text())==protocol, 'Existing protocol differs; use a fresh output directory.'
save('PROTOCOL.json',protocol)

# Prefer the explicitly configured OpenAI-compatible reranker service when it
# is available.  This keeps the method runnable on machines where the local
# CUDA driver is unavailable while preserving the same 128-document scoring
# contract and model identity in the protocol.
remote_base=os.environ.get('RERANKER_BASE_URL','').rstrip('/')
remote_path=os.environ.get('RERANKER_API_PATH','/v1/rerank')
if remote_base:
    endpoint=(remote_base[:-3] if remote_base.endswith('/v1') and remote_path.startswith('/v1/') else remote_base)
    endpoint=endpoint+remote_path if remote_path.startswith('/') else endpoint+'/'+remote_path
    remote_key=os.environ.get('RERANKER_API_KEY','dummy')
    rankings=[];started=time.time()
    for number,s in enumerate(ids,1):
        docs=[raw[r]['retrieval_text'] for r in pools[s]]
        body={'model':os.environ.get('RERANKER_MODEL',modelpath),'query':qs[s]['query_text'],'documents':docs,
              'instruction':'Given a web search query, retrieve relevant passages that answer the query','top_n':len(docs)}
        request=urllib.request.Request(endpoint,data=json.dumps(body,ensure_ascii=False).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+remote_key},method='POST')
        with urllib.request.urlopen(request,timeout=float(os.environ.get('RERANKER_TIMEOUT_SECONDS','300'))) as response:
            payload=json.loads(response.read())
        result=payload.get('results',payload.get('data',[]))
        if len(result)!=len(docs): raise RuntimeError(f'remote reranker returned {len(result)} scores for {len(docs)} docs: {s}')
        ordered=sorted(result,key=lambda x:(-float(x.get('relevance_score',x.get('score'))),int(x.get('index',0))))
        rankings.append({'sample_id':s,'original':pools[s],
                         'ranking':[pools[s][int(x.get('index',0))] for x in ordered],
                         'reranker_backend':'http','gold_visible':False})
        if number<=2 or number%25==0 or number==len(ids): print(f'Remote rerank {number}/{len(ids)}',flush=True)
    save('RANKINGS.json',rankings)
    save('RUNTIME.json',{'elapsed_seconds':time.time()-started,'new_pairs':sum(len(pools[s]) for s in ids),
                         'protocol_sha256':fingerprint,'torch':torch.__version__,'gpu':None,
                         'backend':'http','endpoint':endpoint,'model':os.environ.get('RERANKER_MODEL',modelpath)})
    print(f'All {len(ids)} rankings frozen by remote reranker.',flush=True)
    raise SystemExit(0)
scorepath=H/'SCORES.npy'
if scorepath.exists():
    scores=np.lib.format.open_memmap(scorepath,mode='r+')
    assert scores.shape==(len(ids),128)
else:
    scores=np.lib.format.open_memmap(scorepath,mode='w+',dtype='float32',shape=(len(ids),128))
    scores[:]=np.nan
    scores.flush()
start=time.time()
tok=AutoTokenizer.from_pretrained(modelpath,padding_side='left',local_files_only=True)
pi=tok.encode(prefix,add_special_tokens=False);si=tok.encode(suffix,add_special_tokens=False)
limit=2048-len(pi)-len(si)
pending=[]
truncated=0
token_total=0
for i,s in enumerate(ids):
    indexes=[j for j in range(128) if np.isnan(scores[i,j])]
    if not indexes: continue
    texts=[f'<Instruct>: {instruction}\n<Query>: {qs[s]["query_text"]}\n<Document>: {raw[pools[s][j]]["retrieval_text"]}' for j in indexes]
    enc=tok(texts,padding=False,truncation=False,add_special_tokens=False)['input_ids']
    for j,v in zip(indexes,enc):
        truncated+=len(v)>limit
        v=pi+v[:limit]+si
        token_total+=len(v)
        pending.append((len(v),i,j,v))
pending.sort(key=lambda x:(x[0],x[1],x[2]))
save('PENDING_INPUT_STATS.json',{'pairs':len(pending),'truncated':truncated,'tokens':token_total,
                              'min_length':pending[0][0] if pending else 0,'max_length':pending[-1][0] if pending else 0})
print('Tokenized',len(pending),'pending pairs; tokens',token_total,'truncated',truncated,flush=True)
if pending:
    model=AutoModelForCausalLM.from_pretrained(modelpath,torch_dtype=torch.bfloat16,
          attn_implementation='sdpa',local_files_only=True).cuda().eval()
    ni,yi=tok.convert_tokens_to_ids('no'),tok.convert_tokens_to_ids('yes')
    inferstart=time.time()
    lastlog=0
    for st in range(0,len(pending),args.batch_size):
        chunk=pending[st:st+args.batch_size]
        batch=tok.pad({'input_ids':[v[3] for v in chunk]},padding=True,return_tensors='pt').to('cuda')
        with torch.inference_mode():
            h=model.model(**batch).last_hidden_state[:,-1,:]
            logits=torch.nn.functional.linear(h,model.lm_head.weight[[ni,yi]]).float()
            values=logits.softmax(-1)[:,1].cpu().numpy()
        for (_,i,j,_),v in zip(chunk,values): scores[i,j]=v
        now=time.time()
        if now-lastlog>=30 or st+len(chunk)==len(pending):
            scores.flush()
            done=st+len(chunk)
            status={'new_pairs_done':done,'new_pairs_total':len(pending),'all_pairs_done':int(np.isfinite(scores).sum()),
                    'total_pairs':int(scores.size),'inference_seconds':now-inferstart,'max_batch_tokens':chunk[-1][0],
                    'estimated_remaining_seconds':(len(pending)-done)*(now-inferstart)/done}
            save('PROGRESS.json',status)
            print(json.dumps(status),flush=True)
            lastlog=now
    scores.flush()
assert np.isfinite(scores).all() and ((scores>=0)&(scores<=1)).all()
rankings=[{'sample_id':s,'original':pools[s],
           'ranking':sorted(pools[s],key=lambda r:(-float(scores[i,pools[s].index(r)]),r))} for i,s in enumerate(ids)]
save('RANKINGS.json',rankings)
save('RUNTIME.json',{'elapsed_seconds':time.time()-start,'new_pairs':len(pending),'protocol_sha256':fingerprint,
                     'torch':torch.__version__,'gpu':torch.cuda.get_device_name() if torch.cuda.is_available() else None})
print(f'All {len(ids)} rankings frozen. Run evaluate.py.',flush=True)
