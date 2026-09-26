"""One additional retrieval round, same conversation, generated gap queries only."""
import json, os, urllib.request
os.environ.setdefault('HF_HUB_OFFLINE','1')
import numpy as np
import torch
try:
    from .common import H, RAW, QUERIES, RANKS, sha, save
except ImportError:  # direct-script compatibility
    from common import H, RAW, QUERIES, RANKS, sha, save

class Retriever:
    def __init__(self):
        self.ids=sorted(RAW)
        self.embedding_base=os.environ.get('EMBEDDING_BASE_URL','').rstrip('/')
        self.embedding_key=os.environ.get('EMBEDDING_API_KEY','dummy')
        self.embedding_model=os.environ.get('EMBEDDING_MODEL','BAAI/bge-m3')
        self.reranker_base=os.environ.get('RERANKER_BASE_URL','').rstrip('/')
        self.reranker_path=os.environ.get('RERANKER_API_PATH','/v1/rerank')
        self.reranker_key=os.environ.get('RERANKER_API_KEY','dummy')
        self.reranker_model=os.environ.get('RERANKER_MODEL','Qwen/Qwen3-Reranker-8B')
        self.remote=bool(self.embedding_base and self.reranker_base)
        if not self.remote:
            from transformers import AutoTokenizer,AutoModelForCausalLM,AutoModel
            # Same dense CLS + normalization operation as installed M3 modeling.py.
            bge=os.environ.get('BGE_M3_MODEL','')
            qwen=os.environ.get('QWEN_RERANKER_MODEL','')
            if not bge or not qwen:
                raise RuntimeError('gap retrieval requires embedding+reranker APIs or local BGE/Qwen paths')
            self.enc_tok=AutoTokenizer.from_pretrained(bge,local_files_only=True)
            self.encoder=AutoModel.from_pretrained(bge,torch_dtype=torch.float16,local_files_only=True).cuda().eval()
        fingerprint=sha([(r,RAW[r]['retrieval_text']) for r in self.ids])
        path=H/f'bge_dense_{fingerprint}.npy'
        if path.exists():self.matrix=np.load(path)
        else:
            self.matrix=self.encode([RAW[r]['retrieval_text'] for r in self.ids])
            np.save(path,self.matrix)
        self.groups={}
        for i,r in enumerate(self.ids):self.groups.setdefault(RAW[r]['conversation_id'],[]).append(i)
        if not self.remote:
            self.tok=AutoTokenizer.from_pretrained(qwen,padding_side='left',local_files_only=True)
            self.model=AutoModelForCausalLM.from_pretrained(qwen,torch_dtype=torch.bfloat16,
                        attn_implementation='sdpa',local_files_only=True).cuda().eval()
            self.pi=self.tok.encode('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n',add_special_tokens=False)
            self.si=self.tok.encode('<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n',add_special_tokens=False)
            self.no=self.tok.convert_tokens_to_ids('no');self.yes=self.tok.convert_tokens_to_ids('yes')
    def encode(self,texts):
        if self.remote:
            out=[]
            for st in range(0,len(texts),32):
                request=urllib.request.Request(self.embedding_base+'/embeddings',data=json.dumps({'model':self.embedding_model,'input':texts[st:st+32]}).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+self.embedding_key},method='POST')
                with urllib.request.urlopen(request,timeout=float(os.environ.get('EMBEDDING_TIMEOUT_SECONDS','300'))) as response:payload=json.loads(response.read())
                out.extend(x['embedding'] for x in sorted(payload['data'],key=lambda x:x['index']))
            matrix=np.asarray(out,dtype=np.float32);return matrix/np.linalg.norm(matrix,axis=1,keepdims=True).clip(min=1e-8)
        out=[]
        for st in range(0,len(texts),32):
            batch=self.enc_tok(texts[st:st+32],padding=True,truncation=True,max_length=2048,return_tensors='pt').to('cuda')
            with torch.inference_mode():
                hidden=self.encoder(**batch).last_hidden_state[:,0]
                out.append(torch.nn.functional.normalize(hidden.float(),dim=-1).cpu().numpy())
        return np.concatenate(out)
    def search(self,s,initial,gaps):
        assert 1<=len(gaps)<=2
        idx=self.groups[QUERIES[s]['conversation_id']]
        candidates={};lanes=[]
        threshold=float(os.environ.get('GAP_RERANK_MIN_SCORE','0.5'))
        for gap in gaps:
            query=gap['retrieval_query']
            v=self.encode([query])[0]
            similarity=self.matrix[idx]@v
            order=np.argsort(-similarity,kind='stable')[:64]
            dense=[self.ids[idx[int(j)]] for j in order]
            # The frozen Candidate128 has already demonstrated much higher
            # recall than the small initial packet.  Completion may inspect its
            # tail without changing or reordering the frozen TopK input.
            frozen_tail=[r for r in RANKS[s]['ranking'] if r not in initial]
            proposals=list(dict.fromkeys(dense+frozen_tail))
            constraint=(f"Fill this exact evidence gap, not merely the same topic. Missing role: {gap['missing_role']}. "
                        f"Target entity: {gap['target_entity']}. Target occurrence: {gap['target_occurrence']}. "
                        f"Time scope: {gap['time_scope']}. Competing occurrences to avoid: "
                        f"{'; '.join(str(x) for x in gap['competing_occurrences']) or 'none identified'}.")
            texts=[f'<Instruct>: {constraint}\n<Query>: {QUERIES[s]["question"]}\nMissing evidence: {query}\n<Document>: {RAW[r]["retrieval_text"]}' for r in proposals]
            scores=[]
            if self.remote:
                endpoint_base=self.reranker_base[:-3] if self.reranker_base.endswith('/v1') and self.reranker_path.startswith('/v1/') else self.reranker_base
                endpoint=endpoint_base+self.reranker_path if self.reranker_path.startswith('/') else endpoint_base+'/'+self.reranker_path
                body={'model':self.reranker_model,'query':QUERIES[s]['question']+'\n'+query+'\n'+constraint,
                      'documents':[RAW[r]['retrieval_text'] for r in proposals]}
                request=urllib.request.Request(endpoint,data=json.dumps(body,ensure_ascii=False).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+self.reranker_key},method='POST')
                with urllib.request.urlopen(request,timeout=float(os.environ.get('RERANKER_TIMEOUT_SECONDS','300'))) as response:payload=json.loads(response.read())
                data=payload.get('results',payload.get('data',[]))
                scores=[float(x.get('relevance_score',x.get('score'))) for x in sorted(data,key=lambda x:x.get('index',0))]
                if len(scores)!=len(proposals):raise RuntimeError('gap reranker response length mismatch')
            else:
                for st in range(0,len(texts),16):
                    vs=self.tok(texts[st:st+16],padding=False,truncation=True,max_length=2048-len(self.pi)-len(self.si),add_special_tokens=False)['input_ids']
                    batch=self.tok.pad({'input_ids':[self.pi+x+self.si for x in vs]},padding=True,return_tensors='pt').to('cuda')
                    with torch.inference_mode():
                        h=self.model.model(**batch).last_hidden_state[:,-1,:]
                        logits=torch.nn.functional.linear(h,self.model.lm_head.weight[[self.no,self.yes]]).float()
                        scores.extend(logits.softmax(-1)[:,1].cpu().tolist())
            ranking=sorted(zip(proposals,scores),key=lambda x:(-x[1],x[0]))
            lanes.append({'gap':gap,'dense_top64':dense,'frozen_candidate_tail':frozen_tail,
                          'proposal_union_size':len(proposals),'reranked':ranking,'minimum_score':threshold})
            for r,value in ranking:
                if r not in initial and value>=threshold:candidates[r]=max(value,candidates.get(r,0))
        added=sorted(candidates,key=lambda r:(-candidates[r],r))[:5]
        final=initial+added
        assert len(final)<=len(initial)+5 and len(set(final))==len(final)
        assert all(RAW[r]['conversation_id']==QUERIES[s]['conversation_id'] for r in final)
        return {'sample_id':s,'initial':initial,'added':added,'final':final,'lanes':lanes,
                'selection':{'minimum_score':threshold,'qualified_candidates':len(candidates),'max_added':5}}
