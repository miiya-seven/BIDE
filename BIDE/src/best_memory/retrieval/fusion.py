#!/usr/bin/env python3
"""Gold-blind fusion-only ablation over the six frozen stable families."""
import json
import os
from pathlib import Path
ROOT=Path(os.environ.get('BEST276_ROOT',Path(__file__).resolve().parents[4])); B=ROOT/'src'; RUN=Path(os.environ.get('BEST276_RUN_ROOT',ROOT/'runs/default')); H=Path(os.environ.get('FUSION_OUTPUT_DIR',RUN/'retrieval'));H.mkdir(parents=True,exist_ok=True)
INPUT_ROOT=Path(os.environ.get('BEST276_INPUT_ROOT',ROOT/'runs/default'))
def rows(p):return [json.loads(x) for x in open(p,encoding='utf8') if x.strip()]
def collapse(p):
 out={}
 for x in rows(p):
  best={}
  for v in x['rankings'].values():
   for i,r in enumerate(v[:128],1):best[r]=min(best.get(r,10**9),i)
  out[x['sample_id']]=best
 return out
E=INPUT_ROOT/'retrieval';F={
 'clause':collapse(E/'CLAUSE/CLAUSE_LANES_REBUILT_QUERY.jsonl'),'occurrence':collapse(E/'OCCURRENCE/OCCURRENCE_LANES.jsonl'),'structural':collapse(E/'STRUCTURAL/V41_LEGACY_STRUCTURAL_LANES.jsonl'),'multivector':collapse(E/'MULTIVECTOR/V41_MULTIVECTOR_LANES.jsonl'),'splade':collapse(E/'SPLADE/SPLADE_LANE.jsonl')}
typed={}
typed_path=E/'TYPED_L2_LANES.jsonl'
if not typed_path.exists():
    raise FileNotFoundError(f"missing typed-l2 lanes: {typed_path}; run the typed-l2 stage before fusion")
for x in rows(typed_path):
 tabs=[{r:i for i,r in enumerate(v[:128],1)} for v in x['rankings'].values()];u=set().union(*(set(t) for t in tabs));sc={r:sum(1/(20+t[r]) for t in tabs if r in t) for r in u};rr=sorted(u,key=lambda r:(-sc[r],r))[:128];typed[x['sample_id']]={r:i for i,r in enumerate(rr,1)}
F['typed_l2']=typed
out=[]
for sid in sorted(typed):
 tabs=[F[n][sid] for n in F];u=set().union(*(set(t) for t in tabs));scores={}
 for r in u:
  vals=[1/(20+t[r]) for t in tabs if r in t];best=max(vals);rrf=sum(vals)
  scores[r]={'RRF_BASELINE':rrf,'COMBMAX':best,'SHARP_RRF_P2':sum(v*v for v in vals),'BEST_PRIMARY_L025':best+.25*(rrf-best)}
 rankings={arm:sorted(u,key=lambda r:(-scores[r][arm],r))[:128] for arm in next(iter(scores.values()))}
 out.append({'sample_id':sid,'rankings':rankings,'candidate_size':len(next(iter(rankings.values()),[])),'gold_visible':False})
with open(H/'CANDIDATE128.jsonl','w',encoding='utf8') as f:
 for x in out:f.write(json.dumps(x,ensure_ascii=False)+'\n')
(H/'CONSTRUCTION.json').write_text(json.dumps({'schema':'fusion-geometry-ablation-v1','gold_used':False,'families':list(F),'family_depth':128,'candidate_policy':'min(128, available unique raw views)','rrf_k':20,'arms':{'RRF_BASELINE':'sum reciprocal','COMBMAX':'max reciprocal','SHARP_RRF_P2':'sum reciprocal squared','BEST_PRIMARY_L025':'best + 0.25*(rrf-best)'},'tie_break':'raw_id ascending'},indent=2)+'\n')
print(len(out))
