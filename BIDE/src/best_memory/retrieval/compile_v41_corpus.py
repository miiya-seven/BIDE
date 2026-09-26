#!/usr/bin/env python3
import json,os
from collections import Counter
from pathlib import Path
try:
 from .assertion_text import render_assertion
 from .query_contract import publish_query
except ImportError:
 from assertion_text import render_assertion
 from query_contract import publish_query
H=Path(__file__).resolve().parent;B=H.parent
OUT=Path(os.environ.get('CORPUS_OUTPUT_DIR',H));OUT.mkdir(parents=True,exist_ok=True)
PREFIX=os.environ.get('CORPUS_PREFIX','V41')
TEXT_STYLE=os.environ.get('BEST276_ASSERTION_TEXT_STYLE','legacy')
if TEXT_STYLE not in {'legacy','deduplicated'}: raise ValueError('unknown assertion text style')
SIDE=Path(os.environ.get('CORPUS_SIDECAR',Path(os.environ.get('BEST276_RUN_ROOT',B/'../../runs/default'))/'memory/CONTEXTUAL_MEMORY_V41_SIDECAR.jsonl'))
DATA=Path(os.environ.get('BEST276_DATA',B.parent.parent/'data/locomo10.json'))
QFILES=[Path(x) for x in os.environ.get('CORPUS_QUERY_FILES','').split(os.pathsep) if x]
def rows(p):return [json.loads(x) for x in p.open(encoding='utf8') if x.strip()]
def write(name,data):
 with (OUT/name.replace('V41',PREFIX,1)).open('w',encoding='utf8') as f:
  for x in data:f.write(json.dumps(x,ensure_ascii=False)+'\n')
original={}
for conv in json.load(DATA.open(encoding='utf8')):
 cid=conv['sample_id'];src=conv['conversation'];day=1
 while f'session_{day}' in src:
  for turn_index,turn in enumerate(src[f'session_{day}'],1):original[f'{cid}::D{day}:{turn_index}']={'speaker':turn.get('speaker',''),'timestamp':src.get(f'session_{day}_date_time',''),'text':turn.get('text',''),'source_session_id':turn.get('source_session_id'),'has_answer':bool(turn.get('has_answer',False))}
  day+=1
assertions=[];propositions=[];edges=[];raw_views=[];entities=[]
for row in rows(SIDE):
 rid=row['raw_id'];cid=rid.split('::')[0];c=row.get('contextual_v41')
 raw_parts=[]
 if c:
  raw_parts.append(c.get('center_summary',''))
  for i,act in enumerate(c.get('speech_acts',[]),1):
   raw_parts.append(act.get('center_quote',''))
   a=act.get('assertion')
   if a:
    aid=f'{rid}::v41::a{i}';text=' '.join(str(a.get(k) or '') for k in ('subject','relation','value','event','time'))
    raw_parts.append(text)
    assertions.append({'assertion_id':aid,'raw_id':rid,'conversation_id':cid,**a,'retrieval_text':text,'authority':'CENTER_RAW','speech_act':'ASSERTION'})
    for key in ('subject','value'):
     if a.get(key):entities.append({'entity':str(a[key]),'raw_id':rid,'assertion_id':aid,'role':key.upper()})
   p=act.get('resolved_proposition')
   if p:
    pid=f'{rid}::v41::p{i}';text=' '.join(str(p.get(k) or '') for k in ('subject','relation','value','event','time'))
    raw_parts.append(text)
    propositions.append({'proposition_id':pid,'raw_id':rid,'conversation_id':cid,'speech_act':act.get('act'),**p,'retrieval_text':text})
    # A proposition always records its sources.  A typed speech-act edge is
    # meaningful only across distinct Raw records; CENTER is not its own edge.
    for source in p.get('source_raw_ids',[]):
     if source != rid: edges.append({'source_raw_id':rid,'target_raw_id':source,'edge_type':act.get('act'),'proposition_id':pid})
   for source in act.get('antecedent_raw_ids',[]):edges.append({'source_raw_id':rid,'target_raw_id':source,'edge_type':'ANTECEDENT'})
  for i,link in enumerate(c.get('occurrence_links',[]),1):
   raw_parts.append(' '.join(str(link.get(k) or '') for k in ('owner','event','relation')))
   for source in link.get('antecedent_raw_ids',[]):edges.append({'source_raw_id':rid,'target_raw_id':source,'edge_type':'OCCURRENCE','occurrence_index':i})
  # Generic contextual-v2 uses a flat assertions list.
  for i,a in enumerate(c.get('assertions',[]),1):
   aid=f'{rid}::ctx::a{i}';text=' '.join(str(a.get(k) or '') for k in ('subject','relation','value','event','time'))
   assertions.append({'assertion_id':aid,'raw_id':rid,'conversation_id':cid,**a,'retrieval_text':text,'authority':'CENTER_RAW','speech_act':c.get('evidence_role','ASSERTION')})
   propositions.append({'proposition_id':f'{rid}::ctx::p{i}','raw_id':rid,'conversation_id':cid,'subject':a.get('subject'),'relation':a.get('relation'),'value':a.get('value'),'event':a.get('event'),'time':a.get('time'),'retrieval_text':text,'source_assertion_id':aid,'derived_deterministically':True})
  # Flat contextual-v2 compatibility is intentionally limited to antecedents.
  # It must never fabricate occurrence or speech-act edges. Complete v4.1
  # reparsing supplies those relations explicitly above.
  if not c.get('speech_acts'):
   for source_id in c.get('antecedent_raw_ids',[]):edges.append({'source_raw_id':rid,'target_raw_id':source_id,'edge_type':'ANTECEDENT'})
  # Context reparses add semantics; they must not erase validated BASE L2 facts.
  if row.get('base_l2_direct'):
   existing={(str(x.get('subject') or '').lower(),str(x.get('relation') or '').lower(),str(x.get('value') or '').lower()) for x in assertions if x['raw_id']==rid}
   for a in row.get('base_l2_direct',[]):
    value=(a.get('answer_value') or {}).get('value');sig=(str(a.get('subject') or '').lower(),str(a.get('surface_relation') or '').lower(),str(value or '').lower())
    if sig in existing:continue
    aid=a.get('assertion_id');text=render_assertion(a) if TEXT_STYLE=='deduplicated' else ' '.join(str(x or '') for x in (a.get('subject'),a.get('surface_relation'),value,a.get('retrieval_text')))
    assertions.append({'assertion_id':aid,'raw_id':rid,'conversation_id':cid,'subject':a.get('subject'),'relation':a.get('surface_relation'),'value':value,'event':(a.get('roles') or {}).get('event'),'time':(a.get('scope') or {}).get('time_expression'),'modality':a.get('modality'),'polarity':a.get('polarity'),'retrieval_text':text,'authority':'BASE_VALIDATED','speech_act':'ASSERTION'})
    raw_parts.append(text);existing.add(sig)
    if a.get('subject'):entities.append({'entity':str(a['subject']),'raw_id':rid,'assertion_id':aid,'role':'SUBJECT'})
 else:
  for a in row.get('base_l2_direct',[]):
   aid=a.get('assertion_id');value=(a.get('answer_value') or {}).get('value');text=render_assertion(a) if TEXT_STYLE=='deduplicated' else ' '.join(str(x or '') for x in (a.get('subject'),a.get('surface_relation'),value,a.get('retrieval_text')))
   assertions.append({'assertion_id':aid,'raw_id':rid,'conversation_id':cid,'subject':a.get('subject'),'relation':a.get('surface_relation'),'value':value,'event':(a.get('roles') or {}).get('event'),'time':(a.get('scope') or {}).get('time_expression'),'modality':a.get('modality'),'polarity':a.get('polarity'),'retrieval_text':text,'authority':'BASE_VALIDATED','speech_act':'ASSERTION'})
   raw_parts.append(text)
   if a.get('subject'):entities.append({'entity':str(a['subject']),'raw_id':rid,'assertion_id':aid,'role':'SUBJECT'})
 source=original[rid]
 raw_views.append({'raw_id':rid,'conversation_id':cid,'source_session_id':source.get('source_session_id'),'has_answer':source.get('has_answer',False),'speaker':source['speaker'],'timestamp':source['timestamp'],'raw_text':source['text'],'retrieval_text':' '.join([source['speaker'],source['timestamp'],source['text'],*raw_parts]),'sidecar_status':row['sidecar_status']})
queries=[]
for path in QFILES:
 if not path.exists(): continue
 for q in rows(path):
  queries.append(publish_query(q))
# Remove exact duplicate relations while preserving distinct proposition and
# occurrence evidence, which downstream weighting treats separately.
seen=set();deduped=[]
for edge in edges:
 key=(edge.get('source_raw_id'),edge.get('target_raw_id'),edge.get('edge_type'),edge.get('proposition_id'),edge.get('occurrence_index'))
 if edge.get('source_raw_id')==edge.get('target_raw_id') or key in seen: continue
 seen.add(key);deduped.append(edge)
edges=deduped
write('V41_ASSERTIONS.jsonl',assertions);write('V41_PROPOSITIONS.jsonl',propositions);write('V41_GRAPH_EDGES.jsonl',edges);write('V41_RAW_VIEWS.jsonl',raw_views);write('V41_ENTITY_POSTINGS.jsonl',entities)
# Keep the corpus-local query copy under the canonical v41 name consumed by all
# retrieval families.  The non-suffixed file is retained as a provenance alias
# for older analysis scripts, but is not used as an input contract.
write('V41_QUERIES_REBUILT.jsonl',queries);write('V41_QUERIES.jsonl',queries)
receipt={'raw_views':len(raw_views),'assertions':len(assertions),'propositions':len(propositions),'graph_edges':len(edges),'entity_postings':len(entities),'queries':len(queries),'edge_types':dict(Counter(x['edge_type'] for x in edges)),'gold_used':False}
receipt['assertion_text_style']=TEXT_STYLE
(OUT/f'{PREFIX}_CORPUS_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
