#!/usr/bin/env python3
"""Normalize LoCoMo/LongMemEval records into the method's request contract.

LongMemEval is intentionally handled here instead of by copying its data into
the LoCoMo schema.  Each question owns a timestamped haystack; preserving the
question id and source session id is required for the official retrieval
metrics after the method has ranked raw records.
"""
import argparse, hashlib, json
from pathlib import Path

def load(path):
    text=Path(path).read_text(encoding='utf8').strip()
    if path.suffix.lower()=='.jsonl': return [json.loads(x) for x in text.splitlines() if x.strip()]
    obj=json.loads(text); return obj if isinstance(obj,list) else [obj]
def _lme_turns(record, *, granularity="session"):
    """Return normalized LongMemEval turns.

    LongMemEval's retrieval unit is a session, but the 276-compatible Memory
    extraction unit is a turn/utterance.  ``turn`` is therefore the default
    for LongMemEval; ``session`` remains an explicit throughput experiment and
    ``conversation`` an explicit upper-level view.  Turn mode preserves
    ``has_answer`` and source-session metadata.
    """
    sessions = record.get("haystack_sessions") or []
    dates = record.get("haystack_dates") or []
    session_ids = record.get("haystack_session_ids") or []
    out = []
    for day, session in enumerate(sessions, 1):
        source_session_id = str(session_ids[day - 1]) if day - 1 < len(session_ids) else f"session-{day}"
        timestamp = str(dates[day - 1]) if day - 1 < len(dates) else ""
        if granularity == "session":
            parts = []
            clauses = []
            has_answer = False
            for index, item in enumerate(session if isinstance(session, list) else [], 1):
                if not isinstance(item, dict):
                    continue
                role = str(item.get("role", item.get("speaker", ""))).strip()
                text = str(item.get("content", item.get("text", ""))).strip()
                if not text:
                    continue
                has_answer = has_answer or bool(item.get("has_answer"))
                clause_text = f"{role}: {text}" if role else text
                parts.append(clause_text)
                # A session is one retrieval unit, but it is not one
                # proposition.  Keep each source turn as an independently
                # addressable clause so the L1/L2 extractor can preserve
                # speaker boundaries and provenance without duplicating the
                # entire session in every clause.
                clauses.append({"clause_id": f"c{index}", "text": clause_text})
            if parts:
                out.append((day, 1, {
                    "speaker": "session",
                    "text": "\n".join(parts),
                    "timestamp": timestamp,
                    "source_session_id": source_session_id,
                    "has_answer": has_answer,
                    "clauses": clauses,
                }))
            continue
        for index, item in enumerate(session if isinstance(session, list) else [], 1):
            if not isinstance(item, dict):
                continue
            out.append((day, index, {
                **item,
                "timestamp": timestamp,
                "source_session_id": source_session_id,
            }))
    return out


def turns(record, *, dataset="auto", granularity="session"):
    if (dataset == "longmemeval" or
            (dataset == "auto" and isinstance(record.get("haystack_sessions"), list))):
        return _lme_turns(record, granularity=granularity)
    conv=record.get('conversation',record.get('messages',[]))
    if isinstance(conv,list): return [(1,i+1,x) for i,x in enumerate(conv)]
    out=[]
    for key,val in conv.items():
        if key.startswith('session_') and not key.endswith('_date_time') and isinstance(val,list):
            try: day=int(key.split('_',1)[1])
            except ValueError: day=1
            out.extend((day,i+1,x) for i,x in enumerate(val))
    return out
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',required=True)
    ap.add_argument('--run-root',required=True)
    ap.add_argument('--id-field',default='sample_id')
    ap.add_argument('--sample-id',help='optional record id to select')
    ap.add_argument('--dataset',choices=['auto','locomo','longmemeval'],default='auto')
    # Raw-level Memory extraction is the 276-compatible default for
    # LongMemEval.  Session/conversation remain explicit views for retrieval
    # and evaluation, but must not silently become one giant L1/L2 request.
    ap.add_argument('--granularity',choices=['session','turn','conversation'],default=None)
    a=ap.parse_args()
    if a.granularity is None:
        a.granularity = 'turn' if a.dataset == 'longmemeval' else 'session'
    source=Path(a.input); root=Path(a.run_root); mem=[]; qs=[]; normalized=[]
    records=load(source)
    if a.sample_id: records=[r for r in records if str(r.get(a.id_field,r.get('id','')))==a.sample_id]
    if a.sample_id and not records: raise SystemExit(f'sample id not found: {a.sample_id}')
    for ri,r in enumerate(records):
        is_lme = a.dataset == 'longmemeval' or (a.dataset == 'auto' and isinstance(r.get('haystack_sessions'), list))
        cid=str(r.get(a.id_field) or r.get('question_id') or r.get('id') or f'sample-{ri}'); source_conv=r.get('conversation',{}) if isinstance(r.get('conversation',{}),dict) else {}; conv={}; byday={}
        # Conversation-owned build: keep every source session as a clause group
        # under one memory request.  The extractor may chunk this request, but
        # the resulting memory owner remains the conversation, not a session.
        if is_lme and a.granularity == 'conversation':
            all_clauses=[]; all_parts=[]
            for day,idx,t in turns(r, dataset=a.dataset, granularity='turn'):
                sid=str(t.get('source_session_id') or f'session-{day}')
                text=str(t.get('text',t.get('content','')))
                if not text.strip(): continue
                cid_clause=f's{day}::c{idx}'
                all_clauses.append({'clause_id':cid_clause,'text':f'[session_id={sid}] {t.get("speaker",t.get("role", ""))}: {text}'})
                all_parts.append(all_clauses[-1]['text'])
            if all_clauses:
                mem.append({'center':{'raw_id':f'{cid}::CONVERSATION','speaker':'conversation','text':'\n'.join(all_parts),'timestamp':str(r.get('question_date','')),'source_session_id':cid,'clauses':all_clauses},'context_only':[],'source_channels':{f'{cid}::CONVERSATION':'DIALOGUE_TEXT'}})
        request_granularity = 'session' if a.granularity == 'conversation' else a.granularity
        turn_rows = turns(r, dataset=a.dataset, granularity=request_granularity)
        # Keep a bounded, same-session context window on turn-level requests.
        # The center remains exactly one Raw; neighbors are never promoted to
        # center clauses and therefore cannot leak propositions into L2.
        by_session_turns = {}
        if is_lme and request_granularity == 'turn':
            for pos, (d0, i0, t0) in enumerate(turn_rows):
                sid0 = str(t0.get('source_session_id') or f'session-{d0}')
                by_session_turns.setdefault(sid0, []).append((pos, d0, i0, t0))
        for row_pos, (day,idx,t) in enumerate(turn_rows):
            if is_lme and a.granularity == 'conversation':
                # Conversation memory was emitted above; this loop only builds
                # the normalized conversation view used by retrieval/eval.
                pass
            # LoCoMo stores the timestamp on the session container
            # (conversation.session_N_date_time), rather than on each turn.
            # The historical 276 raw contract exposes that timestamp on every
            # Raw record so relative expressions such as "yesterday" can be
            # resolved by retrieval and the reader.
            session_date = source_conv.get(f'session_{day}_date_time', '')
            item={'speaker':str(t.get('speaker',t.get('role',''))),
                  'text':str(t.get('text',t.get('content',''))),
                  'timestamp':str(t.get('timestamp') or session_date or r.get('date',''))}
            if t.get('source_session_id'): item['source_session_id']=str(t['source_session_id'])
            if t.get('has_answer') is not None: item['has_answer']=bool(t.get('has_answer'))
            byday.setdefault(day,[]).append(item)
            supplied_clauses = t.get('clauses')
            if isinstance(supplied_clauses, list) and supplied_clauses:
                item['clauses']=[
                    {'clause_id':str(c.get('clause_id') or f'c{n}'),
                     'text':str(c.get('text') or '')}
                    for n,c in enumerate(supplied_clauses, 1)
                    if isinstance(c, dict) and str(c.get('text') or '').strip()
                ]
            if not item.get('clauses'):
                item['clauses']=[{'clause_id':f'c{idx}','text':item['text']}]
            if not (is_lme and a.granularity == 'conversation'):
                context_only=[]
                if is_lme and request_granularity == 'turn':
                    sid = str(t.get('source_session_id') or f'session-{day}')
                    peers = by_session_turns.get(sid, [])
                    center_pos = next((p0 for p0, d0, i0, _ in peers
                                       if d0 == day and i0 == idx), 0)
                    for pos0, d0, i0, t0 in peers:
                        if pos0 == center_pos or abs(pos0 - center_pos) > 3:
                            continue
                        rid0 = f'{cid}::D{d0}:{i0}'
                        context_only.append({'raw_id':rid0,
                                             'speaker':str(t0.get('speaker',t0.get('role',''))),
                                             'text':str(t0.get('text',t0.get('content',''))),
                                             'timestamp':str(t0.get('timestamp',r.get('date',''))),
                                             'source_session_id':sid,
                                             'context_only':True})
                rid=f'{cid}::D{day}:{idx}'
                mem.append({'center':{'raw_id':rid,**item},
                            'context_only':context_only,
                            'source_channels':{rid:'DIALOGUE_TEXT'}})
        for day,items in byday.items(): conv[f'session_{day}']=items; conv[f'session_{day}_date_time']=items[0].get('timestamp','') if items else ''
        normalized_record={'sample_id':cid,'conversation':conv}
        if is_lme:
            normalized_record.update({
                'dataset':'longmemeval',
                'question_id':str(r.get('question_id') or cid),
                'question_type':r.get('question_type',''),
                'question_date':r.get('question_date',''),
                'answer':r.get('answer',''),
                'answer_session_ids':[str(x) for x in (r.get('answer_session_ids') or [])],
                'haystack_session_ids':[str(x) for x in (r.get('haystack_session_ids') or [])],
                'haystack_dates':[str(x) for x in (r.get('haystack_dates') or [])],
                'granularity':a.granularity,
            })
        normalized.append(normalized_record)
        rawq=r.get('qa',r.get('questions',r.get('question',[]))); rawq=rawq if isinstance(rawq,list) else [rawq]
        for qi,x in enumerate(rawq):
            if isinstance(x,str): q=x; extra={}
            else: q=str(x.get('question',x.get('query',''))); extra={k:v for k,v in x.items() if k not in ('question','query')}
            if q:
                qid=str(r.get('question_id') or cid) if is_lme and len(rawq)==1 else f'{cid}__qa_{qi}'
                record={'sample_id':qid,'conversation_id':cid,'question':q,**extra}
                if is_lme:
                    record.update({'dataset':'longmemeval','question_id':str(r.get('question_id') or cid),
                                   'question_type':r.get('question_type',''),'question_date':r.get('question_date',''),
                                   'answer':r.get('answer',''),'answer_session_ids':[str(v) for v in (r.get('answer_session_ids') or [])],
                                   'haystack_session_ids':[str(v) for v in (r.get('haystack_session_ids') or [])]})
                qs.append(record)
    (root/'memory').mkdir(parents=True,exist_ok=True); (root/'query').mkdir(parents=True,exist_ok=True)
    (root/'data_normalized.json').write_text(json.dumps(normalized,ensure_ascii=False,indent=2)+'\n')
    (root/'memory'/'MEMORY_REQUESTS.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in mem)); (root/'query'/'QUERY_REQUESTS.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in qs)); (root/'query'/'QUERY_KEYS.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in qs))
    out={'status':'PREPARED','dataset':a.dataset,'granularity':a.granularity,'records':len(normalized),'memory_requests':len(mem),'query_requests':len(qs),'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    if a.dataset == 'longmemeval' or any(x.get('dataset') == 'longmemeval' for x in normalized):
        manifest=root/'memory'/'V2_STATUS_MANIFEST.jsonl'
        # Session records are already the benchmark's context unit. Turn rows
        # are routed only after their validated L1/L2 result is available;
        # marking every turn for contextual reparsing doubles the API work and
        # contradicts the contextual stage's selective contract.
        status='SELF_CONTAINED_RAW' if a.granularity == 'session' else 'PENDING_MEMORY_CLASSIFICATION'
        manifest.write_text(''.join(json.dumps({'raw_id':x['center']['raw_id'],'status':status},ensure_ascii=False)+'\n' for x in mem))
        out['context_manifest_status']=status
    (root/'PREPARATION_RECEIPT.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n'); print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__': main()
