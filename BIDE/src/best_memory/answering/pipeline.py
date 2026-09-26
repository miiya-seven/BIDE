import argparse
import ast
import concurrent.futures
import fcntl
import json
import os
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
try:
    from .common import API,H,B,ROOT,RANKS,QUERIES,RAW,sha,save,rows
    from .policy import should_retrieve,select_answer
except ImportError:  # direct-script compatibility
    from common import API,H,B,ROOT,RANKS,QUERIES,RAW,sha,save,rows
    from policy import should_retrieve,select_answer

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default=str(ROOT/'configs/main_experiment.yaml'))
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    # A process-local threading lock cannot prevent two resumed pipelines from
    # issuing the same content-addressed request.  Hold one OS lock for the
    # complete run so accidental double starts fail before any API work.
    lock_file=(H/'.pipeline.lock').open('w')
    try:
        fcntl.flock(lock_file,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('another 276 pipeline instance is already running')
    ids=sorted(RANKS,key=lambda s:sha(s))
    # Canonical 276 runs may provide an explicit query-id manifest (for
    # example the historical 1540-question LoCoMo contract).  Filter before
    # constructing Top10/Top20 cells so extended QA rows never enter the
    # official answering denominator.
    id_manifest=os.environ.get('BEST276_ID_MANIFEST')
    if id_manifest:
        manifest_path=Path(id_manifest)
        allowed={line.strip() for line in manifest_path.read_text().splitlines() if line.strip()}
        ids=[s for s in ids if s in allowed]
    if args.limit:ids=ids[:args.limit]
    run=H/(f'smoke_{args.limit}' if args.limit else 'full')
    run.mkdir(exist_ok=True)
    api=API(args.config)
    dataset_name=os.environ.get('BEST276_DATASET','locomo')
    protocol={'ids':ids,'ks':[10,20],'model':api.client.llm_model,'provider':api.client.llm_base,
              'initial':'274 Qwen8B structured ranking','arms':['DIRECT','CHECKED'],
              'checker':'answer-aware claim support, requirement coverage, target alignment, conflicts, and actionable gaps',
              'max_retrieval_rounds':1,'max_gap_queries':2,'retrieval':'same-conversation BGE-M3 Top64 per structured gap; gap-aware Qwen8B reranking with score threshold',
              'max_added_raw':5,'retain_initial':True,'second_check_failure':'retain direct answer; revised answer requires new-evidence certificate',
              'judge':('unchanged JUDGE_PROMPT from scripts/judge_locomo_official.py'
                       if dataset_name=='locomo' else 'LongMemEval reference-equivalence JSON judge'),
              'dataset':dataset_name,
              'granularity':os.environ.get('BEST276_LME_GRANULARITY') if dataset_name=='longmemeval' else None,
              'gold_visibility':'Gold/reference loaded only after all generation is frozen',
              'interpretation':'CHECKED final budgets <=15/25; QA gains can include additional evidence budget',
              'source_hashes':{'frozen_rankings':sha(RANKS),'queries':sha(QUERIES),'raw':sha(RAW)},
              'answer_prompt':'occurrence/time/scope aware; caption enabled; answer claims and exact evidence citations',
              'implementation_note':'Schema-normalization repairs may resume failed stages; every generated request/response remains content-addressed.'}
    if (run/'PROTOCOL.json').exists():
        prior=json.loads((run/'PROTOCOL.json').read_text())
        for key in ['ids','ks','model','provider','initial','arms','max_retrieval_rounds','max_gap_queries','retrieval','max_added_raw','retain_initial','second_check_failure','judge','gold_visibility','interpretation']:
            assert prior[key]==protocol[key], 'semantic protocol changed: '+key
    save(run/'PROTOCOL.json',protocol)
    cells=[(s,k) for s in ids for k in [10,20]]
    def stage(name,work,items):
        """Run a stage with resumable, bounded recovery.

        Successful cells are materialized by the worker itself, so a retry
        must submit only cells that are still missing.  This is important for
        the 1540-question run: a transient provider failure must not discard a
        mostly complete answer stage or produce a short judge denominator.
        """
        pending=list(items); results=[]; last_errors=[]
        for recovery_round in range(4):
            errors=[]; round_results=[]
            total=len(pending)
            save(run/'STATUS.json',{'stage':name,'done':0,'total':total,
                                    'recovery_round':recovery_round,
                                    'status':'RUNNING'})
            if not pending: return results
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures={pool.submit(work,*item):item for item in pending}
                for future in concurrent.futures.as_completed(futures):
                    try: round_results.append(future.result())
                    except Exception as exc:
                        errors.append({'cell':futures[future],
                                       'error_type':type(exc).__name__,
                                       'error':str(exc)[:500]})
                        print('stage_error', futures[future], type(exc).__name__, str(exc)[:500], flush=True)
                    done=len(round_results)+len(errors)
                    save(run/'STATUS.json',{'stage':name,'done':done,'total':total,
                            'failed':len(errors),'recovery_round':recovery_round,'status':'RUNNING'})
                    if done<=2 or done%10==0 or done==total:
                        print(name,done,'/',total,'failed',len(errors),
                              'recovery_round',recovery_round,flush=True)
            results.extend(round_results); last_errors=errors
            if not errors: return results
            pending=[tuple(x['cell']) for x in errors]
            save(run/(name+'_errors.json'),errors)
            if recovery_round < 3:
                print(name,'retrying',len(pending),'failed cells',flush=True)
                time.sleep(min(8,2**recovery_round))
        raise RuntimeError(name+' incomplete after recovery; inspect cached requests/errors')
    def initial(s,k):
        path=run/'initial'/f'{s}_{k}.json'
        if path.exists():return json.loads(path.read_text())
        packet=RANKS[s]['ranking'][:k]
        direct=api.answer(s,packet)
        check=api.check(s,packet,direct)
        result={'sample_id':s,'k':k,'initial':packet,'direct':direct,'check1':check}
        save(path,result);return result
    pending_initial=[(s,k) for s,k in cells if not (run/'initial'/f'{s}_{k}.json').exists()]
    if pending_initial:
        stage('initial_answer_and_check_pending',initial,pending_initial)
    initial_results=[json.loads((run/'initial'/f'{s}_{k}.json').read_text()) for s,k in cells]
    first={(x['sample_id'],x['k']):x for x in initial_results}
    triggered=[(s,k) for s,k in cells if should_retrieve(first[s,k]['check1'])]
    save(run/'TRIGGERED.json',[{'sample_id':s,'k':k,'check':first[s,k]['check1']} for s,k in triggered])
    retriever=None
    pending_retrieval=[(s,k) for s,k in triggered if not (run/'retrieval'/f'{s}_{k}.json').exists()]
    for n,(s,k) in enumerate(pending_retrieval):
        path=run/'retrieval'/f'{s}_{k}.json'
        if retriever is None:
            try:
                from .retrieve import Retriever
            except ImportError:
                from retrieve import Retriever
            retriever=Retriever()
        result=retriever.search(s,first[s,k]['initial'],first[s,k]['check1']['gaps'])
        save(path,result)
        save(run/'STATUS.json',{'stage':'gap_retrieval_pending','done':n+1,'total':len(pending_retrieval),'status':'RUNNING'})
        print('gap_retrieval_pending',n+1,'/',len(pending_retrieval),flush=True)
    if retriever is not None:
        del retriever
        import torch
        torch.cuda.empty_cache()
    def finish(s,k):
        path=run/'final'/f'{s}_{k}.json'
        if path.exists():return json.loads(path.read_text())
        base=first[s,k]
        if not should_retrieve(base['check1']):
            result={**base,'final':base['initial'],'added':[],'check2':None,'adaptive':base['direct'],'selection_reason':'checker_keep','triggered':False}
        else:
            retrieved=json.loads((run/'retrieval'/f'{s}_{k}.json').read_text())
            try:
                answer=api.answer(s,retrieved['final'])
                checked=api.check(s,retrieved['final'],answer,base['check1']['gaps'],retrieved['added'])
            except Exception:
                answer=base['direct']; answer['answer_fallback']='initial_direct'
                checked=None
            selected,reason=select_answer(base['direct'],answer,checked,retrieved['added'])
            result={**base,'final':retrieved['final'],'added':retrieved['added'],'check2':checked,
                    'adaptive':selected,'revised':answer,'selection_reason':reason,'triggered':True}
        save(path,result);return result
    # Materialize the non-triggered arm locally: check1 already established
    # sufficiency and the direct answer is already cached.  Only genuinely
    # unfinished triggered cells should occupy API worker slots or progress.
    for s,k in cells:
        if not should_retrieve(first[s,k]['check1']):
            finish(s,k)
    pending=[(s,k) for s,k in triggered if not (run/'final'/f'{s}_{k}.json').exists()]
    save(run/'STATUS.json',{
        'stage':'recheck_and_answer_pending',
        'done':0,
        'total':len(pending),
        'already_final':len(cells)-len(pending),
        'status':'RUNNING'
    })
    if pending:
        stage('recheck_and_answer_pending',finish,pending)
    final=[json.loads((run/'final'/f'{s}_{k}.json').read_text()) for s,k in cells]
    final.sort(key=lambda x:(x['sample_id'],x['k']))
    save(run/'FROZEN_FINAL.json',final)
    # Gold/reference access begins here, after generation is frozen.
    source_data=Path(os.environ.get('BEST276_SOURCE_DATA',ROOT/'data/locomo10.json'))
    source_rows=json.loads(source_data.read_text())
    if dataset_name=='longmemeval':
        refs={str(row['question_id']):row for row in source_rows}
        template=('You are evaluating a LongMemEval answer. Return JSON only with exactly one field: '
                  '{{"label":"CORRECT"}} or {{"label":"WRONG"}}. A response is CORRECT when it '
                  'answers the question with the reference-equivalent fact(s), handles the requested scope '
                  'and time, and does not claim unsupported facts. For an abstention question whose id ends '
                  'with _abs, CORRECT means recognizing that the requested information is unavailable; do not '
                  'require the exact wording of the reference explanation.\n'
                  'QUESTION: {question}\nREFERENCE ANSWER: {golden_answer}\nGENERATED ANSWER: {generated_answer}')
    else:
        refs={f"{conv['sample_id']}__qa_{i}":q for conv in source_rows for i,q in enumerate(conv['qa'])}
        judge_path = Path(os.environ.get('BEST276_JUDGE_SCRIPT', ROOT/'source/judge_locomo_official.py'))
        if not judge_path.exists(): judge_path = ROOT/'scripts/judge_locomo_official.py'
        if not judge_path.exists(): judge_path = ROOT/'archive/276/final_bundle/snapshot/scripts/judge_locomo_official.py'
        if not judge_path.exists(): raise FileNotFoundError('judge_locomo_official.py')
        judge_ast=ast.parse(judge_path.read_text())
        try:
            template=next(ast.literal_eval(node.value) for node in judge_ast.body if isinstance(node,ast.Assign)
                          and any(isinstance(t,ast.Name) and t.id=='JUDGE_PROMPT' for t in node.targets))
        except StopIteration:
            from best_memory.evaluation.judge import JUDGE_PROMPT as template
    def judge(s,k,arm,pred):
        reference=refs[s]
        assert reference['question']==QUERIES[s]['question']
        if reference.get('answer') in (None,''):raise ValueError('missing_gold_answer')
        if dataset_name=='longmemeval':
            payload={'question':reference['question'],'reference_answer':str(reference['answer']),
                     'generated_answer':str(pred),'question_id':str(reference['question_id']),
                     'question_type':reference.get('question_type',''),
                     'abstention':str(reference['question_id']).endswith('_abs')}
            parsed,key=api.call('longmemeval_judge',template,payload,160)
            label=str(parsed.get('label','')).upper()
            if label not in {'CORRECT','WRONG'}: raise ValueError('judge_label')
            return {'sample_id':s,'k':k,'arm':arm,'grade':label=='CORRECT','pred_answer':pred,'judge_key':key}
        prompt=template.format(question=reference['question'],golden_answer=str(reference['answer']),generated_answer=pred)
        body={'model':api.client.llm_model,'messages':[{'role':'user','content':prompt}],'temperature':0,'max_tokens':160}
        key=sha([body,api.client.llm_base]);path=H/'judge_cache'/f'{key}.json'
        if path.exists():v=json.loads(path.read_text())
        else:
            base=api.client.llm_base.rstrip('/')
            endpoint=base+('/chat/completions' if base.endswith('/v1') else '/v1/chat/completions')
            save(H/'judge_requests'/f'{key}.json',{'body':body,'provider':base})
            for attempt in range(4):
                try:
                    request=urllib.request.Request(endpoint,data=json.dumps(body).encode(),headers={
                        'Content-Type':'application/json','Authorization':'Bearer '+api.client.llm_key},method='POST')
                    with urllib.request.urlopen(request,timeout=180) as response:envelope=json.loads(response.read())
                    save(H/'judge_responses'/f'{key}.json',envelope)
                    text=envelope['choices'][0]['message']['content']
                    parsed=json.loads(text.strip().removeprefix('```json').removesuffix('```').strip())
                    label=parsed['label'].upper()
                    if label not in ['CORRECT','WRONG']:raise ValueError('judge_label')
                    v={'label':label,'raw':text,'usage':envelope.get('usage'),'request_key':key}
                    save(path,v);break
                except Exception as exc:
                    diagnostic={'attempt':attempt+1,'error_type':type(exc).__name__,
                                'http_status':getattr(exc,'code',None),'request_key':key}
                    save(H/'judge_errors'/f'{key}.json',diagnostic)
                    if attempt==3:
                        raise RuntimeError('judge_failure:'+type(exc).__name__+':http='+str(getattr(exc,'code',None))) from None
                    time.sleep(2**attempt)
        return {'sample_id':s,'k':k,'arm':arm,'grade':v['label']=='CORRECT','pred_answer':pred,'judge_key':key}
    jobs=[(x['sample_id'],x['k'],arm,x[field]['pred_answer']) for x in final for arm,field in [('DIRECT','direct'),('CHECKED','adaptive')]]
    unique_jobs={(s,pred):(s,k,arm,pred) for s,k,arm,pred in jobs}
    # Finish the whole judge queue despite isolated failures, then retry only
    # failed cells. Successful results are retained and never re-requested.
    judge_items=list(unique_jobs.values())
    for recovery_round in range(3):
        try:
            stage('official_prompt_judge',judge,judge_items)
            break
        except RuntimeError:
            if recovery_round==2:raise
            errors=json.loads((run/'official_prompt_judge_errors.json').read_text())
            judge_items=[tuple(x['cell']) for x in errors]
            print('judge recovery round',recovery_round+1,'pending',len(judge_items),flush=True)
    unique_judgments=[judge(*item) for item in unique_jobs.values()]
    judged={(x['sample_id'],x['pred_answer']):x for x in unique_judgments}
    judgments=[{**judged[s,pred],'k':k,'arm':arm} for s,k,arm,pred in jobs]
    save(run/'JUDGMENTS.json',judgments)
    request_keys={part['request_key'] for x in final for part in [x['direct'],x.get('revised'),x['adaptive'],x['check1'],x['check2']]
                  if part and part.get('request_key')}
    cost=defaultdict(float)
    for key in request_keys:
        cache_path=H/'api_cache'/f'{key}.json'
        # A resumed LongMemEval run may contain a validated cached response
        # from an older answer root without the corresponding local envelope.
        # Keep metrics generation resumable and report the missing cache rather
        # than failing after FROZEN_FINAL has already been produced.
        if not cache_path.exists():
            cost['missing_cached_request']+=1
            continue
        item=json.loads(cache_path.read_text())
        cost[item['kind']+'_unique_requests']+=1
        cost['summed_request_seconds']+=item['seconds']
    cost['unique_judge_requests']=len({x['judge_key'] for x in judgments})
    save(run/'COST.json',{'counts':dict(cost),'note':'Semantic request counts, excludes hidden HTTP retries; summed latency is not wall time; reused identical answers/judgments counted once.'})
    metrics=defaultdict(lambda:defaultdict(int))
    gold_path=Path(os.environ.get('BEST276_GOLD_RAW', B/'GOLD_RAW_FIXED_EVAL.jsonl'))
    gold={x['sample_id']:set(x.get('gold_raw_ids') or []) for x in rows(gold_path)} if gold_path.exists() else {}
    grades={(x['sample_id'],x['k'],x['arm']):x['grade'] for x in judgments}
    for x in final:
        s,k=x['sample_id'],x['k'];m=metrics[str(k)];g=gold.get(s,set())
        before=set(x['initial']);after=set(x['final'])
        m['n']+=1;m['direct_correct']+=grades[s,k,'DIRECT'];m['checked_correct']+=grades[s,k,'CHECKED']
        m['answer_gain']+=grades[s,k,'CHECKED'] and not grades[s,k,'DIRECT']
        m['answer_harm']+=grades[s,k,'DIRECT'] and not grades[s,k,'CHECKED']
        m['triggered']+=x['triggered'];m['total_added_raw']+=len(x['added'])
        m['final_abstained']+=x['adaptive']['pred_answer']=='INSUFFICIENT_EVIDENCE'
        closed=x['check1']['decision']=='KEEP'
        m['initial_closed']+=closed;m['initial_closed_but_direct_wrong']+=closed and not grades[s,k,'DIRECT']
        m['initial_open_but_direct_correct']+=not closed and grades[s,k,'DIRECT']
        m['checker_validation_issues']+=bool(x['check1']['validation_issues'])
        if x['check2']:
            m['second_closed']+=x['check2']['decision']=='USE_REVISED'
            m['second_closed_but_answer_wrong']+=x['check2']['decision']=='USE_REVISED' and not grades[s,k,'CHECKED']
        m['revision_selected']+=x.get('selection_reason')=='verified_revision'
        m['direct_protected']+=x['triggered'] and x.get('selection_reason')!='verified_revision'
        if g:
            m['gold_eval_n']+=1;m['initial_exact']+=g<=before;m['final_exact']+=g<=after
            m['initial_hit']+=bool(g&before);m['final_hit']+=bool(g&after)
            m['gold_promoted']+=len(g&(after-before));m['gold_displaced']+=len(g&(before-after))
            m['initial_closed_but_gold_incomplete_proxy']+=closed and not g<=before
            m['initial_open_but_gold_complete_proxy']+=not closed and g<=before
            m['unconditional_k_plus_5_exact_control']+=g<=set(RANKS[s]['ranking'][:k+5])
    if dataset_name=='longmemeval':
        # LongMemEval labels evidence at session level.  The method may rank
        # turns or aggregated sessions, so map every raw result through the
        # preserved source_session_id before scoring.  This metric is kept
        # separate from the LoCoMo-specific gold-raw diagnostics above.
        retrieval=defaultdict(lambda:defaultdict(int))
        def unique_sessions(sample_id, raw_ids):
            out=[];seen=set()
            for rid in raw_ids:
                sid=RAW.get(rid,{}).get('source_session_id')
                if sid and sid not in seen:
                    seen.add(sid);out.append(str(sid))
            return out
        for k in (10,20):
            rows_k=[x for x in final if x['k']==k]
            for x in rows_k:
                ref=refs[x['sample_id']];gold_sessions={str(v) for v in (ref.get('answer_session_ids') or [])}
                ranked=unique_sessions(x['sample_id'],RANKS[x['sample_id']]['ranking'])
                top=ranked[:k]
                hit=bool(gold_sessions.intersection(top)); complete=gold_sessions.issubset(set(top))
                rel=[1 if sid in gold_sessions else 0 for sid in ranked]
                ideal=sum(1.0/(__import__('math').log2(i+2)) for i in range(min(k,len(gold_sessions))))
                actual=sum(v/__import__('math').log2(i+2) for i,v in enumerate(rel[:k]))
                retrieval[str(k)]['n']+=1
                retrieval[str(k)]['recall_any']+=hit
                retrieval[str(k)]['recall_all']+=complete
                retrieval[str(k)]['ndcg_sum']+=actual/ideal if ideal else 0.0
                retrieval[str(k)]['abstention_n']+=str(ref['question_id']).endswith('_abs')
                qtype=str(ref.get('question_type',''))
                types=retrieval[str(k)].setdefault('question_types',{})
                types[qtype]=types.get(qtype,0)+1
            retrieval[str(k)]['recall_any_rate']=retrieval[str(k)]['recall_any']/max(1,retrieval[str(k)]['n'])
            retrieval[str(k)]['recall_all_rate']=retrieval[str(k)]['recall_all']/max(1,retrieval[str(k)]['n'])
            retrieval[str(k)]['ndcg']=retrieval[str(k)].pop('ndcg_sum')/max(1,retrieval[str(k)]['n'])
        save(run/'LONGMEMEVAL_RETRIEVAL_METRICS.json',retrieval)
        # Official QA evaluation consumes one hypothesis per question.  The
        # primary checked arm uses the larger K=20 packet; the full DIRECT /
        # CHECKED table remains in JUDGMENTS.json.
        by_q={x['sample_id']:x for x in final if x['k']==20}
        with (run/'longmemeval_hypotheses.jsonl').open('w',encoding='utf8') as handle:
            for qid in sorted(by_q):
                item=by_q[qid]
                handle.write(json.dumps({'question_id':qid,'hypothesis':item['adaptive']['pred_answer']},ensure_ascii=False)+'\n')
        metrics['longmemeval']['retrieval_metrics_file']='LONGMEMEVAL_RETRIEVAL_METRICS.json'
        metrics['longmemeval']['hypotheses_file']='longmemeval_hypotheses.jsonl'
    save(run/'METRICS.json',metrics)
    report=['# Top10 / Top20 检查与补检问答实验','',
            '| 初始K | N | 直接回答正确 | 检查补检后正确 | Gain/Harm | 补检触发 | 最终拒答 |',
            '|---:|---:|---:|---:|---:|---:|---:|']
    # The LongMemEval retrieval block is intentionally a separate metric
    # namespace and does not contain answer-arm counters.  Keep it in
    # METRICS.json, but only render rows that belong to the answer table here.
    for k,m in metrics.items():
        if 'n' not in m or 'direct_correct' not in m:
            continue
        report.append(f"| {k} | {m['n']} | {m['direct_correct']} | {m['checked_correct']} | {m['answer_gain']}/{m['answer_harm']} | {m['triggered']} | {m['final_abstained']} |")
    judge_note=('使用仓库原版 LoCoMo JUDGE_PROMPT。' if dataset_name=='locomo'
                else '使用 LongMemEval reference-equivalence JSON Judge；该结果不属于历史 276 复现口径。')
    report+=[ '',judge_note,
             '最终证据预算最多K+5；与直接回答对照的差异包含额外证据和检查拒答。',
             'Gold完整性仅为检查器诊断代理，不能等同实际可回答性；同时报告闭合但答错和触发但原本答对。',
             '全量/试跑范围见PROTOCOL.json，失败不能算作有效完成。']
    (run/'REPORT.md').write_text('\n'.join(report)+'\n')
    save(run/'STATUS.json',{'status':'COMPLETE','questions':len(ids),'answer_arms':len(judgments)})
    print(json.dumps(metrics),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        import sys
        limit=int(sys.argv[sys.argv.index('--limit')+1]) if '--limit' in sys.argv else 0
        status_path=H/(f'smoke_{limit}' if limit else 'full')/'STATUS.json'
        previous=json.loads(status_path.read_text()) if status_path.exists() else {}
        save(status_path,{**previous,'status':'FAILED','error_type':type(exc).__name__})
        print('PIPELINE_FAILED',type(exc).__name__,flush=True)
        raise SystemExit(1)
