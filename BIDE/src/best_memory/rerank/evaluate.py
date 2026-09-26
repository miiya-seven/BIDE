"""Evaluate only completed, frozen full-domain rankings."""
import json, os
from collections import defaultdict
from pathlib import Path

H=Path(__file__).resolve().parent
B=H.parent
def rows(p): return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def save(name,value): (H/name).write_text(json.dumps(value,ensure_ascii=False,indent=2))
rankings=json.loads((H/'RANKINGS.json').read_text())
assert len(rankings)==1540 and len({x['sample_id'] for x in rankings})==1540
run=Path(os.environ.get('BEST276_RUN_ROOT', B/'../../../../runs/default'))
gold_path=Path(os.environ.get('BEST276_GOLD_RAW', run/'evaluation/GOLD_RAW_FIXED_EVAL.jsonl'))
frame_path=Path(os.environ.get('BEST276_QUERY_FRAMES', run/'query/QUERY_FRAMES_V2_MERGED_1540.jsonl'))
gold={x['sample_id']:set(x.get('gold_raw_ids') or []) for x in rows(gold_path)}
frames={x['sample_id']:x for x in rows(frame_path)}
metrics=defaultdict(lambda:defaultdict(int))
paired=defaultdict(lambda:defaultdict(int))
failures=[]
empty=[]
for x in rankings:
    s=x['sample_id'];g=gold[s]
    assert len(x['ranking'])==128 and set(x['ranking'])==set(x['original'])
    if not g:
        empty.append(s)
        continue
    strata=['all','single_gold' if len(g)==1 else 'multi_gold',
            'shape:'+frames[s]['evidence_shape'],'conversation:'+s.split('__')[0]]
    for k in [1,5,10,20,50,128]:
        before=set(x['original'][:k]);after=set(x['ranking'][:k])
        for group in strata:
            for arm,p in [('ORIGINAL',before),('QWEN_STRUCTURED',after)]:
                m=metrics[f'{arm}/{k}/{group}']
                m['n']+=1;m['exact']+=g<=p;m['hit']+=bool(g&p)
                m['zero']+=not bool(g&p);m['raw']+=len(g&p);m['total_gold']+=len(g)
                m['gold_count_exceeds_k']+=len(g)>k
            z=paired[f'{k}/{group}']
            z['n']+=1;z['exact_gain']+=g<=after and not g<=before
            z['exact_harm']+=g<=before and not g<=after
            z['hit_gain']+=bool(g&after) and not bool(g&before)
            z['hit_harm']+=bool(g&before) and not bool(g&after)
            z['gold_promoted']+=len(g&(after-before));z['gold_displaced']+=len(g&(before-after))
        if k==10 and not g<=after:
            failures.append({'sample_id':s,'question':frames[s]['question'],'evidence_shape':frames[s]['evidence_shape'],
                             'gold_count':len(g),'hit':bool(g&after),
                             'missing':[{'raw_id':r,'original_rank':x['original'].index(r)+1 if r in x['original'] else None,
                                         'qwen_rank':x['ranking'].index(r)+1 if r in x['ranking'] else None}
                                        for r in sorted(g-after)]})
assert metrics['ORIGINAL/128/all']==metrics['QWEN_STRUCTURED/128/all']
save('EVALUATION.json',metrics);save('PAIRED.json',paired);save('TOP10_FAILURES.json',failures)
save('EXCLUDED_EMPTY_GOLD.json',empty)
lines=['# 全量六族 Candidate128 + Qwen3-Reranker-8B', '',
       f"全量1540题，非空Gold评估{metrics['ORIGINAL/10/all']['n']}题，空Gold排除{len(empty)}题。",
       '输入为既有 query_text / retrieval_text，沿用272结构化精排；完整重排128条。',
       '批次按token长度分组、batch64、bf16，可能存在相对于272的小幅数值差异。',
       'Exact = 全部标注Gold覆盖；Hit = 至少一条Gold覆盖；Raw = 所有Gold条目覆盖。均不是最终QA正确率。', '',
       '| K | 排序 | Exact | Hit | Zero | Raw |', '|---:|---|---:|---:|---:|---:|']
for k in [1,5,10,20,50,128]:
    for arm in ['ORIGINAL','QWEN_STRUCTURED']:
        m=metrics[f'{arm}/{k}/all']
        lines.append(f"| {k} | {arm} | {m['exact']}/{m['n']} ({m['exact']/m['n']:.2%}) | {m['hit']}/{m['n']} ({m['hit']/m['n']:.2%}) | {m['zero']} | {m['raw']}/{m['total_gold']} ({m['raw']/m['total_gold']:.2%}) |")
lines+=['','## Top10分层','','| 分层 | N | 原排序Exact | 精排Exact | 原排序Hit | 精排Hit |','|---|---:|---:|---:|---:|---:|']
for group in sorted({key.split('/',2)[2] for key in metrics if '/10/' in key and 'conversation:' not in key}):
    a=metrics['ORIGINAL/10/'+group];b=metrics['QWEN_STRUCTURED/10/'+group]
    lines.append(f"| {group} | {a['n']} | {a['exact']} | {b['exact']} | {a['hit']} | {b['hit']} |")
lines+=['','## Top10配对损益','',json.dumps(paired['10/all'],ensure_ascii=False), '',
        '候选池完全相同；池外Gold不能由本精排找回。未执行充分性审核、补检或最终QA。',
        '每题完整排序、逐题失败、分对话/证据形状统计、输入哈希与分数均保存在本目录。']
save('TOP10_SUMMARY.json',{'original':metrics['ORIGINAL/10/all'],'qwen':metrics['QWEN_STRUCTURED/10/all'],
                          'paired':paired['10/all'],'empty_gold_excluded':empty})
(H/'REPORT.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
