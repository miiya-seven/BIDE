#!/usr/bin/env python3
import importlib.util, os, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
checks=[]
def check(name, ok, detail): checks.append({'name':name,'ok':bool(ok),'detail':detail})

def main():
    check('python',sys.version_info>=(3,11),sys.version.split()[0])
    check('dataset',(ROOT/'data/locomo10.json').exists(),str(ROOT/'data/locomo10.json'))
    for mod in ['numpy','torch','transformers','yaml']:
        check('python:'+mod,importlib.util.find_spec(mod) is not None,mod)
    defaults={'QWEN_RERANKER_MODEL':'','BGE_M3_MODEL':'','SPLADE_MODEL':''}
    for key,default in defaults.items():
        val=os.environ.get(key,default); check('model:'+key,Path(val).exists(),val)
    cfg=ROOT/'configs/main_experiment.yaml'
    check('answer_model_config',cfg.exists() and bool(os.environ.get('ANSWER_MODEL') or os.environ.get('LLM_BASE_URL') or 'generation:' in cfg.read_text()), 'configs/main_experiment.yaml or ANSWER_MODEL/LLM_BASE_URL')
    out=ROOT/'runs'; check('output_writable',out.exists() and os.access(out,os.W_OK),str(out))
    for x in checks: print(('OK  ' if x['ok'] else 'FAIL'),x['name'],x['detail'])
    return 0 if all(x['ok'] for x in checks) else 1
if __name__=='__main__': raise SystemExit(main())
