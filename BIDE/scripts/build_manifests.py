#!/usr/bin/env python3
from pathlib import Path
import hashlib, json, os

ROOT=Path(__file__).resolve().parents[1]

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def files(*roots):
    out=[]
    for root in roots:
        p=ROOT/root
        if p.is_file(): out.append(p)
        elif p.exists(): out.extend(x for x in p.rglob('*') if x.is_file() and '__pycache__' not in x.parts)
    return sorted(out)

def main():
    tracked=files('src/best_memory','src/ab_memory','prompts','schemas','configs')
    hashes={str(p.relative_to(ROOT)):digest(p) for p in tracked}
    (ROOT/'manifests/source_hashes.json').write_text(json.dumps({'algorithm':'sha256','files':hashes},ensure_ascii=False,indent=2))
    models={
      'qwen_reranker':os.environ.get('QWEN_RERANKER_MODEL',''),
      'bge_m3':os.environ.get('BGE_M3_MODEL',''),
      'splade':os.environ.get('SPLADE_MODEL',''),
      'answer_model':os.environ.get('ANSWER_MODEL','')}
    (ROOT/'manifests/model_manifest.json').write_text(json.dumps({'models':models,'note':'paths are environment supplied; weights are not committed'},ensure_ascii=False,indent=2))
    print(f'hashed {len(hashes)} files')
if __name__=='__main__': main()
