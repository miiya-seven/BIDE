#!/usr/bin/env python3
import json, os
from collections import Counter
from pathlib import Path

H = Path(__file__).resolve().parent
B = H.parent

def rows(path):
    return [json.loads(line) for line in path.open(encoding='utf8') if line.strip()]

run_root=Path(os.environ.get('BEST276_RUN_ROOT', B/'../../runs/default'))
manifest_path=run_root/'memory'/'V2_STATUS_MANIFEST.jsonl'
manifest = rows(manifest_path) if manifest_path.exists() else []
base = {}
base_paths = [Path(x) for x in os.environ.get('BEST276_MEMORY_INPUTS','').split(os.pathsep) if x]
if not base_paths:
    base_paths = [run_root/'memory/L1_L2_MEMORY.jsonl', run_root/'memory/MEMORY_OUTPUT/L1_L2_MEMORY.jsonl', run_root/'memory/MEMORY_OUTPUT/DEBUG_0000.jsonl']
for path in base_paths:
    if not path.exists(): continue
    for item in rows(path):
        base[item['center_raw_id']] = item

# Session-level run_generation stores one OpenAI-style envelope per request
# under MEMORY_OUTPUT/wire/.  Accept that canonical wire layout directly when
# the legacy aggregate JSONL has not yet been materialized.
if not base:
    wire_dir = run_root / 'memory' / 'MEMORY_OUTPUT' / 'wire'
    for path in sorted(wire_dir.glob('*.json')) if wire_dir.exists() else []:
        try:
            envelope = json.loads(path.read_text(encoding='utf8'))
            content = envelope.get('choices', [{}])[0].get('message', {}).get('content')
            item = json.loads(content) if isinstance(content, str) else content
            if isinstance(item, dict) and item.get('center_raw_id'):
                base[item['center_raw_id']] = item
        except Exception:
            # A malformed wire is handled by the normal missing-row checks;
            # do not make one bad request prevent other rows from merging.
            continue

parsed = {}
for path in [run_root/'memory/CONTEXTUAL_V2.jsonl', run_root/'memory/CONTEXTUAL_V41.jsonl']:
    if not path.exists(): continue
    for item in rows(path):
        parsed[item['raw_id']] = item
retry_path=H / 'V41_GPT54MINI_UNCERTAIN_RETRY.jsonl'
if os.environ.get('BEST276_IMPORT_LEGACY_CONTEXT','0')=='1' and retry_path.exists():
    for item in rows(retry_path):
        if item['v41_status'] == 'VALIDATED':
            parsed[item['raw_id']] = item
if not manifest:
    manifest=[{'raw_id':rid,'status':'CONTEXT_REPARSE_REQUIRED'} for rid in base]

output = []
for entry in manifest:
    rid = entry['raw_id']
    if entry['status'] in {'SELF_CONTAINED_REUSED', 'SELF_CONTAINED_RAW', 'SELF_CONTAINED_LLM'} and parsed.get(rid,{}).get('status')!='VALIDATED':
        memory = base[rid]
        output.append({
            'raw_id': rid, 'sidecar_status': 'BASE_REUSED',
            'manifest_status': entry['status'], 'contextual_v41': None,
            'base_l1': memory.get('l1'), 'base_l2_direct': memory.get('l2_direct', []),
            'schema_version': 'contextual-memory-v4.1-sidecar', 'gold_visible': False,
        })
    else:
        item = parsed.get(rid, {})
        valid = item.get('status', item.get('v41_status', item.get('v2_status'))) == 'VALIDATED'
        output.append({
            'raw_id': rid,
            'sidecar_status': 'V41_VALIDATED' if valid else 'V41_UNCERTAIN_FALLBACK',
            'manifest_status': entry['status'],
            'contextual_v41': item.get('contextual_v4', item.get('contextual_v2')) if valid else None,
            'base_l1': base[rid].get('l1'),
            # Context resolution supplements direct facts; it never erases the
            # validated L2 extraction.
            'base_l2_direct': base[rid].get('l2_direct', []),
            'validation_errors': item.get('validation_errors', []) if not valid else [],
            'schema_version': 'contextual-memory-v4.1-sidecar', 'gold_visible': False,
        })

# The historical full run has 5,882 rows, but adapters may intentionally use a
# smaller dataset.  Validate against the rows actually supplied by this run.
expected_ids=set(base)
assert len(output) == len(expected_ids)
assert {x['raw_id'] for x in output} == expected_ids
run_root = Path(os.environ.get('BEST276_RUN_ROOT', B/'../../runs/default'))
sidecar_out = run_root / 'memory' / 'CONTEXTUAL_MEMORY_V41_SIDECAR.jsonl'
sidecar_out.parent.mkdir(parents=True, exist_ok=True)
with sidecar_out.open('w', encoding='utf8') as handle:
    for item in output:
        handle.write(json.dumps(item, ensure_ascii=False) + '\n')

status = Counter(x['sidecar_status'] for x in output)
acts = Counter()
assertions = 0
for item in output:
    contextual = item.get('contextual_v41') or {}
    for act in contextual.get('speech_acts', []):
        acts[act['act']] += 1
        assertions += act['act'] == 'ASSERTION'
receipt = {
    'rows': len(output), 'unique_raw_ids': len({x['raw_id'] for x in output}),
    'status': dict(status), 'v41_speech_acts': dict(acts),
    'v41_assertions': assertions, 'gold_used': False,
}
(run_root / 'memory' / 'CONTEXTUAL_MEMORY_V41_SIDECAR_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt))
