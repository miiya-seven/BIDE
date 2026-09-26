#!/usr/bin/env python3
import json
from collections import Counter
from pathlib import Path
H=Path(__file__).resolve().parent
def rows(p): return [json.loads(x) for x in p.open() if x.strip()]
q=rows(H/"QUERY_OUTPUT/QUERY_KEYS.jsonl"); m=rows(H/"MEMORY_OUTPUT/L1_L2_MEMORY.jsonl")
units=[u for b in m for c in b["l1"]["clause_units"] for u in c["units"]]; aa=[a for b in m for a in b["l2_direct"]]
report={"status":"MECHANICAL_AUDIT_COMPLETE","gold_visible":False,"query":{"count":len(q),"relations":dict(Counter(x["relation_family"] for x in q)),"answer_types":dict(Counter(x["answer_type"] for x in q)),"shapes":dict(Counter(x["evidence_shape"] for x in q)),"modality_constraints":dict(Counter(x["modality_constraint"] for x in q))},"memory":{"raws":len(m),"atomic_units":len(units),"assertions":len(aa),"unit_status":dict(Counter(x["status"] for x in units)),"relations":dict(Counter(x["relation_family"] for x in aa)),"modalities":dict(Counter(x["modality"] for x in aa)),"assertions_with_occurrence":sum(bool(x.get("occurrence_local_id")) for x in aa),"assertions_with_time_owner":sum(bool(x.get("scope",{}).get("time_owner")) for x in aa),"assertions_with_observer":sum(bool(x.get("roles",{}).get("observer")) for x in aa),"assertions_with_evaluated_subject":sum(bool(x.get("roles",{}).get("evaluated_subject")) for x in aa),"unresolved_units":sum(x["status"]=="UNRESOLVED" for x in units),"nonpropositional_units":sum(x["status"]=="NON_PROPOSITIONAL" for x in units)}}
(H/"MECHANICAL_AUDIT.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n"); print(json.dumps(report,ensure_ascii=False,indent=2))
