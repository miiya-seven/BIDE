#!/usr/bin/env python3
import hashlib,json
from collections import defaultdict
from pathlib import Path
H=Path(__file__).resolve().parent
def rows(p): return [json.loads(x) for x in p.open() if x.strip()]
def canon(x): return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(",",":"))
def keystr(*x): return "|".join(str(v or "UNKNOWN").strip().casefold() for v in x)
def atom(x): return canon(x) if isinstance(x,(dict,list)) else str(x)
mem=rows(H/"MEMORY_OUTPUT/L1_L2_MEMORY.jsonl")
entity=defaultdict(lambda:{"assertion_ids":[],"raw_ids":set(),"by_relation":defaultdict(list)})
occ=defaultdict(lambda:{"assertion_ids":[],"raw_ids":set(),"participants":set(),"times":set(),"locations":set(),"modalities":set()})
coll=defaultdict(lambda:{"members":defaultdict(lambda:{"assertion_ids":[],"raw_ids":set(),"occurrence_ids":set()}),"frontier_status":"OPEN"})
comp=defaultdict(lambda:{"alternatives":defaultdict(lambda:{"assertion_ids":[],"raw_ids":set(),"modalities":set(),"polarities":set()})})
temporal=defaultdict(list)
for bundle in mem:
  rid=bundle["center_raw_id"]
  for a in bundle.get("l2_direct",[]):
    aid=a["assertion_id"]; rel=a["relation_family"]; subj=a["subject"]; val=a["answer_value"].get("value"); typ=a["answer_value"].get("type"); roles=a.get("roles",{}); scope=a.get("scope",{}); oid=a.get("occurrence_local_id")
    entities={subj}|{str(v) for v in roles.values() if v}
    for e in entities:
      z=entity[e]; z["assertion_ids"].append(aid); z["raw_ids"].add(rid); z["by_relation"][rel].append(aid)
    if oid:
      ok=f"{rid}::{oid}"; z=occ[ok]; z["assertion_ids"].append(aid); z["raw_ids"].add(rid); z["modalities"].add(a["modality"])
      for rk in ("participant","agent","observer","evaluated_subject"):
        if roles.get(rk): z["participants"].add(str(roles[rk]))
      if scope.get("time_expression"): z["times"].add(atom(scope["time_expression"]))
      if scope.get("location"): z["locations"].add(atom(scope["location"]))
    ck=keystr(subj,rel,typ); member=keystr(val,oid)
    cz=coll[ck]["members"][member]; cz["assertion_ids"].append(aid); cz["raw_ids"].add(rid)
    if oid: cz["occurrence_ids"].add(f"{rid}::{oid}")
    pk=keystr(subj,rel,oid); alt=keystr(val)
    pz=comp[pk]["alternatives"][alt]; pz["assertion_ids"].append(aid); pz["raw_ids"].add(rid); pz["modalities"].add(a["modality"]); pz["polarities"].add(a["polarity"])
    if scope.get("time_expression"):
      temporal[keystr(subj,rel)].append({"assertion_id":aid,"raw_id":rid,"occurrence_id":f"{rid}::{oid}" if oid else None,"time_expression":scope["time_expression"],"time_owner":scope.get("time_owner"),"ordering_status":"UNORDERED"})
def clean(x):
  if isinstance(x,set): return sorted(x)
  if isinstance(x,defaultdict) or isinstance(x,dict): return {k:clean(v) for k,v in x.items()}
  if isinstance(x,list): return [clean(v) for v in x]
  return x
out=H/"L3_OUTPUT"; out.mkdir(exist_ok=True)
files={"ENTITY_VIEWS.jsonl":[{"entity":k,**clean(v)} for k,v in sorted(entity.items())],"OCCURRENCE_VIEWS.jsonl":[{"occurrence_id":k,**clean(v)} for k,v in sorted(occ.items())],"COLLECTION_VIEWS.jsonl":[{"collection_key":k,**clean(v)} for k,v in sorted(coll.items())],"COMPETITION_VIEWS.jsonl":[{"competition_key":k,**clean(v)} for k,v in sorted(comp.items()) if len(v["alternatives"])>1],"TEMPORAL_VIEWS.jsonl":[{"timeline_key":k,"entries":clean(v)} for k,v in sorted(temporal.items())]}
hashes={}
for name,data in files.items():
  p=out/name; p.write_text("".join(canon(x)+"\n" for x in data)); hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
rec={"status":"COMPLETE_DETERMINISTIC","gold_visible":False,"counts":{k:len(v) for k,v in files.items()},"input_sha256":hashlib.sha256((H/"MEMORY_OUTPUT/L1_L2_MEMORY.jsonl").read_bytes()).hexdigest(),"outputs_sha256":hashes}
(out/"RECEIPT.json").write_text(json.dumps(rec,indent=2)+"\n"); print(json.dumps(rec,indent=2))
