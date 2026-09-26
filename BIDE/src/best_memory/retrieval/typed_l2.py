#!/usr/bin/env python3
"""Typed L2 retrieval from the public corpus and complete query contract."""
import json, os, re, time, urllib.request
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
RUN=Path(os.environ.get("BEST276_RUN_ROOT",HERE/"../../../../runs/default"))
INPUT=Path(os.environ.get("TYPED_L2_INPUT_DIR",RUN/"retrieval"))
QUERY=Path(os.environ.get("TYPED_L2_QUERY",RUN/"query/V41_QUERIES_REBUILT.jsonl"))
OUTPUT=Path(os.environ.get("TYPED_L2_OUTPUT_DIR",RUN/"retrieval"));OUTPUT.mkdir(parents=True,exist_ok=True)
URL=os.environ.get("EMBEDDING_URL",os.environ.get("EMBEDDING_URL", ""));MODEL="BAAI/bge-m3"

def rows(path):return [json.loads(x) for x in path.open(encoding="utf8") if x.strip()]
def words(value):return set(re.findall(r"[a-z0-9]+",str(value or "").lower()))
def embed(texts):
 out=[]
 for start in range(0,len(texts),64):
  body=json.dumps({"model":MODEL,"input":texts[start:start+64]}).encode();req=urllib.request.Request(URL,data=body,headers={"Content-Type":"application/json","Authorization":"Bearer dummy"})
  with urllib.request.urlopen(req,timeout=300) as response:data=sorted(json.loads(response.read())["data"],key=lambda x:x["index"])
  out.extend(x["embedding"] for x in data)
 matrix=np.asarray(out,dtype=np.float32);matrix/=np.linalg.norm(matrix,axis=1,keepdims=True).clip(min=1e-8);return matrix

assertions=rows(INPUT/"V41_ASSERTIONS.jsonl");queries=rows(QUERY);started=time.time();by_conversation=defaultdict(list)
for index,item in enumerate(assertions):by_conversation[item["conversation_id"]].append(index)
if not assertions:
 output=[{"sample_id":query["sample_id"],"rankings":{"typed_semantic":[],"typed_owner":[],"typed_value":[]},"gold_visible":False} for query in queries]
 with (OUTPUT/"TYPED_L2_LANES.jsonl").open("w",encoding="utf8") as handle:
  for item in output:handle.write(json.dumps(item,ensure_ascii=False)+"\n")
 receipt={"questions":len(queries),"assertions":0,"lanes":["typed_semantic","typed_owner","typed_value"],"model":MODEL,"backend":"empty_l2","elapsed_seconds":time.time()-started,"legacy_artifacts_used":False,"gold_used":False}
 (OUTPUT/"TYPED_L2_RECEIPT.json").write_text(json.dumps(receipt,indent=2)+"\n");print(json.dumps(receipt,indent=2));raise SystemExit(0)
documents=[x.get("retrieval_text","") for x in assertions]
values=[" | ".join(str(x.get(k) or "") for k in ("subject","value","event","time")) for x in assertions]
predicate_queries=[];value_queries=[]
for query in queries:
 plan=query.get("query_v41") or {}
 predicate_queries.append(" | ".join([query["question"],*plan.get("target_entities",[]),*plan.get("relation_queries",[]),*plan.get("event_queries",[]),*plan.get("temporal_queries",[])]))
 value_queries.append(" | ".join([query["question"],*plan.get("value_queries",[]),*plan.get("target_entities",[])]))
doc_matrix=embed(documents);value_matrix=embed(values);predicate_matrix=embed(predicate_queries);query_value_matrix=embed(value_queries);output=[]
for qi,query in enumerate(queries):
 indices=by_conversation[query["conversation_id"]];full=doc_matrix[indices]@predicate_matrix[qi];value=value_matrix[indices]@query_value_matrix[qi];owner_terms=words(" ".join((query.get("query_v41") or {}).get("target_entities",[])));per_raw={}
 for local,index in enumerate(indices):
  item=assertions[index];rid=item["raw_id"];candidate=(float(full[local]),float(value[local]),bool(owner_terms&words(item.get("subject"))) if owner_terms else True)
  if rid not in per_raw or max(candidate[:2])>max(per_raw[rid][:2]):per_raw[rid]=candidate
 semantic=sorted(per_raw,key=lambda r:(-per_raw[r][0],r))[:128];typed_value=sorted(per_raw,key=lambda r:(-per_raw[r][1],r))[:128];owner=sorted((r for r in per_raw if per_raw[r][2]),key=lambda r:(-per_raw[r][0],r));owner=(owner+[r for r in semantic if r not in set(owner)])[:128]
 output.append({"sample_id":query["sample_id"],"rankings":{"typed_semantic":semantic,"typed_owner":owner,"typed_value":typed_value},"gold_visible":False})
with (OUTPUT/"TYPED_L2_LANES.jsonl").open("w",encoding="utf8") as handle:
 for item in output:handle.write(json.dumps(item,ensure_ascii=False)+"\n")
receipt={"questions":len(queries),"assertions":len(assertions),"lanes":["typed_semantic","typed_owner","typed_value"],"model":MODEL,"elapsed_seconds":time.time()-started,"legacy_artifacts_used":False,"gold_used":False}
(OUTPUT/"TYPED_L2_RECEIPT.json").write_text(json.dumps(receipt,indent=2)+"\n");print(json.dumps(receipt,indent=2))
