#!/usr/bin/env python3
"""Compile one gold-free evidence interface for V41_VALIDATED and BASE_REUSED Raw."""
import json,re,os
from collections import defaultdict,Counter
from pathlib import Path
H=Path(__file__).resolve().parent
INPUT=Path(os.environ.get("EVIDENCE_INPUT_DIR",H));PREFIX=os.environ.get("EVIDENCE_PREFIX","V41")
OUTPUT=Path(os.environ.get("EVIDENCE_OUTPUT_DIR",H));OUTPUT.mkdir(parents=True,exist_ok=True)
OUTPUT_NAME=os.environ.get("EVIDENCE_OUTPUT_NAME","RAW_EVIDENCE_INTERFACE")
SIDE=Path(os.environ.get("EVIDENCE_SIDECAR",Path(os.environ.get('BEST276_RUN_ROOT',H/'../../runs/default'))/'memory/CONTEXTUAL_MEMORY_V41_SIDECAR.jsonl'))
def rows(p):return [json.loads(x) for x in p.open(encoding="utf8") if x.strip()]
def norm(x):return " ".join(str(x or "").lower().split())
STOP=set("a an the and or of to in on at for from with by is are was were be been being do did does have has had what when where who why how which that this these those his her their its he she they it about according".split())
def toks(x):return {w for w in re.findall(r"[a-z0-9']+",norm(x)) if len(w)>2 and w not in STOP}
def value(x):
 v=x.get("value")
 if v is None and isinstance(x.get("answer_value"),dict):v=x["answer_value"].get("value")
 if v is None:v=x.get("event") or x.get("time") or (x.get("scope") or {}).get("time_expression")
 return norm(v)

raws={x["raw_id"]:x for x in rows(INPUT/f"{PREFIX}_RAW_VIEWS.jsonl")}; side={x["raw_id"]:x for x in rows(SIDE)}
by_session=defaultdict(list)
for rid in raws:by_session[rid.rsplit(":",1)[0]].append(rid)
previous={}
for ids in by_session.values():
 for i,rid in enumerate(ids):
  if i:previous[rid]=ids[i-1]
facts=defaultdict(list)
for p in (INPUT/f"{PREFIX}_ASSERTIONS.jsonl",INPUT/f"{PREFIX}_PROPOSITIONS.jsonl"):
 for x in rows(p):facts[x["raw_id"]].append(x)
incoming=defaultdict(set);outgoing=defaultdict(set)
for e in rows(INPUT/f"{PREFIX}_GRAPH_EDGES.jsonl"):
 if e.get("edge_type") in {"ANTECEDENT","OCCURRENCE"}:
  outgoing[e["source_raw_id"]].add(e["target_raw_id"]);incoming[e["target_raw_id"]].add(e["source_raw_id"])

out=[];stats=Counter()
for rid in sorted(raws):
 sc=side.get(rid,{});ctx=sc.get("contextual_v41") or {};acts=ctx.get("speech_acts") or []
 act_names=sorted({str(a.get("act") or "UNKNOWN").upper() for a in acts})
 strong_response=bool(set(act_names)&{"CONFIRMATION","DENIAL","ANSWER","CORRECTION"}) or any(a.get("resolved_proposition") for a in acts)
 bindings=[]
 for x in facts[rid]:
  role_values={k:norm(v) for k,v in (x.get("roles") or {}).items() if norm(v)}
  sig={"subject":norm(x.get("subject") or x.get("speaker")),"relation":norm(x.get("relation") or x.get("surface_relation")),"value":value(x),"event":norm(x.get("event")),"time":norm(x.get("time") or (x.get("scope") or {}).get("time_expression")),"polarity":norm(x.get("polarity")),"modality":norm(x.get("modality")),"role_values":role_values}
  if any(sig.values()) and sig not in bindings:bindings.append(sig)
 response=bool(set(act_names)&{"CONFIRMATION","DENIAL","ANSWER","CORRECTION","REACTION"})
 antecedents=sorted(set(ctx.get("antecedent_raw_ids") or [])|outgoing[rid])
 inferred_links=[]
 if raws[rid].get("sidecar_status")=="BASE_REUSED" and rid in previous:
  prior=previous[rid];text=norm(raws[rid]["raw_text"]);prior_text=norm(raws[prior]["raw_text"])
  marker=bool(re.match(r"^(yes|yeah|yep|no|nope|sure|thanks|thank you|exactly|definitely|absolutely|that|this|it|they|he|she|sounds|wow|great|nice|sorry)\b",text))
  answers_question=prior_text.rstrip().endswith("?") and raws[prior].get("speaker")!=raws[rid].get("speaker")
  fact_terms=set().union(*(toks(" ".join((b["subject"],b["relation"],b["value"],b["event"]))) for b in bindings)) if bindings else set()
  prior_terms=toks(raws[prior]["retrieval_text"])
  same_event=len(fact_terms&prior_terms)>=2
  if marker or answers_question or same_event:
   inferred_links.append({"target_raw_id":prior,"reason":"ANSWER_TO_QUESTION" if answers_question else "DISCOURSE_RESPONSE" if marker else "SHARED_EVENT_TERMS"})
 carrier=bool(bindings) or response
 social_only=not carrier and bool(act_names) and set(act_names)<={"NON_PROPOSITIONAL","REACTION","QUESTION","REQUEST"}
 item={"raw_id":rid,"conversation_id":raws[rid]["conversation_id"],"sidecar_status":raws[rid].get("sidecar_status"),"bindings":bindings,"value_signatures":sorted({x["value"] for x in bindings if x["value"]}),"time_signatures":sorted({x["time"] for x in bindings if x["time"]}),"speech_acts":act_names,"context_dependency":ctx.get("context_dependency") or ("INFERRED_LOCAL" if inferred_links else "NONE"),"antecedent_raw_ids":antecedents,"inferred_context_links":inferred_links,"is_antecedent":bool(incoming[rid]),"is_response":response,"is_strong_response":strong_response,"is_carrier":carrier,"social_only":social_only,"schema_version":"raw-evidence-interface-v1.2","gold_visible":False}
 out.append(item);stats[raws[rid].get("sidecar_status")]+=1;stats["carrier"]+=carrier;stats["response"]+=response;stats["strong_response"]+=strong_response;stats["antecedent"]+=bool(incoming[rid]);stats["inferred_context"]+=bool(inferred_links)
with (OUTPUT/f"{OUTPUT_NAME}.jsonl").open("w",encoding="utf8") as f:
 for x in out:f.write(json.dumps(x,ensure_ascii=False)+"\n")
(OUTPUT/f"{OUTPUT_NAME}_RECEIPT.json").write_text(json.dumps({"raws":len(out),"stats":stats,"gold_used":False},ensure_ascii=False,indent=2)+"\n")
print(json.dumps({"raws":len(out),"stats":stats,"gold_used":False},ensure_ascii=False))
