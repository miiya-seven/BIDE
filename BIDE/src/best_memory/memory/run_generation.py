#!/usr/bin/env python3
import argparse,concurrent.futures,hashlib,json,os,re,time,urllib.request,subprocess
from pathlib import Path

import json_repair

from best_memory.memory.context_gate import contextual_status

H=Path(__file__).resolve().parent
def row_iter(p):
  """Stream JSONL requests instead of retaining the full LongMemEval corpus.

  A session-level LongMemEval request file is hundreds of MB.  The previous
  list-based loader could be killed by the host before the first API request,
  which made a run look as if it had silently stopped at ``RUN memory``.
  """
  with Path(p).open(encoding="utf8") as f:
    for line in f:
      if line.strip(): yield json.loads(line)

def request_count(p):
  with Path(p).open(encoding="utf8") as f:
    return sum(1 for line in f if line.strip())
def canon(x): return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(",",":"))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def text_sha(value): return hashlib.sha256(value.encode("utf8")).hexdigest()
def parse(wire):
  env=json.loads(wire); text=env["choices"][0]["message"]["content"].strip()
  # Small local replicas occasionally answer with generated Python code
  # instead of the requested JSON object.  This is task drift, not malformed
  # JSON that json_repair can safely recover.
  if (text.startswith("```python") or text.startswith("```py")
      or text.startswith("import json") or text.startswith("from dataclasses")):
    raise ValueError("model_task_drift: python code instead of JSON")
  if text.startswith("```"): text=text.split("\n",1)[1].rsplit("```",1)[0]
  try:
    value=json.loads(text)
  except json.JSONDecodeError:
    value=json_repair.loads(text)
    if not isinstance(value,dict): raise
    value.setdefault("_best276_wire_repairs",[]).append("json_syntax_repair")
  return value,env.get("usage",{})

def _repair_memory_shape(v, src):
  """Apply only lossless contract repairs before semantic validation.

  LongMemEval sessions are large and the remote model occasionally returns a
  mostly-correct legacy object with a malformed L1 wrapper.  Rejecting the
  whole session for a missing assertion link wastes the request and leaves a
  hole in the corpus.  These repairs never invent a proposition: unsupported
  values are kept with the ``OTHER`` type/relation, and an asserted unit with
  no linked assertion is downgraded to ``UNRESOLVED``.
  """
  if not isinstance(v,dict): raise ValueError("memory object is not a JSON object")
  repairs=[]
  center=src["center"]; rid=center["raw_id"]
  v.setdefault("center_raw_id",rid)
  l1=v.get("l1")
  if not isinstance(l1,dict):
    l1={}; v["l1"]=l1; repairs.append("missing_l1_wrapper")
  raw_cu=l1.get("clause_units")
  if not isinstance(raw_cu,list): raw_cu=[]; repairs.append("missing_clause_units")
  # Some responses accidentally place l2_direct as list/string entries inside
  # clause_units.  Recover list entries; string fragments are discarded.
  hidden=[]; clean_cu=[]
  for cu in raw_cu:
    if isinstance(cu,dict): clean_cu.append(cu)
    elif isinstance(cu,list): hidden.extend(x for x in cu if isinstance(x,dict)); repairs.append("lifted_nested_assertions")
    else: repairs.append("dropped_malformed_clause_unit")
  if not isinstance(v.get("l2_direct"),list):
    v["l2_direct"]=hidden
    if hidden: repairs.append("lifted_l2_direct")
  elif hidden:
    v["l2_direct"].extend(hidden); repairs.append("merged_nested_l2_direct")
  valid_clause_ids=[str(x.get("clause_id")) for x in center.get("clauses",[]) if isinstance(x,dict) and x.get("clause_id")]
  if not valid_clause_ids: valid_clause_ids=["c1"]
  by_id={str(x.get("clause_id")):x for x in clean_cu if x.get("clause_id") is not None}
  fixed=[]
  for cid in valid_clause_ids:
    cu=by_id.get(cid,{"clause_id":cid,"units":[]})
    units=cu.get("units")
    if not isinstance(units,list): units=[]; repairs.append("normalized_units_list")
    fixed_units=[]
    for u in units:
      if not isinstance(u,dict): repairs.append("dropped_malformed_unit"); continue
      if not isinstance(u.get("assertion_ids"),list):
        u["assertion_ids"]=([u["assertion_ids"]] if u.get("assertion_ids") else []); repairs.append("normalized_assertion_ids")
      if u.get("status") not in ("ASSERTED","NON_PROPOSITIONAL","UNRESOLVED"):
        # A valid assertion link is direct evidence that the unit was
        # extracted.  Preserve that evidence when only the model's status
        # label falls outside the closed enum; otherwise the contextual gate
        # would pay for a second call to recover an already-linked fact.
        u["status"]="ASSERTED" if u.get("assertion_ids") else "UNRESOLVED"
        repairs.append("normalized_unit_status")
      fixed_units.append(u)
    if not fixed_units:
      fixed_units=[{"unit_id":"u1","span":str(next((x.get("text","") for x in center.get("clauses",[]) if str(x.get("clause_id"))==cid),"")),"status":"UNRESOLVED","assertion_ids":[],"reason":"NO_VALID_UNIT","resolution_status":"SCHEMA_FALLBACK"}]
      repairs.append("inserted_unresolved_unit")
    cu["clause_id"]=cid; cu["units"]=fixed_units; fixed.append(cu)
  l1["clause_units"]=fixed
  if not isinstance(l1.get("mentions"),list): l1["mentions"]=[]; repairs.append("normalized_mentions")
  normalized_mentions=[]
  for mention in l1["mentions"]:
    if not isinstance(mention,dict):
      repairs.append("dropped_malformed_mention"); continue
    cid=mention.get("clause_id")
    if cid not in valid_clause_ids:
      supplied=[str(x) for x in (mention.get("clause_ids") or []) if str(x) in valid_clause_ids]
      # Each turn request normally has one authoritative clause.  Mapping an
      # otherwise valid mention to that sole clause is lossless provenance
      # normalization, not semantic inference.
      if len(supplied)==1:
        cid=supplied[0]
      elif len(valid_clause_ids)==1:
        cid=valid_clause_ids[0]
      else:
        repairs.append("dropped_unanchored_mention"); continue
      mention["clause_id"]=cid; repairs.append("normalized_mention_provenance")
    mention.pop("clause_ids",None)
    if mention.get("role") not in ONTO["roles"]:
      mention["role"]="UNKNOWN"; repairs.append("normalized_mention_role")
    if "canonical_entity" not in mention:
      mention["canonical_entity"]=mention.get("normalized")
    mention.setdefault("occurrence_local_id",None)
    normalized_mentions.append(mention)
  l1["mentions"]=normalized_mentions
  assertions=[]
  for a in v.get("l2_direct") or []:
    if not isinstance(a,dict): repairs.append("dropped_malformed_assertion"); continue
    av=a.get("answer_value")
    if not isinstance(av,dict): repairs.append("dropped_missing_answer_value"); continue
    if not a.get("assertion_id"): a["assertion_id"]=f"a{len(assertions)+1}"; repairs.append("generated_local_assertion_id")
    if av.get("type") not in ONTO["answer_types"]:
      av["type"]="OTHER"; repairs.append("normalized_answer_type")
    if a.get("relation_family") not in ONTO["relation_families"]:
      a["relation_family"]="OTHER"; repairs.append("normalized_relation_family")
    if a.get("modality") not in ONTO["modalities"]:
      a["modality"]="UNKNOWN"; repairs.append("normalized_modality")
    if a.get("polarity") not in ONTO["polarities"]:
      a["polarity"]="UNKNOWN"; repairs.append("normalized_polarity")
    if not isinstance(a.get("roles"),dict): a["roles"]={}; repairs.append("normalized_roles")
    if not isinstance(a.get("scope"),dict): a["scope"]={"time_expression":None,"time_owner":None,"location":None}; repairs.append("normalized_scope")
    p=a.get("provenance")
    if not isinstance(p,dict): p={}; a["provenance"]=p; repairs.append("normalized_provenance")
    if p.get("raw_ids") != [rid]: p["raw_ids"]=[rid]; repairs.append("normalized_raw_provenance")
    cids=[str(x) for x in (p.get("clause_ids") or []) if str(x) in valid_clause_ids]
    if not cids: cids=[valid_clause_ids[0]]; repairs.append("normalized_clause_provenance")
    p["clause_ids"]=cids
    # Preserve the model's local L1 provenance when present.  The final
    # compiler validates these IDs against the canonicalized units below.
    if not isinstance(p.get("unit_ids"),list):
      p["unit_ids"]=[]
    if not a.get("subject") or not a.get("retrieval_text") or av.get("value") in (None,"","scalar"):
      repairs.append("dropped_incomplete_assertion"); continue
    assertions.append(a)
  v["l2_direct"]=assertions
  kept={str(a.get("assertion_id")) for a in assertions if a.get("assertion_id")}
  assertions_by_clause={cid:[] for cid in valid_clause_ids}
  for assertion in assertions:
    assertion_id=str(assertion.get("assertion_id"))
    for cid in (assertion.get("provenance") or {}).get("clause_ids",[]):
      if cid in assertions_by_clause: assertions_by_clause[cid].append(assertion_id)
  for cu in l1["clause_units"]:
    cid=cu["clause_id"]
    for u in cu["units"]:
      u["assertion_ids"]=[x for x in u.get("assertion_ids",[]) if str(x) in kept]
      if u.get("status")=="ASSERTED" and not u["assertion_ids"]:
        u["status"]="UNRESOLVED"; u["reason"]=u.get("reason") or "MISSING_ASSERTION_LINK"; repairs.append("downgraded_unlinked_asserted_unit")
    # When the model returned valid assertions but malformed the L1 wrapper,
    # provenance already provides an exact, lossless clause-to-assertion link.
    # Recover that link instead of routing the record to contextual resolution.
    linked=assertions_by_clause.get(cid,[])
    if linked and len(cu["units"])==1 and cu["units"][0].get("reason")=="NO_VALID_UNIT":
      cu["units"][0].update({"status":"ASSERTED","assertion_ids":linked,"reason":None})
      repairs.append("reconstructed_unit_from_assertion_provenance")
  if repairs: v["_best276_schema_repairs"]=sorted(set(repairs))
  return v

ONTO=json.loads((H/"ontology.json").read_text())
QCONTRACT={"sample_id":"string","question":"verbatim string","target_entities":["string"],"relation_family":ONTO["relation_families"],"answer_type":ONTO["answer_types"],"required_roles":ONTO["roles"],"evidence_shape":ONTO["evidence_shapes"],"modality_constraint":ONTO["query_modality_constraints"],"polarity_constraint":ONTO["polarities"],"temporal_constraint":None,"location_constraint":None,"allowed_inference":ONTO["allowed_inference"]}
MCONTRACT={"center_raw_id":"string","l1":{"clause_units":[{"clause_id":"center clause id","units":[{"unit_id":"u1","span":"exact or minimal source text","status":"ASSERTED|NON_PROPOSITIONAL|UNRESOLVED","assertion_ids":["a1"],"reason":None}]}],"mentions":[{"text":"source mention","canonical_entity":None,"role":"ontology role or UNKNOWN","clause_id":"center clause id","occurrence_local_id":None}]},"l2_direct":[{"assertion_id":"a1","subject":"explicit resolved subject","relation_family":"enum","surface_relation":"literal predicate","answer_value":{"type":"answer type enum","value":"scalar"},"roles":{"observer":None,"evaluated_subject":None,"agent":None,"participant":None,"event":None,"time_owner":None,"location":None,"cause":None,"result":None,"left_side":None,"right_side":None,"set_member":None},"occurrence_local_id":None,"modality":"enum","polarity":"enum","scope":{"time_expression":None,"time_owner":None,"location":None},"authority":"DIALOGUE_TEXT|VISUAL_CAPTION","retrieval_text":"self-contained text","provenance":{"raw_ids":["center only"],"clause_ids":["center clauses"],"unit_ids":["c1::unit::1"]}}]}

def validate_query(v,src):
  if v.get("sample_id")!=src["sample_id"] or v.get("question")!=src["question"]: raise ValueError("query identity/text drift")
  # The endpoint occasionally collapses a one-item JSON array to a scalar.
  # This is a lossless wire-shape repair, not an ontology or semantic rewrite.
  for key in ("target_entities","required_roles","allowed_inference"):
    if isinstance(v.get(key),str): v[key]=[v[key]]
  # Closed-ontology lexical alias: a requested TIME role is operationally the
  # owner whose timestamp constrains the answer, never a new free-form role.
  v["required_roles"]=["TIME_OWNER" if x=="TIME" else x for x in v.get("required_roles",[])]
  for key,allowed in (("relation_family",ONTO["relation_families"]),("answer_type",ONTO["answer_types"]),("evidence_shape",ONTO["evidence_shapes"]),("polarity_constraint",ONTO["polarities"])):
    if v.get(key) not in allowed: raise ValueError("query enum "+key)
  if v.get("modality_constraint") not in ONTO["query_modality_constraints"]: raise ValueError("query enum modality_constraint")
  for key,allowed in (("required_roles",ONTO["roles"]),("allowed_inference",ONTO["allowed_inference"])):
    if any(x not in allowed for x in v.get(key,[])): raise ValueError("query list enum "+key)

def compile_memory(v,src):
  # Current gpt-5.4-mini follows the shared-semantic-ontology contract from
  # memory_prompt.md (`l1_resolution` / `l2_assertions`), while the historical
  # retrieval stages consume the older normalized `l1` / `l2_direct` view.
  # Convert only schema shape here; all assertion values, clause provenance,
  # modality and polarity remain model-produced and are validated below.
  if "l2_assertions" in v and "l1_resolution" in v:
    center=src["center"]
    resolution=v.get("l1_resolution") if isinstance(v.get("l1_resolution"),dict) else {}
    rid=str(resolution.get("center_raw_id") or v.get("center_raw_id") or center["raw_id"])
    if rid != center["raw_id"]: raise ValueError("memory identity drift")
    assertions=[]; by_clause={}
    raw_assertions=v.get("l2_assertions") if isinstance(v.get("l2_assertions"),list) else []
    for index,a in enumerate(raw_assertions,1):
      if not isinstance(a,dict): continue
      clause_ids=[str(x) for x in (a.get("clause_ids") or [])]
      local_id=str(a.get("assertion_id") or f"a{index}")
      answer_type=str(a.get("answer_type") or "OTHER")
      value=a.get("answer_value")
      if isinstance(value,dict):
        value_type=str(value.get("type") or answer_type); value=value.get("value")
      else: value_type=answer_type
      if value_type not in ONTO["answer_types"]: value_type="OTHER"
      relation=str(a.get("surface_relation") or a.get("relation_family") or "OTHER")
      subject=str(a.get("subject") or "")
      retrieval_text=" ".join(str(x) for x in (subject,relation,value) if x not in (None,""))
      item={"assertion_id":local_id,"subject":subject,"relation_family":str(a.get("relation_family") or "OTHER"),
            "surface_relation":relation,"answer_value":{"type":value_type,"value":value},
            "roles":{str(a.get("role") or "ANSWER_VALUE"):subject},
            "occurrence_local_id":a.get("occurrence_local_id"),
            "modality":str(a.get("modality") or "UNKNOWN"),"polarity":str(a.get("polarity") or "UNKNOWN"),
            "scope":{"time_expression":a.get("time"),"time_owner":None,"location":None},
            "authority":"DIALOGUE_TEXT","retrieval_text":retrieval_text,
            "provenance":{"raw_ids":[rid],"clause_ids":clause_ids}}
      assertions.append(item)
      for clause_id in clause_ids: by_clause.setdefault(clause_id,[]).append(local_id)
    units=[]
    for clause in center.get("clauses",[]):
      cid=str(clause["clause_id"]); ids=by_clause.get(cid,[])
      units.append({"clause_id":cid,"units":[{"unit_id":f"u{len(units)+1}","span":str(clause.get("text") or ""),
          "status":"ASSERTED" if ids else "NON_PROPOSITIONAL","assertion_ids":ids,
          "reason":None if ids else "NO_DIRECT_ASSERTION"}]})
    v={"center_raw_id":rid,"l1":{"clause_units":units,"mentions":[]},"l2_direct":assertions,
       "schema_version":"shared-semantic-ontology-v1-adapted"}
  v=_repair_memory_shape(v,src)
  center=src["center"]; rid=center["raw_id"]
  if v.get("center_raw_id")!=rid: raise ValueError("memory identity drift")
  clause_ids={x["clause_id"] for x in center["clauses"]}; seen=[]; local_assert={}
  for cu in v.get("l1",{}).get("clause_units",[]):
    cid=cu.get("clause_id"); seen.append(cid)
    if cid not in clause_ids: raise ValueError("foreign clause unit")
    if not cu.get("units"): raise ValueError("empty clause units")
    for u in cu["units"]:
      if u.get("status") not in ("ASSERTED","NON_PROPOSITIONAL","UNRESOLVED"): raise ValueError("unit status")
  if set(seen)!=clause_ids or len(seen)!=len(set(seen)): raise ValueError("clause coverage mismatch")
  assertions=v.get("l2_direct",[])
  for i,a in enumerate(assertions,1):
    old=a.get("assertion_id"); new=f"{rid}::l2::{i}"; local_assert[old]=new; a["assertion_id"]=new
    if a.get("relation_family") not in ONTO["relation_families"]: raise ValueError(f"assertion {old} invalid relation_family={a.get('relation_family')!r}; choose one exact relation_families enum")
    av=a.get("answer_value")
    if not isinstance(av,dict): raise ValueError(f"assertion {old} missing required answer_value object; infer its actual type and value from the cited center clause")
    if av.get("type") not in ONTO["answer_types"]: raise ValueError(f"assertion {old} invalid answer_value.type={av.get('type')!r}; choose one exact answer_types enum")
    if av.get("value") in (None,"","scalar"): raise ValueError(f"assertion {old} answer_value.value is missing or a placeholder; copy the actual answer-bearing value from the cited center clause")
    if a.get("modality") not in ONTO["modalities"] or a.get("polarity") not in ONTO["polarities"]: raise ValueError("memory scope enum")
    p=a.get("provenance",{})
    if p.get("raw_ids") != [rid] or not p.get("clause_ids") or not set(p["clause_ids"])<=clause_ids: raise ValueError("memory provenance")
    if not a.get("subject") or not a.get("retrieval_text"): raise ValueError("memory retrieval identity")
  unit_by_id={}
  for ci,cu in enumerate(v["l1"]["clause_units"],1):
    for ui,u in enumerate(cu["units"],1):
      u["unit_id"]=f"{cu['clause_id']}::unit::{ui}"
      span=str(u.get("span") or "").strip()
      if not span: raise ValueError("unit span empty")
      if re.fullmatch(r"\s*\d+\s*[-:]\s*\d+\s*",span): raise ValueError("unit span is character range")
      u["assertion_ids"]=[local_assert[x] for x in u.get("assertion_ids",[]) if x in local_assert]
      if u.get("status")=="ASSERTED" and not u["assertion_ids"]: raise ValueError("asserted unit without assertion")
      unit_by_id[u["unit_id"]]=u
  for a in assertions:
    p=a["provenance"]
    unit_ids=p.get("unit_ids")
    if not isinstance(unit_ids,list) or not unit_ids: raise ValueError("assertion missing provenance.unit_ids")
    for uid in unit_ids:
      if uid not in unit_by_id:
        # Local replicas can occasionally retain a stale L2 unit reference
        # after L1 compaction.  The assertion has no surviving evidence;
        # skip that link rather than rejecting the whole Raw.
        continue
      if uid.split("::unit::",1)[0] not in p["clause_ids"]: raise ValueError("assertion unit/clause provenance mismatch")
      unit=unit_by_id[uid]
      if unit.get("status")!="ASSERTED": raise ValueError("assertion points to non-ASSERTED unit")
      if a["assertion_id"] not in unit.get("assertion_ids",[]): raise ValueError("L1/L2 assertion link mismatch")
  for m in v.get("l1",{}).get("mentions",[]):
    if m.get("clause_id") not in clause_ids: raise ValueError("mention provenance")
  return v


def self_contained_raw_memory(src):
  """Build a lossless session memory without inventing semantic L2 facts.

  LongMemEval's session is already the benchmark retrieval unit.  Calling the
  memory extractor once per 15k-character session would make the 23k-session
  full run impractical, while extracting a synthetic assertion would leak an
  interpretation into retrieval.  Keep the complete session as a
  NON_PROPOSITIONAL clause; the raw corpus lane remains authoritative and the
  receipt records this explicit adaptation backend.
  """
  center=src["center"]; rid=center["raw_id"]; text=str(center.get("text") or "")
  clause_id="c1"
  return {
    "center_raw_id": rid,
    "l1": {"clause_units": [{"clause_id": clause_id, "units": [{
      "unit_id": f"{clause_id}::unit::1", "span": text,
      "status": "NON_PROPOSITIONAL", "assertion_ids": [],
      "reason": "LONGMEMEVAL_SESSION_SELF_CONTAINED_RAW"
    }]}], "mentions": []},
    "l2_direct": [],
  }

def memory_request_view(src, dataset):
  """Return the non-redundant source sent to the memory extractor.

  LongMemEval session rows retain ``center.text`` for raw retrieval, while
  ``center.clauses`` contains the exact per-turn source segments.  Sending
  both copies nearly doubles the prompt and makes the model treat a whole
  session as one undifferentiated clause.  The extractor only needs the
  clause text plus provenance metadata, so omit the derived concatenation from
  the API payload.  The original ``src`` remains the validator authority.
  """
  if dataset != "longmemeval":
    return src
  center = src.get("center") or {}
  keep = {"raw_id", "speaker", "timestamp", "source_session_id", "has_answer", "clauses"}
  compact = {k: center[k] for k in keep if k in center}
  clauses = compact.get("clauses")
  if not isinstance(clauses, list) or not clauses:
    compact["clauses"] = [{"clause_id": "c1", "text": str(center.get("text") or "")}]
  # Preserve the bounded same-session neighbors supplied by the LongMemEval
  # turn adapter.  They are explicitly context-only and are never promoted to
  # center clauses or valid L2 provenance.
  context_only = src.get("context_only")
  if isinstance(context_only, list):
    return {"center": compact, "context_only": context_only}
  return {"center": compact, "context_only": []}

def split_memory_sources(src, max_chars=6000):
  """Split a LongMemEval session into bounded clause-preserving requests.

  The benchmark's retrieval unit remains one session/raw.  This split only
  bounds the model extraction call; all chunk outputs are merged back under
  the original raw id before the corpus stage.  A clause is never split in
  the middle, so provenance remains exact.
  """
  compact = memory_request_view(src, "longmemeval")
  center = compact["center"]
  clauses = center.get("clauses") or []
  # Conversation-owned requests are assembled with clause ids ``sN::cM``.
  # Preserve the session boundary first; only split inside one session when
  # that session itself exceeds the model budget.
  if str(center.get("raw_id", "")).endswith("::CONVERSATION"):
    groups=[]; current_key=None; current=[]
    for clause in clauses:
      key=str(clause.get("clause_id", "")).split("::", 1)[0]
      if current and key != current_key:
        groups.append(current); current=[]
      current_key=key; current.append(clause)
    if current: groups.append(current)
    chunks=[]
    for group in groups:
      sub=[]; chars=0
      for clause in group:
        cost=len(str(clause.get("text") or ""))+len(str(clause.get("clause_id") or ""))+32
        if sub and chars+cost > max_chars:
          chunks.append({"center":{**center,"clauses":sub}}); sub=[]; chars=0
        sub.append(clause); chars += cost
      if sub: chunks.append({"center":{**center,"clauses":sub}})
    return chunks or [compact]
  # A single LongMemEval turn can itself be a very long generated document.
  # Do not let the clause-preservation rule bypass the chunk budget: split an
  # oversized clause into lossless paragraph/character parts while retaining
  # the parent clause id in every derived id.  The merge stage treats these as
  # extraction-only views and restores the original session/raw owner.
  expanded=[]
  for clause in clauses:
    text=str(clause.get("text") or "")
    cid=str(clause.get("clause_id") or "c1")
    budget=max(1000, int(max_chars)-len(cid)-64)
    if len(text) <= budget:
      expanded.append(clause); continue
    parts=[]
    for para in text.split("\n\n"):
      para = para.strip()
      if not para: continue
      sentences=re.split(r'(?<=[.!?。！？])(?=\s+[A-Z0-9"“(]|$)', para)
      sentences=[s.strip() for s in sentences if s.strip()]
      cur=""
      for sent in sentences:
        if len(sent) <= budget and cur and len(cur)+1+len(sent) <= budget:
          cur += " " + sent
        elif len(sent) <= budget:
          if cur: parts.append(cur)
          cur=sent
        else:
          if cur: parts.append(cur); cur=""
          rest=sent
          while len(rest) > budget:
            cut=rest.rfind(" ", 0, budget)
            if cut < max(200, budget//2): cut=budget
            parts.append(rest[:cut].rstrip()); rest=rest[cut:].lstrip()
          if rest: cur=rest
      if cur: parts.append(cur)
    for j, part in enumerate(parts, 1):
      expanded.append({**clause, "clause_id": f"{cid}::part{j}", "text": part,
                       "parent_clause_id": cid})
  clauses=expanded
  chunks=[]; current=[]; current_chars=0
  for clause in clauses:
    text=str(clause.get("text") or "")
    cost=len(text)+len(str(clause.get("clause_id") or "c1"))+32
    if current and current_chars+cost > max_chars:
      chunks.append({"center":{**center,"clauses":current}})
      current=[]; current_chars=0
    current.append(clause); current_chars += cost
  if current:
    chunks.append({"center":{**center,"clauses":current}})
  return chunks or [compact]

def merge_memory_chunks(chunk_values, chunk_sources, src):
  """Merge raw chunk JSON into one validated canonical memory object."""
  rid=src["center"]["raw_id"]
  merged={"center_raw_id":rid,"l1":{"clause_units":[],"mentions":[]},"l2_direct":[]}
  next_assertion=1
  for chunk_index,(value,chunk_src) in enumerate(zip(chunk_values,chunk_sources),1):
    # Local compact L2 models sometimes omit unit_ids even though they return
    # a clause-local assertion.  Recover that link only when it is lossless:
    # the assertion names exactly one clause and that clause has exactly one
    # ASSERTED unit.  Ambiguous assertions remain unlinked and are rejected by
    # the normal shape repair instead of being attached by guesswork.
    raw_l1 = value.get("l1") if isinstance(value,dict) else None
    raw_l2 = value.get("l2_direct") if isinstance(value,dict) else None
    if isinstance(raw_l1,dict) and isinstance(raw_l2,list):
      clause_units = {str(c.get("clause_id")): c.get("units",[]) for c in raw_l1.get("clause_units",[]) if isinstance(c,dict)}
      for assertion in raw_l2:
        if not isinstance(assertion,dict):
          continue
        prov = assertion.setdefault("provenance", {})
        ids = prov.get("unit_ids")
        if isinstance(ids,list) and ids:
          continue
        cids = [str(x) for x in (prov.get("clause_ids") or []) if str(x) in clause_units]
        candidates=[]
        if len(cids)==1:
          candidates=[u for u in clause_units[cids[0]] if isinstance(u,dict) and u.get("status")=="ASSERTED" and u.get("unit_id")]
        if len(candidates)==1:
          prov["unit_ids"]=[str(candidates[0]["unit_id"])]
          assertion.setdefault("_best276_repairs",[]).append("recovered_unit_link_from_single_asserted_clause")
        elif not candidates:
          # Last-resort lexical anchoring for compact models that omit both
          # unit_ids and clause provenance.  Accept only a unique, strong
          # overlap between retrieval_text/subject and one ASSERTED span.
          text = " ".join(str(assertion.get(k) or "") for k in ("retrieval_text","subject","surface_relation")).lower()
          toks = {t for t in re.findall(r"[\w'-]+", text) if len(t) >= 4}
          scored=[]
          for cid, units in clause_units.items():
            for u in units:
              if isinstance(u,dict) and u.get("status")=="ASSERTED" and u.get("unit_id"):
                ut={t for t in re.findall(r"[\w'-]+", str(u.get("span") or "").lower()) if len(t)>=4}
                overlap=len(toks & ut)
                if overlap: scored.append((overlap/(len(ut) or 1), overlap, str(u["unit_id"])))
          scored.sort(reverse=True)
          if scored and scored[0][0] >= 0.5 and (len(scored)==1 or scored[0][:2] > scored[1][:2]):
            prov["unit_ids"]=[scored[0][2]]
            assertion.setdefault("_best276_repairs",[]).append("recovered_unit_link_by_unique_lexical_overlap")
      # Complete the inverse edge before shape repair.  Otherwise
      # _repair_memory_shape sees valid unit_ids but still downgrades every
      # L1 unit because its assertion_ids array is empty.
      for ai, assertion in enumerate(raw_l2, 1):
        if not isinstance(assertion,dict):
          continue
        assertion.setdefault("assertion_id", f"a{ai}")
        ids=set(str(x) for x in ((assertion.get("provenance") or {}).get("unit_ids") or []))
        if not ids:
          continue
        for units in clause_units.values():
          for unit in units:
            if isinstance(unit,dict) and str(unit.get("unit_id")) in ids:
              unit.setdefault("assertion_ids",[])
              if assertion["assertion_id"] not in unit["assertion_ids"]:
                unit["assertion_ids"].append(assertion["assertion_id"])
    # Keep compatibility with the shared-semantic response shape supported by
    # compile_memory, while the normal contract already returns l1/l2_direct.
    fixed = compile_memory(value, chunk_src) if ("l1_resolution" in value and "l2_assertions" in value) else _repair_memory_shape(value, chunk_src)
    id_map={}
    for assertion in fixed.get("l2_direct",[]):
      old=str(assertion.get("assertion_id"))
      new=f"a{next_assertion}"; next_assertion += 1; id_map[old]=new
      assertion["assertion_id"]=new
      merged["l2_direct"].append(assertion)
    for clause_unit in fixed.get("l1",{}).get("clause_units",[]):
      for unit in clause_unit.get("units",[]):
        unit["assertion_ids"]=[id_map.get(str(x),str(x)) for x in unit.get("assertion_ids",[])]
      merged["l1"]["clause_units"].append(clause_unit)
    merged["l1"]["mentions"].extend(fixed.get("l1",{}).get("mentions",[]))
  # Do not run compile_memory against the original unsplit source here.  The
  # extraction-local clause IDs (for example ``c1::part7``) are deliberately
  # finer than the original ``c1``; re-compiling against ``src`` would treat
  # them as unknown clauses and silently discard otherwise valid chunk facts.
  # Each chunk was already shape-checked above.  Preserve the merged rows and
  # let the corpus compiler retain the parent raw/session provenance.
  merged.setdefault("_best276_schema_repairs", []).append("chunk_preserving_merge")
  return merged

def main():
  ap=argparse.ArgumentParser(); ap.add_argument("--mode",choices=["query","memory"],required=True); ap.add_argument("--workers",type=int,default=8); ap.add_argument("--resume",action="store_true"); ap.add_argument("--only-index",type=int); ap.add_argument("--only-indices",help="comma-separated sparse indices"); ap.add_argument("--start-index",type=int); ap.add_argument("--max-items",type=int); ap.add_argument("--memory-strategy",choices=["joint","two_stage","two_stage_reasonless","two_stage_compact","two_stage_local_llama"],default="joint"); ap.add_argument("--shard-count",type=int,default=1); ap.add_argument("--shard-index",type=int,default=0); args=ap.parse_args()
  # The local 8B adapter is intentionally the compact two-stage contract:
  # canonical 276 semantic rules, but stage-local JSON output small enough for
  # Llama-3.1-8B to preserve all links.  Keep the explicit name in the CLI,
  # while reusing the tested implementation below.
  if args.memory_strategy == "two_stage_local_llama":
    args.memory_strategy = "two_stage_compact"
  sparse_indices=None
  if args.only_indices:
    try: sparse_indices={int(x.strip()) for x in args.only_indices.split(",") if x.strip()}
    except ValueError as exc: raise SystemExit("--only-indices must be comma-separated integers") from exc
    if not sparse_indices or min(sparse_indices) < 0: raise SystemExit("--only-indices requires non-negative integers")
  if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
    raise SystemExit("require shard-count >= 1 and 0 <= shard-index < shard-count")
  if args.only_index is not None and sparse_indices is not None:
    raise SystemExit("--only-index and --only-indices cannot be combined")
  if (args.only_index is not None or sparse_indices is not None) and args.shard_count != 1:
    raise SystemExit("explicit indices and sharding cannot be combined")
  if (args.only_index is not None or sparse_indices is not None) and (args.start_index is not None or args.max_items is not None):
    raise SystemExit("explicit indices cannot be combined with --start-index/--max-items")
  if args.max_items is not None and args.max_items < 1:
    raise SystemExit("--max-items must be >= 1")
  if (args.start_index is not None or args.max_items is not None) and args.shard_count != 1:
    raise SystemExit("bounded ranges cannot be combined with sharding")
  dataset=os.environ.get('BEST276_DATASET','locomo')
  # LongMemEval Memory follows the 276 Raw contract by default.  A complete
  # session is a retrieval grouping, not a single semantic extraction Raw;
  # callers must opt into the legacy session adapter explicitly.
  granularity=os.environ.get('BEST276_LME_GRANULARITY','turn')
  self_contained=(dataset == 'longmemeval' and granularity == 'session' and
                  os.environ.get('BEST276_LME_SELF_CONTAINED_MEMORY','0') == '1')
  auth_env=os.environ.get('LLM_AUTH_ENV','GENERATION_API_KEY')
  # configured provider credentials are endpoint-specific: prefer the explicit
  # GENERATION_API_KEY alias, even when a legacy profile says API_KEY.
  endpoint_hint=os.environ.get('LLM_BASE_URL','')
  if 'selected provider' in endpoint_hint:
   auth_env='GENERATION_API_KEY'
  key=os.environ.get(auth_env) or os.environ.get('API_KEY') or os.environ.get('OPENAI_API_KEY')
  if not self_contained and not key: raise SystemExit(f'{auth_env} (or API_KEY/OPENAI_API_KEY) missing')
  network_mode=os.environ.get('LLM_NETWORK_MODE','inherit').strip().casefold()
  opener=(urllib.request.build_opener(urllib.request.ProxyHandler({}))
          if network_mode == 'direct' else urllib.request)
  mode=args.mode; run_root=Path(os.environ.get('BEST276_RUN_ROOT', H/'../../../../runs/default'))
  # LongMemEval sessions are substantially larger than the original LoCoMo
  # prompts.  The historical fixed 5 x 300-second retry policy allowed a
  # single overloaded request to hold an entire batch for many minutes.  Keep
  # the historical defaults for other datasets, but make this budget explicit
  # and bounded for the long-session run.
  if dataset == 'longmemeval':
    request_timeout = float(os.environ.get('LME_MEMORY_REQUEST_TIMEOUT', '180'))
    request_attempts = max(1, int(os.environ.get(
      'LME_MEMORY_REQUEST_ATTEMPTS', os.environ.get('LME_MEMORY_ATTEMPTS', '3'))))
    configured_batch_size = max(8, int(os.environ.get('LME_MEMORY_BATCH_SIZE', str(max(8, args.workers * 2)))))
  else:
    request_timeout = float(os.environ.get('MEMORY_BUILD_REQUEST_TIMEOUT', '300'))
    request_attempts = max(1, int(os.environ.get('MEMORY_BUILD_REQUEST_ATTEMPTS', '5')))
    configured_batch_size = max(32, args.workers * 4)
  use_openai_sdk = os.environ.get('LLM_USE_OPENAI_SDK', '').strip().casefold() in {'1', 'true', 'yes', 'on'}
  # Optional local replica pool.  A request keeps the same replica for its
  # L1/L2 calls, while different request indices are distributed across the
  # configured vLLM ports.  With no pool configured, preserve the historical
  # single-endpoint behavior.
  endpoint_pool=[x.strip().rstrip('/') for x in os.environ.get('LLM_BASE_URLS','').split(',') if x.strip()]
  if not endpoint_pool:
    endpoint_pool=[os.environ.get('LLM_BASE_URL','').rstrip('/')]
  request_path=run_root/('query/QUERY_REQUESTS.jsonl' if mode=='query' else 'memory/MEMORY_REQUESTS.jsonl')
  total=request_count(request_path)
  prompt=(H/("query_prompt.md" if mode=="query" else "memory_prompt.md")).read_text(); contract=QCONTRACT if mode=="query" else MCONTRACT
  system=prompt+"\nClosed ontology:\n"+canon(ONTO)+"\nExact output shape:\n"+canon(contract)+"\nReturn exactly one JSON object."
  # LongMemEval turn extraction is intentionally compact.  The full ontology
  # prompt was valid but caused the gateway/model route to reserve excessive
  # reasoning time even for a 300-character utterance.  Local compilation
  # remains the authoritative schema and provenance check.
  compact_memory_system=(
    "Extract direct memory facts from exactly one dialogue utterance. "
    "Return one JSON object only: do not write Python, pseudocode, explanations, or Markdown fences. "
    "The first character must be { and the last character must be }. "
    "Return JSON only with center_raw_id, l1.clause_units, l1.mentions, and l2_direct. "
    "Each clause_units item must be {clause_id, units:[{unit_id, span, status, assertion_ids, reason}]}; "
    "each mention must be {text, canonical_entity, role, clause_id, occurrence_local_id}. "
    "Use the supplied clause IDs and cite only this raw. Split compound statements into minimal units. "
    "Use NON_PROPOSITIONAL for greetings/questions/directives with no asserted real-world fact; never invent facts. "
    "A request to generate a story, advice, examples, or an explanation does not assert that its requested or fictional content happened. "
    "Use only the named top-level keys; do not emit dotted keys such as l1.clause_units. "
    "Unit status must be ASSERTED, NON_PROPOSITIONAL, or UNRESOLVED. "
    "For each assertion include subject as a string, relation_family, surface_relation, "
    "answer_value {type,value}, modality, polarity, roles as an object, "
    "scope {time_expression,time_owner,location}, retrieval_text, and provenance {raw_ids,clause_ids,unit_ids}. "
    "Enumerate every minimal direct proposition in every supplied clause; do not summarize, merge away, or "
    "select facts based on likely future questions. Preserve list members, comparison sides, negation, modality, "
    "time, evaluation, cause, and result as independently retrievable assertions. Do not duplicate an assertion "
    "inside a unit. Keep JSON representation concise without dropping semantics: use minimal faithful span and "
    "retrieval_text, reason=null for ASSERTED units, mentions only for assertion entities, roles={} and scope={} "
    "when those structures contain no information. "
    f"relation_family must be one of {canon(ONTO['relation_families'])}; "
    f"answer_value.type must be one of {canon(ONTO['answer_types'])}; "
    f"modality must be one of {canon(ONTO['modalities'])}; "
    f"polarity must be one of {canon(ONTO['polarities'])}; "
    f"mention role must be one of {canon(ONTO['roles'])}.\n"
  )
  l1_system=(
    "Perform only L1 segmentation for exactly one center dialogue Raw. Return JSON only as "
    "one object; never output Python code, pseudocode, explanations, or Markdown fences. "
    "The first character must be { and the last character must be }. "
    "{center_raw_id,l1:{clause_units:[{clause_id,units:[{unit_id,span,status,assertion_ids,reason}]}],mentions:[...]}}. "
    "Enumerate every minimal proposition unit in every supplied clause without summarizing or selecting by a future question. "
    "Use ASSERTED for direct proposition units, NON_PROPOSITIONAL for pure discourse, questions, and directives, and UNRESOLVED only when the unit needs context. "
    "A request to generate a story, advice, examples, or an explanation does not assert that its requested or fictional content happened. "
    "At this stage assertion_ids must be []. Mentions use {text,canonical_entity,role,clause_id,occurrence_local_id}. "
    "Every unit span must be non-empty source text copied from its clause; never emit span=null or span=\"\". "
    "Do not create trailing or numbered filler units. Use only the supplied clause IDs and do not invent unit IDs. "
    f"Mention role must be one of {canon(ONTO['roles'])}. Cite only supplied clause IDs."
  )
  compact_l1_system=l1_system + (
    " For ASSERTED and NON_PROPOSITIONAL units reason must be null; use a short reason code only for UNRESOLVED. "
    "Default to ASSERTED for an explicit factual sentence, including dates, numbers, lists, quotations, headings with factual text, "
    "and encyclopedia/article prose. Use UNRESOLVED only when a pronoun, ellipsis, omitted argument, or relative time genuinely "
    "cannot be resolved from this clause; long text or unfamiliar entities alone are never reasons for UNRESOLVED."
  )
  l2_system=(
    "Perform only L2 direct fact extraction from the center Raw and the frozen L1 segmentation. Return JSON only as "
    "{center_raw_id,l2_direct:[...]}. Emit every minimal direct proposition represented by ASSERTED L1 units; do not summarize, "
    "merge away, or select facts by likely future questions. Each assertion must include assertion_id, unit_ids, subject as a string, "
    "relation_family, surface_relation, answer_value {type,value}, roles as an object, occurrence_local_id, modality, polarity, "
    "scope as an object, authority, retrieval_text, and provenance {raw_ids,clause_ids}. unit_ids must cite frozen L1 unit IDs. "
    "scope must contain time_expression, time_owner, location (null if absent); preserve relative time verbatim in time_expression. "
    "roles may use only observer, evaluated_subject, agent, participant, event, time_owner, location, cause, result, left_side, right_side, set_member. "
    "authority must be DIALOGUE_TEXT or VISUAL_CAPTION according to the source channel. "
    "Make retrieval_text self-contained: replace I with the supplied speaker and resolve it to its supported referent. "
    "An evaluation of an experience must name the experience/event, not automatically the organization or location. "
    "Use the same occurrence_local_id for propositions about the same event; mention IDs are not event IDs. "
    "time_expression is a temporal phrase; time_owner is the named event or state to which that phrase applies, never the date or temporal phrase itself. "
    "For an event assertion populate roles.event with a self-contained event description naming its agent and action. "
    "For an evaluation of that event, reuse roles.event and set subject and roles.evaluated_subject to that event description, not a first-person source quote. "
    "Use event IDs such as e1, shared across assertions about that event, rather than copying mention IDs. "
    "Choose relation_family for the asserted predicate: attending or taking part in an activity is PARTICIPATION; "
    "LOCATION is for an assertion whose predicate locates an entity. Do not classify participation as LOCATION merely because its object names a group. "
    "Do not extract an assertion from NON_PROPOSITIONAL units or turn requested/fictional embedded content into an actual event. "
    f"relation_family is one of {canon(ONTO['relation_families'])}; answer_value.type is one of {canon(ONTO['answer_types'])}; "
    f"modality is one of {canon(ONTO['modalities'])}; polarity is one of {canon(ONTO['polarities'])}. Cite only the center Raw."
  )
  # Keep the semantic rules identical to the canonical 276 memory prompt even
  # when the JSON contract is split into L1 and L2 calls.  The stage-specific
  # suffixes below only constrain which half is emitted; they do not replace
  # the proposition, modality, provenance, and center-only rules.
  canonical_memory_rules = prompt
  compact_l2_system=(
    "Perform only L2 direct fact extraction from the center Raw and frozen L1. Return JSON only as {l2_direct:[...]}. "
    "Never output Python code, pseudocode, explanations, or Markdown fences. "
    "The first character must be { and the last character must be }. "
    "Emit every minimal direct proposition represented by ASSERTED L1 units. Each assertion must include unit_ids, subject as a string, "
    "relation_family, surface_relation, answer_value {type,value}, modality, polarity, and a concise self-contained retrieval_text. "
    "Include roles, scope, and occurrence_local_id only when their values are directly stated; omit null members. "
    "Do not output assertion_id, authority, provenance, center_raw_id, or explanatory text: they are restored deterministically. "
    "unit_ids is mandatory for every assertion: emit one or more exact unit IDs copied verbatim from frozen L1. "
    "Never omit unit_ids, use an empty list, invent, renumber, or guess a unit ID. "
    "If no valid ASSERTED unit exists, return {\"l2_direct\":[]}. Do not extract from NON_PROPOSITIONAL units or turn requested/fictional content into an actual event. "
    f"relation_family is one of {canon(ONTO['relation_families'])}; answer_value.type is one of {canon(ONTO['answer_types'])}; "
    f"modality is one of {canon(ONTO['modalities'])}; polarity is one of {canon(ONTO['polarities'])}."
  )
  compact_shared_rules=(
    "Shared rules: extract only direct propositions stated in the supplied center clause. "
    "Do not use context_only text as center evidence. Preserve negation, modality, time, "
    "comparisons, sets, causes, and results when explicitly stated. Do not infer or solve "
    "questions, advice requests, fictional scenarios, or directives. Keep every span as "
    "minimal verbatim source text. Cite only supplied clause IDs and frozen L1 unit IDs."
  )
  compact_l1_system += "\n" + compact_shared_rules
  compact_l2_system += "\n" + compact_shared_rules
  effective_prompt=((compact_l1_system+"\n---SHARED-RULES---\n"+compact_shared_rules+"\n---L2---\n"+compact_l2_system) if mode == "memory" and args.memory_strategy == "two_stage_compact" else
                    (compact_l1_system+"\n---CANONICAL-276-RULES---\n"+canonical_memory_rules+"\n---L2---\n"+l2_system if mode == "memory" and args.memory_strategy == "two_stage_reasonless" else
                    (l1_system+"\n---CANONICAL-276-RULES---\n"+canonical_memory_rules+"\n---L2---\n"+l2_system if mode == "memory" and args.memory_strategy == "two_stage" else
                    (compact_memory_system if mode == "memory" and dataset == "longmemeval" and granularity in {"turn", "conversation"} else system))
                   ))
  prompt_identity=text_sha(effective_prompt)
  ontology_identity=sha(H/"ontology.json")
  out=run_root/("query/QUERY_OUTPUT" if mode=="query" else "memory/MEMORY_OUTPUT"); (out/"wire").mkdir(parents=True,exist_ok=True); (out/"wire_meta").mkdir(exist_ok=True); (out/"rejected").mkdir(exist_ok=True)
  def call(pair):
    i,src=pair; p=out/"wire"/f"{i:04d}.json"
    meta_path=out/"wire_meta"/f"{i:04d}.json"
    rejected_path=out/"rejected"/f"{i:04d}.json"
    if self_contained:
      value=self_contained_raw_memory(src); rejected_path.unlink(missing_ok=True)
      return i,value,{},"PARSED",0
    if args.resume and p.exists():
      try:
        expected_model=os.environ.get('MEMORY_BUILD_MODEL') or os.environ.get('LLM_MODEL') or 'gpt-5.4-mini'
        meta=json.loads(meta_path.read_text(encoding="utf8"))
        permanent=(meta.get("content_status") == "CONTENT_VALIDATED_PERMANENT"
                   and meta.get("content_sha256") == sha(p))
        if not permanent and (meta.get("requested_model") != expected_model or meta.get("prompt_sha256") != prompt_identity
            or meta.get("memory_strategy", "joint") != args.memory_strategy
            or meta.get("ontology_sha256") != ontology_identity):
          raise ValueError("resume identity mismatch")
        v,u=parse(p.read_bytes()); validate_query(v,src) if mode=="query" else compile_memory(v,src); rejected_path.unlink(missing_ok=True); return i,v,u,"REUSED",0
      except Exception: p.rename(p.with_suffix(".invalid.json"))
    # configured provider's OpenAI-compatible SDK path is most reliable when the entire
    # instruction and source are carried in one user message.  LongMemEval
    # sessions are additionally split into bounded clause-preserving calls;
    # the merged value is written as one canonical wire envelope below.
    if mode == "memory" and dataset == "longmemeval":
      chunk_sources=split_memory_sources(src, max_chars=max(1000, int(os.environ.get("LME_MEMORY_CHUNK_CHARS", "6000"))))
    else:
      chunk_sources=[src]

    def invoke(request_src, validator_src, system_override=None):
      effective_system = system_override or (compact_memory_system if (mode == "memory" and dataset == "longmemeval" and granularity in {"turn", "conversation"}) else system)
      if use_openai_sdk:
        messages=[{"role":"user","content":effective_system+"\n\nSOURCE JSON:\n"+canon(request_src)}]
      else:
        messages=[{"role":"system","content":effective_system},{"role":"user","content":canon(request_src)}]
      # A single turn can contain many clauses; 1800 tokens routinely truncates
      # the required JSON before it can be parsed. Keep the full extraction
      # budget configurable, with 8000 as the safe default for LongMemEval.
      # The local Llama vLLM replica is served with max_model_len=16384.
      # An 8k completion budget plus a 9k-token LongMemEval prompt causes a
      # deterministic HTTP 400 before generation.  Keep the larger historical
      # budget for the full contract, but use a bounded local compact default;
      # callers can still override it explicitly via the environment.
      token_default = ('6000' if (mode == "memory" and
                                  args.memory_strategy == "two_stage_compact" and
                                  endpoint_pool and all("127.0.0.1" in x or "localhost" in x
                                                         for x in endpoint_pool))
                       else '8000')
      request_model = os.environ.get('MEMORY_BUILD_MODEL') or os.environ.get('LLM_MODEL') or 'gpt-5.4-mini'
      body={"model":request_model,"messages":messages,"max_tokens":int(os.environ.get('MEMORY_BUILD_LLM_MAX_TOKENS',token_default))}
      # Optional OpenRouter provider pinning for reproducible latency/quality
      # comparisons.  It is ignored unless explicitly configured.
      provider_order=[x.strip() for x in os.environ.get('OPENROUTER_PROVIDER_ORDER','').split(',') if x.strip()]
      if provider_order:
        body["provider"]={"order":provider_order,
                          "allow_fallbacks":os.environ.get('OPENROUTER_ALLOW_FALLBACKS','0').lower() in ('1','true','yes')}
      if not use_openai_sdk:
        body.update({"temperature":0,"response_format":{"type":"json_object"}})
        # configured provider's gpt-5 route accepts the minimal OpenAI JSON contract more
        # reliably than the legacy seed/reasoning extensions.  Keep the old
        # fields available for compatible providers, but make the compatibility
        # mode explicit for probes and reproducible reruns.
        if 'selected provider' not in endpoint_pool[0].lower() and os.environ.get('GENERIC_PROVIDER_COMPAT','').lower() not in {'1','true','yes'}:
          body.update({"seed":20260827,"reasoning_effort":"low"})
      err=""; last_wire=None
      for attempt in range(1,request_attempts+1):
        try:
          # With global sharding, ``i % shard_count`` is constant within a
          # shard (shard 0 gets only even indices, shard 1 only odd ones).
          # Route by the shard-local ordinal instead, otherwise one server's
          # entire workload would be pinned to a single replica/GPU.
          shard_local_index=i // max(1,args.shard_count)
          endpoint_base=endpoint_pool[shard_local_index % len(endpoint_pool)]
          endpoint=endpoint_base+'/chat/completions'
          request_body=canon(body).encode()
          # Make the actual outbound call observable without recording the
          # credential or full prompt.  This is especially useful for long
          # Memory requests, which can otherwise look like a stalled worker.
          trace_path=run_root/'memory'/'MEMORY_OUTPUT'/f'CALL_TRACE_{i:05d}.json'
          trace_path.parent.mkdir(parents=True, exist_ok=True)
          trace_path.write_text(json.dumps({
              'status':'ATTEMPTING', 'index':i, 'attempt':attempt,
              'transport':'sdk' if use_openai_sdk else ('curl' if network_mode == 'direct' else 'urllib'),
              'timeout_seconds':request_timeout,
              'endpoint':endpoint, 'model':body.get('model'),
              'max_tokens':body.get('max_tokens'),
              'request_fields':sorted(body),
              'has_seed':'seed' in body,
              'has_reasoning_effort':'reasoning_effort' in body,
              'prompt_bytes':len(request_body), 'started_at':time.time()
          }, ensure_ascii=False, indent=2)+'\n')
          if os.environ.get('BEST276_DEBUG_SAVE_REQUEST') == '1':
            (trace_path.parent/f'CALL_BODY_{i:05d}.json').write_bytes(request_body)
          if use_openai_sdk:
            try:
              from openai import OpenAI
              sdk_client=OpenAI(api_key=key, base_url=endpoint_base,
                                timeout=request_timeout, max_retries=0)
              sdk_response=sdk_client.chat.completions.create(**body)
              if hasattr(sdk_response, 'model_dump'):
                wire=canon(sdk_response.model_dump(mode='json')).encode()
              else:
                wire=canon(json.loads(sdk_response.model_dump_json())).encode()
            except Exception as exc:
              raise RuntimeError(f'sdk_transport_failed:{type(exc).__name__}:{str(exc)[:240]}') from None
          elif network_mode == 'direct':
            command=['curl','-sS','--fail-with-body','--max-time',str(int(request_timeout)),'-H','Content-Type: application/json',
                     '-H','Authorization: Bearer '+key,'--data-binary','@-',endpoint]
            try:
              completed=subprocess.run(command,input=request_body,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                                       timeout=request_timeout+10,check=True)
            except subprocess.CalledProcessError as exc:
              detail=(exc.stderr or b"").decode("utf-8",errors="replace").strip().replace("\n"," ")[:240]
              raise RuntimeError(f'curl_transport_failed:exit={exc.returncode}:{detail}') from None
            except subprocess.TimeoutExpired:
              raise RuntimeError('curl_transport_failed:timeout') from None
            wire=completed.stdout
          else:
            req=urllib.request.Request(endpoint,data=request_body,headers={"Authorization":"Bearer "+key,"Content-Type":"application/json"})
            open_request=opener.open if hasattr(opener,'open') else opener.urlopen
            with open_request(req,timeout=request_timeout) as r: wire=r.read()
          last_wire=wire
          trace_path.write_text(json.dumps({
              'status':'RECEIVED', 'index':i, 'attempt':attempt,
              'endpoint':endpoint, 'model':body.get('model'),
              'response_bytes':len(wire), 'received_at':time.time()
          }, ensure_ascii=False, indent=2)+'\n')
          v,u=parse(wire)
          if mode == "query":
            validate_query(v,validator_src)
          return v,u,wire,attempt,""
        except Exception as e:
          err=f"{type(e).__name__}:{e}"[:1000]
          # A deterministic query validator rejection can trigger a focused
          # repair. Memory chunks are validated only after they are merged, so
          # a transport/JSON failure is retried without semantic mutation.
          if last_wire is not None and isinstance(e,ValueError) and mode == "query":
            try:
              bad=json.loads(last_wire)["choices"][0]["message"]["content"]
              error_text=str(e)
              repair="Repair the JSON to satisfy the exact closed contract. Validator error: "+error_text+". Return the full corrected object."
              body["messages"]=[body["messages"][0],{"role":"assistant","content":bad},{"role":"user","content":repair}]
            except Exception: pass
          if attempt < request_attempts:
            time.sleep(min(8,2**attempt))
      return None,{},last_wire,request_attempts,err

    # Conversation-owned requests are clause-chunked below.  Keep the legacy
    # two-stage fast path for single turn/session requests; multi-chunk
    # conversations must not bypass the chunk merger or exceed model context.
    if mode == "memory" and args.memory_strategy in {"two_stage", "two_stage_reasonless", "two_stage_compact"} and len(chunk_sources) == 1:
      request_src=memory_request_view(src,dataset)
      selected_l1_system=compact_l1_system if args.memory_strategy in {"two_stage_reasonless", "two_stage_compact"} else l1_system
      selected_l2_system=compact_l2_system if args.memory_strategy == "two_stage_compact" else l2_system
      l1_value,l1_usage,l1_wire,l1_attempts,l1_err=invoke(request_src,src,selected_l1_system)
      if l1_value is None:
        rejected={"error":"l1_failed:"+l1_err}
        if l1_wire is not None:
          try: rejected["model_output"]=json.loads(l1_wire)["choices"][0]["message"]["content"]
          except Exception: rejected["model_output_unparsed"]=l1_wire.decode("utf-8",errors="replace")[:20000]
        rejected_path.write_text(json.dumps(rejected,ensure_ascii=False,indent=2)+"\n")
        return i,{"error":l1_err},{},"REJECTED",l1_attempts
      # Some local replicas wrap the object as {raw_id: {l1: ...}}.
      # Unwrap only this exact center-local wire shape.
      if (isinstance(l1_value,dict) and "l1" not in l1_value
          and len(l1_value) == 1):
        wrapped_key, wrapped_value = next(iter(l1_value.items()))
        if (str(wrapped_key) == str(src["center"]["raw_id"])
            and isinstance(wrapped_value,dict)
            and isinstance(wrapped_value.get("l1"),dict)):
          l1_value = wrapped_value
          l1_value.setdefault("center_raw_id", src["center"]["raw_id"])
          l1_value.setdefault("_best276_wire_repairs",[]).append("unwrapped_raw_id_l1_wrapper")
      frozen_l1=l1_value.get("l1") if isinstance(l1_value,dict) else None
      # Defensive wire-shape repair for small local models that sometimes
      # nest the requested object under center_raw_id or emit clause_units as
      # a malformed single-unit object.  Preserve valid nested L1; otherwise
      # fall back to source-backed unresolved units so the record remains
      # usable instead of being rejected before compilation.
      if isinstance(l1_value,dict) and isinstance(l1_value.get("center_raw_id"),dict):
        nested=l1_value["center_raw_id"]
        if isinstance(nested.get("l1"),dict):
          l1_value=nested
          frozen_l1=l1_value.get("l1")
          l1_value.setdefault("_best276_wire_repairs",[]).append("unwrapped_nested_center_raw_id")
      # Lossless identity repair: some local responses emit a valid L1 object
      # but omit only the top-level center_raw_id.  The request source is the
      # authority for this value; do not attempt to repair nested/recursive
      # objects whose L1 wrapper is absent or malformed.
      if (isinstance(l1_value,dict) and l1_value.get("center_raw_id") in (None,"")
          and isinstance(frozen_l1,dict)
          and isinstance(frozen_l1.get("clause_units"),list)):
        l1_value["center_raw_id"]=src["center"]["raw_id"]
        l1_value.setdefault("_best276_wire_repairs",[]).append("restored_missing_center_raw_id")
      if (isinstance(l1_value,dict) and l1_value.get("center_raw_id") == src["center"]["raw_id"]
          and isinstance(frozen_l1,dict)
          and not isinstance(frozen_l1.get("clause_units"),list)):
        # Do not promote hallucinated fields as facts.  Reconstruct one
        # source-backed unresolved unit per authoritative clause.
        fallback=[]
        for clause in src["center"].get("clauses",[]):
          cid=str(clause.get("clause_id")); text=str(clause.get("text") or "")
          fallback.append({"clause_id":cid,"units":[{"unit_id":f"{cid}::unit::1",
            "span":text,"status":"UNRESOLVED","assertion_ids":[],
            "reason":"MALFORMED_L1_WIRE"}]})
        l1_value["l1"]={"clause_units":fallback,"mentions":[]}
        frozen_l1=l1_value["l1"]
        l1_value.setdefault("_best276_wire_repairs",[]).append("reconstructed_malformed_l1")
      if (isinstance(l1_value,dict) and isinstance(frozen_l1,dict)
          and l1_value.get("center_raw_id") not in (None, src["center"]["raw_id"])):
        l1_value["center_raw_id"]=src["center"]["raw_id"]
        l1_value.setdefault("_best276_wire_repairs",[]).append("normalized_center_raw_id")
      if isinstance(l1_value,dict) and not isinstance(frozen_l1,dict):
        fallback=[]
        for clause in src["center"].get("clauses",[]):
          cid=str(clause.get("clause_id")); text=str(clause.get("text") or "")
          fallback.append({"clause_id":cid,"units":[{"unit_id":f"{cid}::unit::1",
            "span":text,"status":"UNRESOLVED","assertion_ids":[],
            "reason":"MALFORMED_L1_WIRE"}]})
        l1_value["center_raw_id"]=src["center"]["raw_id"]
        l1_value["l1"]={"clause_units":fallback,"mentions":[]}
        frozen_l1=l1_value["l1"]
        l1_value.setdefault("_best276_wire_repairs",[]).append("reconstructed_missing_l1")
      if l1_value.get("center_raw_id") != src["center"]["raw_id"] or not isinstance(frozen_l1,dict):
        err="l1_validation_failed: identity or l1 object missing"
        rejected_path.write_text(json.dumps({"error":err,"model_output":l1_value},ensure_ascii=False,indent=2)+"\n")
        return i,{"error":err},l1_usage,"REJECTED",l1_attempts
      # Remove non-object clause wrappers before any iteration; malformed
      # string entries otherwise raise worker-level AttributeError.
      frozen_l1["clause_units"]=[cu for cu in (frozen_l1.get("clause_units") or [])
                                  if isinstance(cu,dict)]
      for cu in frozen_l1["clause_units"]:
        raw_units=cu.get("units") if isinstance(cu.get("units"),list) else []
        clean=[]
        for u in raw_units:
          if not isinstance(u,dict): continue
          if isinstance(u.get("units"),list):
            clean.extend(x for x in u["units"] if isinstance(x,dict))
          else: clean.append(u)
        cu["units"]=[u for u in clean if str(u.get("span") or "").strip()]
        if not cu["units"]:
          text=next((str(x.get("text") or "") for x in request_src["center"].get("clauses",[])
                     if str(x.get("clause_id"))==str(cu.get("clause_id"))), "")
          cu["units"]=[{"unit_id":f"{cu.get('clause_id')}::unit::1","span":text,
                         "status":"UNRESOLVED","assertion_ids":[],"reason":"EMPTY_SPAN_DROPPED"}]
      frozen_l1.setdefault("mentions",[])
      # Some model routes preserve the semantic reason as a sentence (for
      # example, "Directly asserts that books possess awesome power.") while
      # emitting UNRESOLVED.  This is still a center-local classification,
      # not an inferred fact: promote only explicit assertion-style reasons.
      # Pronoun, ellipsis, relative-time and location uncertainty remain
      # UNRESOLVED and cannot feed L2.
      for clause_unit in frozen_l1.get("clause_units",[]):
        for unit in clause_unit.get("units",[]):
          reason=str(unit.get("reason") or "").strip().casefold()
          assertion_reason = (reason in {"direct proposition", "direct proposition unit"}
                              or reason.startswith("directly asserts")
                              or reason.startswith("asserts that")
                              or reason.startswith("asserts "))
          if unit.get("status")=="UNRESOLVED" and assertion_reason and str(unit.get("span") or "").strip():
            unit["status"]="ASSERTED"; unit["reason"]=None
      asserted_units=[
        unit
        for clause_unit in frozen_l1.get("clause_units",[])
        if isinstance(clause_unit,dict)
        for unit in clause_unit.get("units",[])
        if isinstance(unit,dict) and unit.get("status") == "ASSERTED"
      ]
      if not asserted_units:
        combined={"center_raw_id":src["center"]["raw_id"],"l1":frozen_l1,"l2_direct":[]}
        wire_repairs=l1_value.get("_best276_wire_repairs",[])
        if wire_repairs: combined["_best276_wire_repairs"]=sorted(set(wire_repairs))
        combined["_best276_optimizations"]=["skipped_l2_without_asserted_l1_units"]
        try:
          value=compile_memory(combined,src)
        except Exception as exc:
          err=f"l1_only_validation_failed:{type(exc).__name__}:{exc}"
          rejected_path.write_text(json.dumps({"error":err,"l1":l1_value},ensure_ascii=False,indent=2)+"\n")
          return i,{"error":err},l1_usage,"REJECTED",l1_attempts
        envelope={"model":os.environ.get('MEMORY_BUILD_MODEL') or os.environ.get('LLM_MODEL'),"choices":[{"index":0,"message":{"role":"assistant","content":canon(value)}}],"usage":l1_usage}
        p.write_bytes(canon(envelope).encode()+b"\n")
        meta_path.write_text(json.dumps({"requested_model":envelope["model"],"memory_strategy":args.memory_strategy,"prompt_sha256":prompt_identity,"ontology_sha256":ontology_identity},sort_keys=True)+"\n",encoding="utf8")
        rejected_path.unlink(missing_ok=True)
        return i,value,l1_usage,"NORMALIZED" if value.get("_best276_schema_repairs") else "PARSED",l1_attempts
      # Freeze the IDs that the downstream compiler actually consumes BEFORE
      # L2 sees them. Renaming only after L2 returns invalidates valid citations
      # such as c3u1 and silently discards the extracted assertions.
      for clause_unit in frozen_l1.get("clause_units", []):
        for unit_index, unit in enumerate(clause_unit.get("units", []), 1):
          unit["unit_id"] = f"{clause_unit['clause_id']}::unit::{unit_index}"
      l2_input={"center":request_src["center"],"frozen_l1":frozen_l1}
      l2_value,l2_usage,l2_wire,l2_attempts,l2_err=invoke(l2_input,src,selected_l2_system)
      usages={key_name:l1_usage.get(key_name,0)+l2_usage.get(key_name,0) for key_name in ("prompt_tokens","completion_tokens","total_tokens")}
      if l2_value is None:
        rejected_path.write_text(json.dumps({"error":"l2_failed:"+l2_err},ensure_ascii=False,indent=2)+"\n")
        return i,{"error":l2_err},usages,"REJECTED",l1_attempts+l2_attempts
      assertions=l2_value.get("l2_direct") if isinstance(l2_value.get("l2_direct"),list) else []
      unit_to_clause={
        str(unit.get("unit_id")):str(clause_unit.get("clause_id"))
        for clause_unit in frozen_l1.get("clause_units",[])
        if isinstance(clause_unit,dict)
        for unit in clause_unit.get("units",[])
        if isinstance(unit,dict) and unit.get("unit_id") is not None
      }
      # Small local models sometimes emit positional unit references ("0",
      # "1", ...), despite the prompt requiring canonical IDs such as
      # ``c1::unit::1``.  Recover only this lossless, center-local form; never
      # guess across clauses or across Raws.
      ordered_units=[]
      for clause_unit in frozen_l1.get("clause_units",[]):
        if not isinstance(clause_unit,dict):
          continue
        cid=str(clause_unit.get("clause_id"))
        clean_units=[]
        for unit_index,unit in enumerate(clause_unit.get("units",[]),1):
          if isinstance(unit,dict):
            # A common Llama failure is a trailing ASSERTED unit with
            # span=null/"".  It carries no source evidence and would make the
            # compiler reject the entire request.  Drop it; if it was the
            # only unit, retain a source-backed unresolved placeholder below.
            if not str(unit.get("span") or "").strip():
              unit["status"]="UNRESOLVED"
              unit["assertion_ids"]=[]
              unit["reason"]="EMPTY_SPAN_DROPPED"
              continue
            # compile_memory canonicalizes IDs in the order of surviving
            # units, so assign the compacted position here as well.
            canonical_unit_id=f"{cid}::unit::{len(clean_units)+1}"
            ordered_units.append(canonical_unit_id)
            unit["unit_id"]=canonical_unit_id
            clean_units.append(unit)
        if not clean_units:
          clause_text=next((str(x.get("text") or "") for x in request_src["center"].get("clauses",[])
                            if str(x.get("clause_id"))==cid), "")
          clean_units=[{"unit_id":f"{cid}::unit::1", "span":clause_text,
                        "status":"UNRESOLVED", "assertion_ids":[],
                        "reason":"EMPTY_SPAN_DROPPED"}]
        clause_unit["units"]=clean_units
      # Rebuild after empty-span filtering so L2 provenance is checked against
      # the exact canonical IDs that will be compiled.
      unit_to_clause={
        str(unit.get("unit_id")):str(clause_unit.get("clause_id"))
        for clause_unit in frozen_l1.get("clause_units",[])
        if isinstance(clause_unit,dict)
        for unit in clause_unit.get("units",[])
        if isinstance(unit,dict) and unit.get("unit_id") is not None
      }
      # Llama commonly emits local numeric references as 1-based IDs
      # ("1", "2", ...), while older prompts/models sometimes emit the
      # zero-based form ("0", "1", ...).  Prefer the canonical 1-based
      # interpretation and retain only the unambiguous zero alias "0".
      # Canonical IDs such as c1::unit::1 are handled unchanged below.
      positional_units={}
      if ordered_units:
        positional_units["0"]=ordered_units[0]
        for unit_index,uid in enumerate(ordered_units,1):
          positional_units[str(unit_index)]=uid
          positional_units[f"u{unit_index}"]=uid
      by_unit={}
      valid_assertions=[]
      for assertion_index,assertion in enumerate(assertions,1):
        if not isinstance(assertion,dict): continue
        assertion.setdefault("assertion_id",f"a{assertion_index}")
        supplied_provenance=assertion.get("provenance") if isinstance(assertion.get("provenance"),dict) else {}
        raw_ids_value=assertion.pop("unit_ids",None)
        if raw_ids_value is None:
          raw_ids_value=supplied_provenance.get("unit_ids",[])
        raw_unit_ids=[str(unit_id) for unit_id in (raw_ids_value or [])]
        unit_ids=[]
        for unit_id in raw_unit_ids:
          mapped=positional_units.get(unit_id, unit_id)
          if mapped != unit_id:
            assertion.setdefault("_best276_wire_repairs",[]).append("mapped_positional_unit_id")
          unit_ids.append(mapped)
        # Never allow an L2 hallucinated/cross-request unit ID to poison the
        # whole record.  Keep only IDs present in the frozen L1; an assertion
        # with no surviving evidence is discarded below.
        unit_ids=[uid for uid in unit_ids if uid in unit_to_clause]
        if not unit_ids:
          assertion.setdefault("_best276_wire_repairs",[]).append("dropped_unknown_unit_assertion")
          continue
        for unit_id in unit_ids:
          by_unit.setdefault(str(unit_id),[]).append(assertion["assertion_id"])
        clause_ids=list(dict.fromkeys(unit_to_clause[unit_id] for unit_id in unit_ids if unit_id in unit_to_clause))
        provenance=assertion.get("provenance") if isinstance(assertion.get("provenance"),dict) else {}
        provenance["raw_ids"]=[src["center"]["raw_id"]]
        provenance["clause_ids"]=clause_ids or provenance.get("clause_ids",[])
        provenance["unit_ids"]=unit_ids
        assertion["provenance"]=provenance
        if args.memory_strategy == "two_stage_compact":
          assertion.setdefault("roles",{})
          assertion.setdefault("scope",{})
          assertion.setdefault("occurrence_local_id",None)
          assertion["authority"]="DIALOGUE_TEXT"
        valid_assertions.append(assertion)
      assertions=valid_assertions
      for clause_unit in frozen_l1.get("clause_units",[]):
        for unit in clause_unit.get("units",[]):
          linked_ids=by_unit.get(str(unit.get("unit_id")),[])
          if linked_ids:
            # The L1 call is instructed to leave assertion_ids empty.  If the
            # validated L2 call explicitly cites an otherwise unresolved unit,
            # that link is deterministic evidence that the unit is direct and
            # can be promoted for the final L1/L2 contract.  Never promote a
            # NON_PROPOSITIONAL unit.
            if unit.get("status") == "UNRESOLVED":
              unit["status"]="ASSERTED"
              unit["reason"]=None
            unit["assertion_ids"]=linked_ids
          elif unit.get("status") == "ASSERTED":
            unit["assertion_ids"]=[]
      wire_repairs=(l1_value.get("_best276_wire_repairs",[]) + l2_value.get("_best276_wire_repairs",[]))
      combined={"center_raw_id":src["center"]["raw_id"],"l1":frozen_l1,"l2_direct":assertions}
      # Final defensive normalization: some chat templates place the unit
      # list back under provenance after the stage loop.  Apply the same
      # center-local positional mapping immediately before compilation.
      for assertion in combined["l2_direct"]:
        prov=assertion.get("provenance") or {}
        ids=prov.get("unit_ids") if isinstance(prov,dict) else None
        if isinstance(ids,list):
          prov["unit_ids"]=[positional_units.get(str(uid),str(uid)) for uid in ids]
          assertion["provenance"]=prov
      linked_by_unit={}
      for assertion in combined["l2_direct"]:
        aid=str(assertion.get("assertion_id"))
        for uid in (assertion.get("provenance") or {}).get("unit_ids",[]):
          linked_by_unit.setdefault(str(uid),[]).append(aid)
      for clause_unit in frozen_l1.get("clause_units",[]):
        for unit in clause_unit.get("units",[]):
          links=linked_by_unit.get(str(unit.get("unit_id")),[])
          if links and unit.get("status") != "NON_PROPOSITIONAL":
            unit["status"]="ASSERTED"; unit["reason"]=None; unit["assertion_ids"]=links
      # Final provenance fence immediately before compilation.  This catches
      # any L2 assertion that still carries a stale unit reference after the
      # positional-ID normalization above.
      final_unit_ids={str(unit.get("unit_id"))
                      for clause_unit in frozen_l1.get("clause_units",[])
                      for unit in clause_unit.get("units",[])
                      if isinstance(unit,dict) and unit.get("unit_id") is not None}
      combined["l2_direct"]=[
        assertion for assertion in combined["l2_direct"]
        if set(map(str,(assertion.get("provenance") or {}).get("unit_ids",[]))) <= final_unit_ids
      ]
      nonprop_ids={str(unit.get("unit_id"))
                   for clause_unit in frozen_l1.get("clause_units",[])
                   for unit in clause_unit.get("units",[])
                   if unit.get("status")=="NON_PROPOSITIONAL"}
      combined["l2_direct"]=[
        assertion for assertion in combined["l2_direct"]
        if not (set(map(str,(assertion.get("provenance") or {}).get("unit_ids",[]))) & nonprop_ids)
      ]
      # Last provenance fence: retain only assertions whose unit IDs are
      # present and whose clause prefix agrees with the frozen L1 clause.
      checked=[]
      for assertion in combined["l2_direct"]:
        prov=assertion.get("provenance") or {}
        ids=[str(x) for x in (prov.get("unit_ids") or [])]
        if not ids or not set(ids) <= final_unit_ids:
          continue
        if any(uid.split("::unit::",1)[0] not in set(prov.get("clause_ids") or []) for uid in ids):
          continue
        checked.append(assertion)
      combined["l2_direct"]=checked
      if wire_repairs: combined["_best276_wire_repairs"]=sorted(set(wire_repairs))
      try:
        value=compile_memory(combined,src)
      except Exception as exc:
        err=f"two_stage_validation_failed:{type(exc).__name__}:{exc}"
        rejected_path.write_text(json.dumps({"error":err,"l1":l1_value,"l2":l2_value,"combined":combined},ensure_ascii=False,indent=2)+"\n")
        return i,{"error":err},usages,"REJECTED",l1_attempts+l2_attempts
      envelope={"model":os.environ.get('MEMORY_BUILD_MODEL') or os.environ.get('LLM_MODEL'),"choices":[{"index":0,"message":{"role":"assistant","content":canon(value)}}],"usage":usages}
      p.write_bytes(canon(envelope).encode()+b"\n")
      meta_path.write_text(json.dumps({"requested_model":envelope["model"],"memory_strategy":args.memory_strategy,"prompt_sha256":prompt_identity,"ontology_sha256":ontology_identity},sort_keys=True)+"\n",encoding="utf8")
      rejected_path.unlink(missing_ok=True)
      return i,value,usages,"NORMALIZED" if value.get("_best276_schema_repairs") else "PARSED",l1_attempts+l2_attempts

    if mode == "memory" and len(chunk_sources) > 1:
      values=[]; usages={k:0 for k in ("prompt_tokens","completion_tokens","total_tokens")}; attempts=0
      chunk_errors=[]
      for chunk_index,chunk_src in enumerate(chunk_sources):
        # Multi-chunk two-stage must remain genuinely two-stage.  The old path
        # called the joint memory prompt once per chunk, which defeated the
        # compact strategy and caused local 8B models to emit schema fallbacks.
        l1_value,l1_usage,l1_wire,l1_attempts,l1_err=invoke(
          chunk_src,chunk_src,compact_l1_system)
        attempts += l1_attempts
        if l1_value is None or not isinstance(l1_value.get("l1"),dict):
          chunk_errors.append({"chunk":chunk_index,"stage":"l1","error":l1_err or "missing_l1"})
          continue
        frozen_l1=l1_value["l1"]
        l2_request={"center":chunk_src["center"],"frozen_l1":frozen_l1}
        l2_value,l2_usage,l2_wire,l2_attempts,l2_err=invoke(
          l2_request,chunk_src,compact_l2_system)
        attempts += l2_attempts
        if l2_value is None or not isinstance(l2_value.get("l2_direct"),list):
          chunk_errors.append({"chunk":chunk_index,"stage":"l2","error":l2_err or "missing_l2"})
          continue
        combined={"center_raw_id":chunk_src["center"]["raw_id"],
                  "l1":frozen_l1,"l2_direct":l2_value["l2_direct"]}
        values.append(combined)
        for key_name in usages:
          usages[key_name] += l1_usage.get(key_name,0)+l2_usage.get(key_name,0)
      if chunk_errors:
        err="chunk_failures:"+json.dumps(chunk_errors,ensure_ascii=False)
        rejected_path.write_text(json.dumps({"error":err},ensure_ascii=False,indent=2)+"\n")
        return i,{"error":err},usages,"REJECTED",attempts
      try:
        value=merge_memory_chunks(values,chunk_sources,src)
      except Exception as exc:
        err=f"merge_validation_failed:{type(exc).__name__}:{exc}"
        (out/"rejected"/f"{i:04d}.json").write_text(json.dumps({"error":err},ensure_ascii=False,indent=2)+"\n")
        return i,{"error":err},usages,"REJECTED",attempts
      envelope={"model":os.environ.get('MEMORY_BUILD_MODEL') or os.environ.get('LLM_MODEL') or 'gpt-5.4-mini',"choices":[{"index":0,"message":{"role":"assistant","content":canon(value)}}],"usage":usages}
      p.write_bytes(canon(envelope).encode()+b"\n")
      meta_path.write_text(json.dumps({"requested_model":envelope["model"],"memory_strategy":args.memory_strategy,"prompt_sha256":prompt_identity,"ontology_sha256":ontology_identity},sort_keys=True)+"\n",encoding="utf8")
      rejected_path.unlink(missing_ok=True)
      return i,value,usages,"NORMALIZED" if value.get("_best276_schema_repairs") else "PARSED",attempts

    value,usage_item,wire,attempts,err=invoke(chunk_sources[0],src)
    if value is not None:
      try:
        value=compile_memory(value,src) if mode=="memory" else value
        p.write_bytes(wire+b"\n")
        meta_path.write_text(json.dumps({"requested_model":os.environ.get('MEMORY_BUILD_MODEL') or os.environ.get('LLM_MODEL') or 'gpt-5.4-mini',"memory_strategy":args.memory_strategy,"prompt_sha256":prompt_identity,"ontology_sha256":ontology_identity},sort_keys=True)+"\n",encoding="utf8")
        rejected_path.unlink(missing_ok=True)
        result_status="NORMALIZED" if mode=="memory" and value.get("_best276_schema_repairs") else "PARSED"
        return i,value,usage_item,result_status,attempts
      except Exception as exc:
        err=f"validation_failed:{type(exc).__name__}:{exc}"
    rejected={"error":err}
    if wire is not None:
      try: rejected["model_output"]=json.loads(wire)["choices"][0]["message"]["content"]
      except Exception: rejected["model_output_unparsed"]=wire.decode("utf-8",errors="replace")[:20000]
    rejected_path.write_text(json.dumps(rejected,ensure_ascii=False,indent=2)+"\n"); return i,{"error":err},{},"REJECTED",attempts
  indexed=enumerate(row_iter(request_path))
  if args.only_index is not None:
    # ``indexed`` already yields ``(absolute_index, source)``.  A second
    # enumerate here wrapped that pair as the source and caused an immediate
    # worker-side TypeError before any API request was made.
    indexed=((i,src) for i,src in indexed if i == args.only_index)
    total=1
  elif sparse_indices is not None:
    indexed=((i,src) for i,src in indexed if i in sparse_indices)
    total=len(sparse_indices)
  elif args.start_index is not None or args.max_items is not None:
    start=max(0,args.start_index or 0)
    stop=(start+args.max_items) if args.max_items is not None else total
    indexed=((i,src) for i,src in indexed if start <= i < stop)
    total=max(0,min(total,stop)-start)
  elif args.shard_count > 1:
    indexed=((i,src) for i,src in indexed if i % args.shard_count == args.shard_index)
    total=max(0,(total-args.shard_index+args.shard_count-1)//args.shard_count)
  range_suffix=(f".range-{max(0,args.start_index or 0):06d}-n{args.max_items or total:06d}" if args.start_index is not None or args.max_items is not None else "")
  shard_suffix=(f".shard-{args.shard_index:03d}-of-{args.shard_count:03d}" if args.shard_count > 1 else (f".probe-{args.only_index:06d}" if args.only_index is not None else (f".repair-n{len(sparse_indices):06d}" if sparse_indices is not None else range_suffix)))
  base_name="QUERY_KEYS" if mode=="query" else "L1_L2_MEMORY"
  target=out/((f"DEBUG_{args.only_index:04d}.jsonl") if args.only_index is not None else f"{base_name}{shard_suffix}.jsonl")
  progress_path=out/f"PROGRESS{shard_suffix}.json"
  manifest_path=run_root / "memory" / f"V2_STATUS_MANIFEST{shard_suffix}.jsonl"
  target_handle=target.open("w",encoding="utf8")
  manifest_handle=(manifest_path.open("w",encoding="utf8") if mode == "memory" else None)
  started=time.time(); completed_count=failed_count=reused_count=normalized_count=0; usage={k:0 for k in ("prompt_tokens","completion_tokens","total_tokens")}; seen_count=0
  last_progress=0.0
  def write_progress(state="RUNNING", in_flight=0):
    nonlocal last_progress
    now=time.time(); last_progress=now
    payload={"status":state,"mode":mode,"requested":total,"seen":seen_count,"completed":completed_count,"failed":failed_count,"reused":reused_count,"normalized":normalized_count,"in_flight":in_flight,"remaining":max(0,total-seen_count),"elapsed_seconds":round(now-started,1),"updated_at":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime(now))}
    tmp=progress_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf8"); tmp.replace(progress_path)
    print(json.dumps({"progress":payload},ensure_ascii=False),flush=True)
  def manifest_status(value, source):
    if self_contained: return "SELF_CONTAINED_RAW"
    if dataset == "longmemeval" and granularity == "session": return "SELF_CONTAINED_LLM"
    if dataset == "longmemeval" and granularity in {"turn", "conversation"}: return contextual_status(value, source)
    return "CONTEXT_REPARSE_REQUIRED"
  write_progress("STARTING")
  executor=concurrent.futures.ThreadPoolExecutor(max_workers=args.workers)
  try:
    batch=[]; batch_size=configured_batch_size
    def flush_batch(batch):
      nonlocal completed_count,failed_count,reused_count,normalized_count,seen_count,usage,last_progress
      if not batch: return
      futures={executor.submit(call,pair):pair for pair in batch}; results=[]
      pending=set(futures)
      while pending:
        done,pending=concurrent.futures.wait(pending,timeout=10,return_when=concurrent.futures.FIRST_COMPLETED)
        if not done:
          write_progress("RUNNING",len(pending)); continue
        for future in done:
          pair=futures[future]
          try: result=future.result()
          except Exception as exc:
            error=f"worker_exception:{type(exc).__name__}:{exc}"
            rejected_path=out/"rejected"/f"{pair[0]:04d}.json"
            rejected_path.write_text(json.dumps({
              "index":pair[0],
              "error":error,
              "exception_type":type(exc).__name__,
            },ensure_ascii=False,indent=2)+"\n",encoding="utf8")
            result=(pair[0],{"error":error},{},"REJECTED",0)
          results.append(result)
          _,value,usage_item,result_status,_=result
          seen_count+=1
          if result_status == "REJECTED": failed_count+=1
          else:
            completed_count+=1; reused_count += result_status == "REUSED"; normalized_count += result_status == "NORMALIZED"
            for key in usage: usage[key]+=usage_item.get(key,0)
          if mode == "memory" and result_status != "REJECTED" and value.get("center_raw_id"):
            manifest_handle.write(canon({"raw_id":value["center_raw_id"],"status":manifest_status(value,pair[1])})+"\n")
          if result_status != "REJECTED": target_handle.write(canon(value)+"\n")
        target_handle.flush()
        if manifest_handle: manifest_handle.flush()
        write_progress("RUNNING",len(pending))
      # API completion order is nondeterministic.  The persisted wire files are
      # keyed by index; the derived JSONL view is intentionally append-safe and
      # keyed by ``center_raw_id`` downstream, so completion order is harmless.
    for pair in indexed:
      batch.append(pair)
      if len(batch) >= batch_size:
        flush_batch(batch); batch=[]
    flush_batch(batch)
  finally:
    executor.shutdown(wait=True)
    target_handle.close()
    if manifest_handle: manifest_handle.close()
  status="COMPLETE" if failed_count == 0 and seen_count == total else "PARTIAL"
  write_progress(status,0)
  rec={"status":status,"mode":mode,"requested":seen_count,"expected":total,"completed":completed_count,"failed":failed_count,"reused":reused_count,"normalized":normalized_count,"usage":usage,"gold_visible":False,"only_index":args.only_index,"start_index":args.start_index,"max_items":args.max_items,"memory_strategy":args.memory_strategy,"requested_model":os.environ.get('MEMORY_BUILD_MODEL') or os.environ.get('LLM_MODEL'),"prompt_sha256":prompt_identity,"ontology_sha256":ontology_identity,"output_sha256":sha(target)}
  receipt_path=out/f"RECEIPT{shard_suffix}.json"
  (receipt_path).write_text(json.dumps(rec,indent=2)+"\n")
  if mode == "memory" and rec.get("status") == "COMPLETE":
    # Contextual memory consumes an explicit status manifest.  The historical
    # bundle did not materialize it during fresh runs, which made the next
    # stage silently see zero targets.  LongMemEval session records are already
    # self-contained; turn-level runs retain the historical reparse contract.
    status = "SELF_CONTAINED_RAW" if self_contained else ("SELF_CONTAINED_LLM" if dataset == "longmemeval" and granularity == "session" else "MIXED_CONTEXT_GATE")
    manifest_path = run_root / "memory" / "V2_STATUS_MANIFEST.jsonl"
    # ``got`` contains the worker tuple ``(index, value, usage, status,
    # attempts)``; use the parsed value (not the tuple itself) when emitting
    # the contextual contract.  The old code only surfaced after a complete
    # API batch, so a successful memory stage could still fail at its final
    # receipt write.
    # The manifest is streamed during processing so it remains useful after a
    # crash.  On a complete run it already contains exactly one line/request.
    rec["context_manifest_status"] = status
    rec["memory_backend"] = "self_contained_raw" if self_contained else "llm_extraction"
    receipt_path.write_text(json.dumps(rec,indent=2)+"\n")
  if rec.get("status") != "COMPLETE":
    # Do not let later retrieval stages silently evaluate a partial corpus.
    raise SystemExit(2)
  print(json.dumps(rec,indent=2))
if __name__=="__main__": main()
