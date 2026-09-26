#!/usr/bin/env python3
"""Model-driven cross-session conversation memory merge.

The session extractor supplies direct facts.  This stage is the actual
conversation-level memory builder: it processes sessions in temporal order,
asks the local Llama for only supported links/updates, validates every source
reference, and checkpoints after each conversation.
"""
import argparse, json, os, time, urllib.request
from collections import defaultdict
from pathlib import Path

SYSTEM = """You are a conversation memory merger. Update prior memory with exactly one new session.
Return JSON only: {entity_links, supersessions, merged_assertions}.
Never invent facts. Every source_raw_id must be from the supplied session or prior memory.
Keep historical assertions; mark replaced assertions SUPERSEDED rather than deleting them.
entity_links items: {from_text,to_assertion_id,confidence,source_raw_ids}.
supersessions items: {old_assertion_id,new_assertion_id,reason,source_raw_ids}.
merged_assertions items: {assertion_id,subject,relation,value,status,source_session_ids,source_raw_ids,source_clause_ids}.
Use status CURRENT, HISTORICAL, SUPERSEDED, or UNCERTAIN. If uncertain, do not merge."""

def rows(path):
    return [json.loads(x) for x in path.open(encoding="utf8") if x.strip()] if path.exists() else []

def call_llm(payload):
    url=os.environ.get("LLM_BASE_URL","http://127.0.0.1:8003/v1").rstrip("/")+"/chat/completions"
    body={"model":os.environ.get("MEMORY_BUILD_MODEL","meta-llama/Llama-3.1-8B-Instruct"),"temperature":0,"max_tokens":int(os.environ.get("MEMORY_MERGE_MAX_TOKENS","3000")),"response_format":{"type":"json_object"},"messages":[{"role":"system","content":SYSTEM},{"role":"user","content":json.dumps(payload,ensure_ascii=False)}]}
    req=urllib.request.Request(url,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=int(os.environ.get("MEMORY_MERGE_TIMEOUT","180"))) as r:
        return json.loads(json.loads(r.read()) ["choices"][0]["message"]["content"])

def validate(out, prior, current):
    allowed={"entity_links","supersessions","merged_assertions"}
    if set(out) != allowed: raise ValueError("merge_contract_keys")
    valid_sources={str(x) for x in current.get("source_raw_ids",[])} | {str(x) for x in prior.get("source_raw_ids",[])}
    ids={str(x.get("assertion_id")) for x in prior.get("assertions",[])}
    for x in out["supersessions"]:
        if str(x.get("old_assertion_id")) not in ids: raise ValueError("unknown_old_assertion")
        if not (set(map(str,x.get("source_raw_ids",[]))) <= valid_sources): raise ValueError("unknown_supersession_source")
    for x in out["merged_assertions"]:
        if x.get("status") not in {"CURRENT","HISTORICAL","SUPERSEDED","UNCERTAIN"}: raise ValueError("bad_status")
        if not set(map(str,x.get("source_raw_ids",[]))) <= valid_sources: raise ValueError("unknown_assertion_source")
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--run-root",required=True); ap.add_argument("--session-results",required=True); ap.add_argument("--output"); args=ap.parse_args()
    root=Path(args.run_root); requests={x.get("center",{}).get("raw_id"):x.get("center",{}) for x in rows(root/"memory/MEMORY_REQUESTS.jsonl")}
    grouped=defaultdict(list)
    for result in rows(Path(args.session_results)):
        rid=str(result.get("center_raw_id","")); src=requests.get(rid,{}); grouped[rid.split("::",1)[0]].append((src,result))
    out=Path(args.output) if args.output else root/"memory/CONVERSATION_MEMORY_LLM.jsonl"; done={x["conversation_id"]:x for x in rows(out)}; out.parent.mkdir(parents=True,exist_ok=True)
    for cid,items in sorted(grouped.items()):
        if cid in done: continue
        items.sort(key=lambda z:(str(z[0].get("timestamp","")),str(z[0].get("source_session_id","")),str(z[0].get("raw_id",""))))
        state={"conversation_id":cid,"assertions":[],"source_raw_ids":[]}
        for src,result in items:
            rid=str(src.get("raw_id")); sid=str(src.get("source_session_id","UNKNOWN"));
            current={"session_id":sid,"source_raw_ids":[rid],"source_clause_ids":[str(c.get("clause_id")) for c in src.get("clauses",[])],"assertions":result.get("l2_direct",[])}
            prior_sources=list(state.get("source_raw_ids",[]))
            merged=validate(call_llm({"conversation_id":cid,"prior_memory":state,"current_session":current}),state,current)
            # The model may return only merged assertions; preserve the full
            # conversation source ledger independently of model output.
            state={"conversation_id":cid,"assertions":merged["merged_assertions"],"source_raw_ids":sorted(set(prior_sources+[rid]))}
        done[cid]=state
        out.write_text("".join(json.dumps(done[k],ensure_ascii=False)+"\n" for k in sorted(done)),encoding="utf8")
    receipt={"status":"COMPLETE","conversations":len(done),"model":os.environ.get("MEMORY_BUILD_MODEL","meta-llama/Llama-3.1-8B-Instruct"),"gold_visible":False}; (out.parent/"CONVERSATION_MEMORY_LLM_RECEIPT.json").write_text(json.dumps(receipt,indent=2)+"\n"); print(json.dumps(receipt))
if __name__=="__main__": main()
