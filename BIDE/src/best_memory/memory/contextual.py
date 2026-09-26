#!/usr/bin/env python3
"""Build the complete, question-independent contextual-memory contract."""
import concurrent.futures, hashlib, json, os, subprocess, time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("BEST276_ROOT", HERE.parents[3]))
RUN = Path(os.environ.get("BEST276_RUN_ROOT", ROOT / "runs/default"))
DATA = Path(os.environ.get("BEST276_DATA", RUN / "data_normalized.json"))
MANIFEST = RUN / "memory/V2_STATUS_MANIFEST.jsonl"
OUT = RUN / "memory/CONTEXTUAL_V41.jsonl"
WIRE = RUN / "memory/CONTEXTUAL_V41_WIRE"
OUT.parent.mkdir(parents=True, exist_ok=True); WIRE.mkdir(parents=True, exist_ok=True)

URL = os.environ.get("CONTEXTUAL_API_URL", "")
MODEL = os.environ.get("CONTEXTUAL_MODEL", "gpt-5.4-mini")
KEY = os.environ.get(os.environ.get("CONTEXTUAL_AUTH_ENV", "GENERATION_API_KEY"), "")
TIMEOUT = int(os.environ.get("CONTEXTUAL_API_TIMEOUT", "180"))
ATTEMPTS = int(os.environ.get("CONTEXTUAL_API_ATTEMPTS", "5"))
FULL_CONTEXT = os.environ.get("CONTEXTUAL_ALL_TURNS", "0") == "1"
PROTOCOL = "contextual-memory-v4.1-timestamp-20260922" + ("-all" if FULL_CONTEXT else "-manifest")

SYSTEM = r'''Resolve the CENTER Raw using only the supplied dialogue context. Neighbor Raw records are context, never automatically claims made by CENTER. Do not use questions, answers, labels, or outside knowledge.

Return one JSON object with exactly these top-level fields:
center_summary: a context-resolved paraphrase of only CENTER's contribution
context_dependency: NONE|COREFERENCE|ELLIPSIS|CONFIRMATION|DEICTIC|OTHER
antecedent_raw_ids: non-center raw_ids needed to interpret CENTER
speech_acts: a list of objects with:
  act: ASSERTION|CONFIRMATION|DENIAL|ANSWER|CORRECTION|QUESTION|REACTION|REQUEST|NON_PROPOSITIONAL
  center_quote: exact supporting span from CENTER
  antecedent_raw_ids: non-center raw_ids needed by this act
  assertion: null or {subject, relation, value, event, time, modality, polarity}
  resolved_proposition: null or {subject, relation, value, event, time, modality, polarity, source_raw_ids}
occurrence_links: a list of {owner, event, relation, antecedent_raw_ids}

Rules:
- source_raw_ids must contain CENTER and may contain only supplied raw_ids.
- CONFIRMATION, DENIAL, ANSWER, CORRECTION and context-resolved propositions require an antecedent.
- assertion records only content literally contributed by CENTER.
- resolved_proposition may resolve pronouns/ellipsis using antecedents but must not copy unrelated neighbor facts.
- occurrence_links identify genuinely shared events/entities across CENTER and antecedents; coreference alone is insufficient.
- QUESTION, REACTION, REQUEST and NON_PROPOSITIONAL normally have null assertion.
- Use [] and null when absent. Return JSON only.'''

ALLOWED_ACTS = {"ASSERTION","CONFIRMATION","DENIAL","ANSWER","CORRECTION","QUESTION","REACTION","REQUEST","NON_PROPOSITIONAL"}
ALLOWED_DEPS = {"NONE","COREFERENCE","ELLIPSIS","CONFIRMATION","DEICTIC","OTHER"}

def rows(path):
    if not path.exists(): return []
    return [json.loads(line) for line in path.open(encoding="utf8") if line.strip()]

raw, sessions = {}, {}
for conv in json.load(DATA.open(encoding="utf8")):
    cid, src, day = conv["sample_id"], conv["conversation"], 1
    while f"session_{day}" in src:
        session = src[f"session_{day}"]; sessions[(cid, day)] = session
        for index, turn in enumerate(session, 1):
            raw[f"{cid}::D{day}:{index}"] = {"speaker":turn.get("speaker",""), "text":turn.get("text",""), "timestamp":turn.get("timestamp") or src.get(f"session_{day}_date_time", "")}
        day += 1

manifest = {x["raw_id"]:x for x in rows(MANIFEST)}
targets = [x for x in manifest.values() if x.get("status") == "CONTEXT_REPARSE_REQUIRED"]
if FULL_CONTEXT:
    targets = [{"raw_id":rid} for rid in raw]
latest = {x["raw_id"]:x for x in rows(OUT)}
done = {rid for rid,x in latest.items() if x.get("status") == "VALIDATED" and x.get("protocol") == PROTOCOL}
def context_needed(item):
    """Honor the Memory manifest; a cheap extra gate is opt-in, never silent."""
    if FULL_CONTEXT or os.environ.get('CONTEXTUAL_USE_LENGTH_GATE','0')!='1': return True
    rid=item["raw_id"]; text=raw.get(rid,{}).get("text") or ""; s=" ".join(text.split()); n=len(s.split())
    cid,rest=rid.split("::"); day,turn=map(int,rest[1:].split(":")); session=sessions.get((cid,day),[])
    prev=" ".join((x.get("text") or "") for x in session[max(0,turn-4):turn-1])
    if not prev: return False
    import re
    if re.match(r'^\s*(yes|yeah|yep|no|nope|exactly|right|correct|same|me too|so do i|i agree)\b',s,re.I) and n<=16: return True
    if n<=6 and not re.search(r'\b(what|which|where|when|who|how|why)\b',s,re.I): return True
    if re.search(r'\b(he|she|they|them|this|that|these|those|here|there|his|her|their|it)\b',s,re.I) and n<=16:
        toks=set(re.findall(r'[a-z]{3,}',s.lower())); pt=set(re.findall(r'[a-z]{3,}',prev.lower()))
        if toks & pt or re.match(r'^\s*(he|she|they|it|this|that|these|those|there|here)\b',s,re.I): return True
    if re.search(r'\b(what|which|where|when|who|how|why)\b',s,re.I) and n<=10 and not re.search(r'\b(you|your|did|do|are|is|was|were)\b',s,re.I): return True
    return False

todo = [x for x in targets if x["raw_id"] not in done and context_needed(x)]
# Materialize non-triggered records without an API call.
for item in targets:
    rid=item["raw_id"]
    if rid not in done and not context_needed(item):
        latest[rid]={"raw_id":rid,"status":"PASSTHROUGH","validation_errors":[],
                     "contextual_v4":{"center_summary":raw.get(rid,{}).get("text", ""),
                     "context_dependency":"NONE","antecedent_raw_ids":[],
                     "speech_acts":[],"occurrence_links":[]},"model":None,
                     "protocol":PROTOCOL,"gold_visible":False}

def validate(result, center, supplied):
    errors=[]
    if set(result) != {"center_summary","context_dependency","antecedent_raw_ids","speech_acts","occurrence_links"}: errors.append("KEY_CONTRACT")
    if result.get("context_dependency") not in ALLOWED_DEPS: errors.append("CONTEXT_DEPENDENCY_ENUM")
    supplied=set(supplied); top_ants=result.get("antecedent_raw_ids")
    if not isinstance(top_ants,list): errors.append("TYPE_ANTECEDENTS"); top_ants=[]
    if center in top_ants or any(x not in supplied for x in top_ants): errors.append("INVALID_ANTECEDENT")
    acts=result.get("speech_acts")
    if not isinstance(acts,list): errors.append("TYPE_SPEECH_ACTS"); acts=[]
    for i,act in enumerate(acts):
        if not isinstance(act,dict): errors.append(f"ACT_{i}_TYPE"); continue
        if act.get("act") not in ALLOWED_ACTS: errors.append(f"ACT_{i}_ENUM")
        ants=act.get("antecedent_raw_ids") or []
        if center in ants or any(x not in supplied for x in ants): errors.append(f"ACT_{i}_ANTECEDENT")
        if act.get("act") in {"CONFIRMATION","DENIAL","ANSWER","CORRECTION"} and not ants: errors.append(f"ACT_{i}_MISSING_ANTECEDENT")
        prop=act.get("resolved_proposition")
        if prop:
            sources=prop.get("source_raw_ids") or []
            if center not in sources or any(x != center and x not in supplied for x in sources): errors.append(f"ACT_{i}_SOURCES")
    links=result.get("occurrence_links")
    if not isinstance(links,list): errors.append("TYPE_OCCURRENCE_LINKS"); links=[]
    for i,link in enumerate(links):
        ants=(link or {}).get("antecedent_raw_ids") or []
        if not ants or any(x not in supplied for x in ants): errors.append(f"OCCURRENCE_{i}_ANTECEDENT")
    return sorted(set(errors))

def normalize(result, supplied, center=None):
    """Apply only lossless/safe schema repairs; never invent a relation."""
    if result.get("context_dependency") == "QUESTION": result["context_dependency"] = "OTHER"
    if result.get("context_dependency") == "NON_PROPOSITIONAL": result["context_dependency"] = "NONE"
    # An intra-turn antecedent is not a cross-Raw edge. Keep CENTER in the
    # proposition's source_raw_ids, but never in neighbor-only antecedent lists.
    if center is not None:
        for item in [result,*(result.get("speech_acts") or [])]:
            if isinstance(item,dict) and isinstance(item.get("antecedent_raw_ids"),list):
                item["antecedent_raw_ids"]=[r for r in item["antecedent_raw_ids"] if r!=center]
    supplied=set(supplied)
    links=result.get("occurrence_links")
    if isinstance(links,list):
        result["occurrence_links"]=[x for x in links if isinstance(x,dict) and x.get("antecedent_raw_ids") and all(a in supplied for a in x["antecedent_raw_ids"])]
    return result

def request(item):
    rid=item["raw_id"]; cid=rid.split("::")[0]; day,turn=map(int,rid.split("::D",1)[1].split(":")); session=sessions[(cid,day)]
    context=[]
    for index in range(max(1,turn-3), min(len(session),turn+2)+1):
        rr=f"{cid}::D{day}:{index}"; context.append({"raw_id":rr,**raw[rr],"is_center":rr==rid})
    supplied=[x["raw_id"] for x in context if not x["is_center"]]
    user_content=json.dumps({"center_raw_id":rid,"context":context},ensure_ascii=False)
    last=""; previous_result=None
    for attempt in range(ATTEMPTS):
        try:
            messages=[{"role":"system","content":SYSTEM},{"role":"user","content":user_content}]
            if last:
                if previous_result is not None:
                    messages.append({"role":"assistant","content":json.dumps(previous_result,ensure_ascii=False)})
                messages.append({"role":"user","content":"Your previous output failed validation: "+last+". Return the complete corrected JSON object. context_dependency must be one of NONE, COREFERENCE, ELLIPSIS, CONFIRMATION, DEICTIC, OTHER. Antecedent arrays must exclude CENTER and contain only supplied neighbor IDs. Do not omit required fields; remove an occurrence link if it has no valid antecedent."})
            body={"model":MODEL,"temperature":0,"max_tokens":int(os.environ.get("CONTEXTUAL_MAX_TOKENS","5000")),"response_format":{"type":"json_object"},"messages":messages}
            canonical=json.dumps(body,ensure_ascii=False,sort_keys=True,separators=(",",":")); sha=hashlib.sha256(canonical.encode()).hexdigest()
            command=["curl","-sS","--fail-with-body","--max-time",str(TIMEOUT),"-H","Content-Type: application/json"]
            if KEY: command += ["-H","Authorization: Bearer "+KEY]
            command += ["--data-binary","@-",URL]
            completed=subprocess.run(command,input=canonical.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=TIMEOUT+5,check=True)
            wire=json.loads(completed.stdout); (WIRE/f"{sha}.json").write_text(json.dumps(wire,ensure_ascii=False)+"\n")
            result=normalize(json.loads(wire["choices"][0]["message"]["content"]),supplied,rid); previous_result=result; errors=validate(result,rid,supplied)
            if not errors: return {"raw_id":rid,"status":"VALIDATED","validation_errors":[],"contextual_v4":result,"model":MODEL,"protocol":PROTOCOL,"request_sha256":sha,"gold_visible":False}
            last=";".join(errors)
        except subprocess.TimeoutExpired: last="CURL_TIMEOUT"
        except subprocess.CalledProcessError as exc: last=f"CURL_EXIT_{exc.returncode}"
        except Exception as exc: last=f"{type(exc).__name__}:{str(exc)[:300]}"
        time.sleep(min(2**attempt,8))
    return {"raw_id":rid,"status":"ERROR","validation_errors":[last],"model":MODEL,"protocol":PROTOCOL,"request_sha256":sha,"gold_visible":False}

with OUT.open("a",encoding="utf8") as handle:
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(os.environ.get("CONTEXTUAL_WORKERS","2"))) as pool:
        futures=[pool.submit(request,item) for item in todo]
        for index,future in enumerate(concurrent.futures.as_completed(futures),1):
            item=future.result()
            handle.write(json.dumps(item,ensure_ascii=False)+"\n");handle.flush();latest[item["raw_id"]]=item
            status={"completed":len(done)+index,"total":len(targets),"validated":sum(x.get("status")=="VALIDATED" for x in latest.values()),"failed":sum(x.get("status")=="ERROR" for x in latest.values()),"updated_at":time.time()}
            (OUT.parent/"CONTEXTUAL_PROGRESS.json").write_text(json.dumps(status,indent=2))
            if index%10==0 or index==len(todo): print(json.dumps(status),flush=True)
with OUT.open("w",encoding="utf8") as handle:
    for rid in sorted(latest): handle.write(json.dumps(latest[rid],ensure_ascii=False)+"\n")
counts=Counter(x.get("status") for x in latest.values())
receipt={"target":len(targets),"rows":len(latest),"status":dict(counts),"protocol":PROTOCOL,"model":MODEL,"gold_used":False}
(OUT.parent/"CONTEXTUAL_V41_RECEIPT.json").write_text(json.dumps(receipt,indent=2)+"\n");print(json.dumps(receipt))
if counts.get("ERROR",0): raise SystemExit("contextual failures remain; resume before compiling")
