#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_data.py — 把后端评测产物聚合成前端用的精简 JSON。

设计见 map-eval-frontend-DESIGN.md。核心原则:
- 五阶段流水线:构建 → 检索 → 注入 → 利用 → 回答
- cat5 一律排除(论文口径只报 cat1-4 / LongMemEval 全类型)
- 数字必须可追溯到后端 per_sample / CSV
- 全量记忆库(构建侧单题溯源)暂缺,预留接口

数据来源:
- LoCoMo:  outputs-5.3/locomo-use1/systems/<sys>/per_sample_results.jsonl  (从 per_sample 现算)
- LongMemEval: outputs-5.3/output-longmemeval/systems/<sys>/  (有现成聚合 CSV + per_sample)

产出到 map-eval-frontend/public/data/:
- overview.json        各 benchmark 各系统的榜单+五阶段指标
- funnel.json          各系统五阶段留存率 + 失败归因(总体&分类别)
- systems.json         各系统机制画像(身份卡)
- samples/<bench>__<sys>.json   单题溯源(失败优先+成功采样),含检索 top-k 明细
"""
import json
import os
import collections
from pathlib import Path

# ---- 路径 ----
ROOT = Path("D:/code/map_platform/outputs-5.3")
LOCOMO_DIR = ROOT / "locomo-use1" / "systems"
LME_DIR = ROOT / "output-longmemeval" / "systems"
OUT = Path(__file__).resolve().parent.parent / "public" / "data"

BENCHMARKS = {
    "locomo": {"dir": LOCOMO_DIR, "label": "LoCoMo", "exclude_cat": "5"},
    "longmemeval": {"dir": LME_DIR, "label": "LongMemEval", "exclude_cat": None},
}

# 系统机制画像(基于适配器代码精读,见记忆 system-mechanisms-from-code)
SYSTEM_MECHANISMS = {
    "mem0": {"build": "LLM抽取语义记忆(SDK infer,黑盒)", "store": "qdrant向量库",
             "retrieve": "语义相似度检索(SDK黑盒)", "inject": "检索记忆平铺中段", "llm_extract": True},
    "memorybank": {"build": "时间分桶→chunk抽取+时段摘要", "store": "分层(摘要+细节)+遗忘曲线",
                   "retrieve": "0.7语义+0.3保留权重,命中强化", "inject": "摘要在前+细节在后", "llm_extract": True},
    "langmem": {"build": "语义+程序性双桶抽取(SDK黑盒)", "store": "langgraph向量双桶",
                "retrieve": "语义检索+query聚焦截断", "inject": "程序性进system+语义进中段", "llm_extract": True},
    "letta": {"build": "每8条一组写入归档(server黑盒)", "store": "核心block+归档向量库",
              "retrieve": "归档段落检索(黑盒)+词重叠回退", "inject": "核心进system+归档进中段", "llm_extract": False},
    "memgpt": {"build": "每8条一组写入归档(server黑盒)", "store": "核心block+归档向量库",
               "retrieve": "归档段落检索(黑盒)+词重叠回退", "inject": "核心进system+归档进中段", "llm_extract": False},
    "readagent": {"build": "按350词分页→gist摘要+原文双层", "store": "gist+raw page双层",
                  "retrieve": "LLM选页lookup(非相似度)+词重叠回退", "inject": "gist在前+展开页在后", "llm_extract": False},
    "simple_vector": {"build": "每turn存原文(非LLM)", "store": "纯Python TF-IDF",
                      "retrieve": "TF-IDF余弦top-k", "inject": "平铺中段", "llm_extract": False},
    "current_memory": {"build": "用数据集observation事实(非LLM)", "store": "纯Python TF-IDF",
                       "retrieve": "TF-IDF余弦top-k", "inject": "平铺中段", "llm_extract": False},
    "full_context": {"build": "每turn一条(带session)", "store": "全turn",
                     "retrieve": "只喂gold session(oracle上界)", "inject": "分session压缩", "llm_extract": False, "oracle": "upper"},
    "no_memory": {"build": "无", "store": "无", "retrieve": "无", "inject": "无记忆", "llm_extract": False, "oracle": "lower"},
}

STAGES = ["build", "retrieval", "inject", "utilize", "answer"]


def cat_of(row):
    return str(row.get("original_category") or row.get("question_type") or "?")


def load_per_sample(sys_dir):
    p = sys_dir / "per_sample_results.jsonl"
    if not p.is_file():
        return []
    rows = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def compute_funnel(rows):
    """从 per_sample 现算五阶段留存率 + 失败归因。"""
    n = len(rows)
    if n == 0:
        return None

    def rate(key):
        return round(sum(1 for r in rows if r.get(key)) / n, 4)

    # 五阶段:构建(has_gold_evidence 视为gold信息存在=起点100%,构建保留率暂用检索前提近似)
    #   真正的"构建保留率"需全量记忆库,暂缺 → 标 None
    funnel = {
        "gold_exists": rate("has_gold_evidence"),
        "build": None,  # 待全量记忆库接入
        "retrieval": rate("retrieval_hit"),
        "inject": rate("prompt_hit"),
        "utilize": rate("gold_in_prompt"),   # gold 进了prompt(利用的前提)
        "answer": rate("answer_correct"),
    }
    # accuracy_given_gold_in_prompt:证据齐全时的答对率(利用能力)
    gip = [r for r in rows if r.get("gold_in_prompt")]
    funnel["accuracy_given_gold"] = round(
        sum(1 for r in gip if r.get("answer_correct")) / len(gip), 4) if gip else None

    # 失败归因(互斥)
    fs = collections.Counter(r.get("failure_stage") for r in rows)
    funnel["failure_stages"] = {k: v for k, v in fs.items()}
    funnel["n"] = n
    return funnel


def build_overview_row(sys_name, rows):
    n = len(rows)
    if n == 0:
        return None

    def rate(key):
        return round(sum(1 for r in rows if r.get(key)) / n, 4)

    def avg(key):
        vals = [r.get(key) for r in rows if isinstance(r.get(key), (int, float))]
        return round(sum(vals) / len(vals), 2) if vals else None

    return {
        "system": sys_name,
        "n": n,
        "answer_accuracy": rate("answer_correct"),
        "retrieval_hit": rate("retrieval_hit"),
        "prompt_hit": rate("prompt_hit"),
        "gold_in_prompt": rate("gold_in_prompt"),
        "avg_total_tokens": avg("total_tokens"),
        "avg_latency": avg("latency"),
        "oracle": SYSTEM_MECHANISMS.get(sys_name, {}).get("oracle"),
    }


def compute_mechanism(rows):
    """系统级机制画像:整体上这个系统怎么构建/检索/注入/回答。"""
    import statistics as st
    n = len(rows)
    if n == 0:
        return None

    def safe_mean(xs):
        xs = [x for x in xs if isinstance(x, (int, float))]
        return round(st.mean(xs), 2) if xs else None

    # 构建:记忆条数
    mem_counts = [
        (r.get("system_diagnostics") or {}).get("num_memory_entries")
        for r in rows
    ]
    mem_counts = [m for m in mem_counts if isinstance(m, (int, float)) and m > 0]

    # 检索:召回条数、gold 命中排名、候选分数分布
    retr_counts = [len(r.get("retrieved_source_ids") or []) for r in rows]
    gold_ranks = []
    for r in rows:
        gold = set(r.get("retrieved_gold_ids") or [])
        for i, s in enumerate(r.get("retrieved_source_ids") or []):
            if s in gold:
                gold_ranks.append(i + 1)
                break
    scores = []
    for r in rows:
        for m in (r.get("retrieved_memories") or []):
            if isinstance(m.get("score"), (int, float)):
                scores.append(m["score"])

    # 注入
    inj_counts = [len(r.get("injected_source_ids") or []) for r in rows]
    prompt_tokens = [r.get("prompt_tokens") for r in rows]

    # 回答:证据齐全答对率
    gip = [r for r in rows if r.get("gold_in_prompt")]
    acc_given_gold = round(
        sum(1 for r in gip if r.get("answer_correct")) / len(gip), 4
    ) if gip else None

    return {
        "build": {
            "avg_memory_entries": safe_mean(mem_counts),
            "min_memory_entries": min(mem_counts) if mem_counts else None,
            "max_memory_entries": max(mem_counts) if mem_counts else None,
        },
        "retrieval": {
            "avg_retrieved": safe_mean(retr_counts),
            "gold_avg_rank": round(st.mean(gold_ranks), 2) if gold_ranks else None,
            "gold_hit_count": len(gold_ranks),
            "score_mean": round(st.mean(scores), 3) if scores else None,
            "score_min": round(min(scores), 3) if scores else None,
            "score_max": round(max(scores), 3) if scores else None,
        },
        "inject": {
            "avg_injected": safe_mean(inj_counts),
            "avg_prompt_tokens": safe_mean(prompt_tokens),
        },
        "answer": {
            "accuracy_given_gold": acc_given_gold,
        },
        "n": n,
    }


def slim_sample(row):
    """单题溯源精简:保留检索 top-k 明细,剥离超大字段。"""
    rm = row.get("retrieved_memories") or []
    slim_rm = []
    for m in rm[:10]:
        slim_rm.append({
            "content": (m.get("content") or "")[:300],
            "score": m.get("score"),
            "rank": m.get("rank"),
            "source_ids": m.get("source_ids") or (m.get("metadata") or {}).get("source_batch_turn_ids", [])[:5],
        })
    gold_units = row.get("gold_evidence_units") or []
    return {
        "sample_id": row.get("sample_id"),
        "question": row.get("question"),
        "gold_answer": row.get("gold_answer"),
        "pred_answer": row.get("pred_answer"),
        "category": cat_of(row),
        "answer_correct": bool(row.get("answer_correct")),
        "failure_stage": row.get("failure_stage"),
        "retrieval_hit": bool(row.get("retrieval_hit")),
        "prompt_hit": bool(row.get("prompt_hit")),
        "gold_in_prompt": bool(row.get("gold_in_prompt")),
        "num_memory_entries": (row.get("system_diagnostics") or {}).get("num_memory_entries"),
        "gold_evidence": [{"source_id": g.get("source_id"), "text": (g.get("text") or "")[:200]}
                          for g in gold_units[:5]],
        "retrieved_top": slim_rm,
        "retrieved_source_ids": (row.get("retrieved_source_ids") or [])[:20],
        "retrieved_gold_ids": row.get("retrieved_gold_ids") or [],
    }


def select_samples(rows, max_fail=40, max_success=15):
    """失败样本优先 + 成功样本采样,控制体积。"""
    fails = [r for r in rows if not r.get("answer_correct")]
    succ = [r for r in rows if r.get("answer_correct")]
    picked = fails[:max_fail] + succ[:max_success]
    return [slim_sample(r) for r in picked]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "samples").mkdir(exist_ok=True)

    overview = {}
    funnels = {}
    mechanisms = {}
    systems_meta = {}

    for bench, cfg in BENCHMARKS.items():
        bdir = cfg["dir"]
        excl = cfg["exclude_cat"]
        if not bdir.is_dir():
            print(f"[skip] {bench}: 目录不存在 {bdir}")
            continue
        overview[bench] = []
        funnels[bench] = {}
        mechanisms[bench] = {}
        sys_names = sorted([d.name for d in bdir.iterdir() if d.is_dir()])
        for sys_name in sys_names:
            rows = load_per_sample(bdir / sys_name)
            if excl:
                rows = [r for r in rows if cat_of(r) != excl]
            if not rows:
                print(f"[warn] {bench}/{sys_name}: 无样本")
                continue
            ov = build_overview_row(sys_name, rows)
            if ov:
                overview[bench].append(ov)
            fn = compute_funnel(rows)
            if fn:
                funnels[bench][sys_name] = fn
            mech = compute_mechanism(rows)
            if mech:
                mechanisms[bench][sys_name] = mech
            # 单题样本
            samples = select_samples(rows)
            with open(OUT / "samples" / f"{bench}__{sys_name}.json", "w", encoding="utf-8") as f:
                json.dump(samples, f, ensure_ascii=False)
            print(f"[ok] {bench}/{sys_name}: {len(rows)}题, {len(samples)}样本")

        overview[bench].sort(key=lambda x: x["answer_accuracy"], reverse=True)

    systems_meta = SYSTEM_MECHANISMS

    with open(OUT / "overview.json", "w", encoding="utf-8") as f:
        json.dump(overview, f, ensure_ascii=False, indent=1)
    with open(OUT / "funnel.json", "w", encoding="utf-8") as f:
        json.dump(funnels, f, ensure_ascii=False, indent=1)
    with open(OUT / "mechanism.json", "w", encoding="utf-8") as f:
        json.dump(mechanisms, f, ensure_ascii=False, indent=1)
    with open(OUT / "systems.json", "w", encoding="utf-8") as f:
        json.dump(systems_meta, f, ensure_ascii=False, indent=1)

    print(f"\n完成。产出目录: {OUT}")


if __name__ == "__main__":
    main()
