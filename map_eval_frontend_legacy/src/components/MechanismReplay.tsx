"use client";

import { useState } from "react";
import { SampleTrace, SystemMechanism } from "@/lib/data";

// 机制回放:基于真实中间数据,展示数据在系统内部的变形与流动。
// 三段:①构建(原文→记忆,含来源映射) ②检索(逐条真实分数排序) ③注入(哪些进了prompt)
// 数据边界:全库分数缺→只展示被检索到的候选;全量记忆库缺→构建段标注"样例记忆"。

type Step = 0 | 1 | 2 | 3;

const STEP_LABELS = ["原始对话", "①构建:抽取记忆", "②检索:相似度排序", "③注入:拼进 prompt"];

export default function MechanismReplay({
  sample,
  mech,
}: {
  sample: SampleTrace;
  mech: SystemMechanism;
}) {
  const [step, setStep] = useState<Step>(0);

  const goldIds = new Set(sample.gold_evidence.map((g) => g.source_id));
  const promptGold = sample.gold_in_prompt;

  // 被检索到的候选(带真实分数),按 rank 排序
  const cands = [...sample.retrieved_top].sort(
    (a, b) => (a.rank ?? 99) - (b.rank ?? 99)
  );
  const maxScore = Math.max(...cands.map((c) => c.score ?? 0), 0.01);

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4">
      {/* 步骤控制条 */}
      <div className="flex items-center gap-2 mb-4">
        <span className="font-semibold text-slate-800 mr-2">机制回放</span>
        {STEP_LABELS.map((l, i) => (
          <button
            key={i}
            onClick={() => setStep(i as Step)}
            className={`px-2.5 py-1 rounded text-xs font-medium transition ${
              step === i
                ? "bg-indigo-600 text-white"
                : step > i
                ? "bg-indigo-50 text-indigo-600"
                : "bg-slate-100 text-slate-400"
            }`}
          >
            {l}
          </button>
        ))}
        <div className="ml-auto flex gap-1">
          <button
            onClick={() => setStep((s) => Math.max(0, s - 1) as Step)}
            disabled={step === 0}
            className="px-2 py-1 rounded border border-slate-200 text-xs disabled:opacity-40"
          >
            ←
          </button>
          <button
            onClick={() => setStep((s) => Math.min(3, s + 1) as Step)}
            disabled={step === 3}
            className="px-2 py-1 rounded border border-slate-200 text-xs disabled:opacity-40"
          >
            下一步 →
          </button>
        </div>
      </div>

      {/* 舞台 */}
      <div className="min-h-[280px]">
        {step === 0 && <StageOriginal sample={sample} />}
        {step === 1 && (
          <StageBuild sample={sample} mech={mech} goldIds={goldIds} cands={cands} />
        )}
        {step === 2 && (
          <StageRetrieve cands={cands} maxScore={maxScore} goldIds={goldIds} />
        )}
        {step === 3 && (
          <StageInject
            cands={cands}
            goldIds={goldIds}
            promptGold={promptGold}
            sample={sample}
          />
        )}
      </div>
    </div>
  );
}

/* ① 原始对话:展示 gold 证据所在原文 */
function StageOriginal({ sample }: { sample: SampleTrace }) {
  return (
    <div>
      <p className="text-xs text-slate-500 mb-3">
        这道题的答案藏在原始多会话对话里。gold 证据(红框)是回答本题必须用到的原文:
      </p>
      {sample.gold_evidence.map((g) => (
        <div key={g.source_id} className="rounded border-2 border-red-300 bg-red-50 p-2 mb-2 text-sm">
          <span className="text-[10px] text-red-500 font-mono mr-2">{g.source_id}</span>
          <span className="text-slate-700">{g.text}</span>
        </div>
      ))}
      <p className="text-xs text-slate-400 mt-2">
        问题:{sample.question} → 正确答案:{sample.gold_answer}
      </p>
    </div>
  );
}

/* ② 构建:原文 → LLM 抽取 → 记忆条目(带来源映射) */
function StageBuild({
  sample,
  mech,
  goldIds,
  cands,
}: {
  sample: SampleTrace;
  mech: SystemMechanism;
  goldIds: Set<string>;
  cands: SampleTrace["retrieved_top"];
}) {
  return (
    <div>
      <p className="text-xs text-slate-500 mb-3">
        {mech.build}。该对话共建成{" "}
        <span className="font-semibold">{sample.num_memory_entries ?? "—"}</span> 条记忆。
        下面是本题相关的几条(每条标出它从哪些原文轮次抽取而来——这就是“原文→记忆”的变形):
      </p>
      <div className="space-y-2">
        {cands.slice(0, 5).map((c, i) => {
          const fromGold = (c.source_ids || []).some((s) => goldIds.has(s));
          return (
            <div
              key={i}
              className={`rounded border p-2 text-sm ${
                fromGold ? "border-red-300 bg-red-50" : "border-slate-200 bg-slate-50"
              }`}
            >
              <div className="text-slate-700">{c.content}</div>
              <div className="text-[10px] text-slate-400 mt-1 font-mono">
                ← 抽取自原文:{(c.source_ids || []).slice(0, 8).join(", ") || "(未知)"}
                {fromGold && <span className="text-red-500 ml-1">含 gold 证据</span>}
              </div>
            </div>
          );
        })}
      </div>
      <p className="text-xs text-amber-600 mt-2">
        ⚠ 全量记忆库({sample.num_memory_entries ?? "?"} 条)逐条内容待接入,此处仅展示与本题相关的样例。
      </p>
    </div>
  );
}

/* ③ 检索:逐条真实分数 + 排序 + gold 位置 */
function StageRetrieve({
  cands,
  maxScore,
  goldIds,
}: {
  cands: SampleTrace["retrieved_top"];
  maxScore: number;
  goldIds: Set<string>;
}) {
  return (
    <div>
      <p className="text-xs text-slate-500 mb-3">
        问题向量化后,与记忆库逐条算相似度并排序。下面是 top-{cands.length} 候选的<b>真实分数</b>
        (条形长度=分数)。gold 记忆(红)排得越靠后,越可能被挤出:
      </p>
      <div className="space-y-1.5">
        {cands.map((c, i) => {
          const isGold = (c.source_ids || []).some((s) => goldIds.has(s));
          const w = ((c.score ?? 0) / maxScore) * 100;
          return (
            <div key={i} className="flex items-center gap-2">
              <span className="w-8 text-right text-xs text-slate-400 tabular-nums shrink-0">
                #{c.rank ?? i + 1}
              </span>
              <div className="flex-1 relative h-7 bg-slate-100 rounded overflow-hidden">
                <div
                  className="h-full flex items-center px-2 rounded transition-all"
                  style={{ width: `${w}%`, background: isGold ? "#ef4444" : "#60a5fa" }}
                >
                  <span className="text-[10px] text-white font-semibold tabular-nums">
                    {(c.score ?? 0).toFixed(3)}
                  </span>
                </div>
                <span className="absolute left-2 top-1/2 -translate-y-1/2 text-[11px] text-slate-600 truncate max-w-[70%] pointer-events-none"
                  style={{ left: `${Math.max(w, 8)}%` }}>
                  {c.content.slice(0, 50)}
                </span>
              </div>
              {isGold && <span className="text-xs text-red-500 shrink-0">← gold</span>}
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ④ 注入:哪些进了 prompt */
function StageInject({
  cands,
  goldIds,
  promptGold,
  sample,
}: {
  cands: SampleTrace["retrieved_top"];
  goldIds: Set<string>;
  promptGold: boolean;
  sample: SampleTrace;
}) {
  return (
    <div>
      <p className="text-xs text-slate-500 mb-3">
        排名靠前的候选被拼进 prompt 交给 LLM。gold 证据{promptGold ? "进入了" : "未进入"} prompt:
      </p>
      <div className="grid grid-cols-1 gap-2">
        {cands.slice(0, 5).map((c, i) => {
          const isGold = (c.source_ids || []).some((s) => goldIds.has(s));
          const injected = i < 5; // top-5 视为注入
          return (
            <div
              key={i}
              className={`rounded border p-2 text-sm flex items-center gap-2 ${
                injected ? "border-emerald-200 bg-emerald-50" : "border-slate-100 bg-slate-50 opacity-50"
              }`}
            >
              <span className="text-xs shrink-0">{injected ? "✓进prompt" : "✗被截断"}</span>
              <span className={`text-slate-700 truncate ${isGold ? "font-semibold" : ""}`}>
                {c.content.slice(0, 60)}
              </span>
              {isGold && <span className="text-red-500 text-xs shrink-0">gold</span>}
            </div>
          );
        })}
      </div>
      <div className={`mt-3 rounded p-2 text-sm ${sample.answer_correct ? "bg-emerald-50 text-emerald-700" : "bg-red-50 text-red-700"}`}>
        最终答案:{sample.pred_answer || "(空)"} {sample.answer_correct ? "✓" : `✗(应为 ${sample.gold_answer})`}
      </div>
    </div>
  );
}
