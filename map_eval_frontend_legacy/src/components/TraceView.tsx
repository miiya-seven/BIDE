"use client";

import { SampleTrace } from "@/lib/data";

// 单题机制溯源:把一道题走过的五阶段链路摊开。
// 构建侧(全量记忆库)待接入,先展示"建了多少条";检索侧数据已足够(top-k 明细)。

export default function TraceView({ sample }: { sample: SampleTrace }) {
  const goldIds = new Set(sample.gold_evidence.map((g) => g.source_id));
  // gold 在检索结果里的排名(retrieved_source_ids 顺序近似召回顺序)
  const goldRankInRetrieved = (() => {
    for (let i = 0; i < sample.retrieved_source_ids.length; i++) {
      if (goldIds.has(sample.retrieved_source_ids[i])) return i + 1;
    }
    return null;
  })();

  return (
    <div className="space-y-4">
      {/* 问题 + 答案对比 */}
      <div className="rounded-lg border border-slate-200 bg-white p-4">
        <div className="text-sm text-slate-500 mb-1">问题(类别 {sample.category})</div>
        <div className="font-medium text-slate-800 mb-3">{sample.question}</div>
        <div className="grid grid-cols-2 gap-3 text-sm">
          <div className="rounded bg-emerald-50 border border-emerald-100 p-2">
            <div className="text-xs text-emerald-600 mb-0.5">标准答案</div>
            <div className="text-slate-800">{sample.gold_answer}</div>
          </div>
          <div
            className={`rounded p-2 border ${
              sample.answer_correct
                ? "bg-emerald-50 border-emerald-100"
                : "bg-red-50 border-red-100"
            }`}
          >
            <div className={`text-xs mb-0.5 ${sample.answer_correct ? "text-emerald-600" : "text-red-600"}`}>
              系统答案 {sample.answer_correct ? "✓" : "✗"}
            </div>
            <div className="text-slate-800">{sample.pred_answer || "(空)"}</div>
          </div>
        </div>
      </div>

      {/* 五阶段链路 */}
      <div className="rounded-lg border border-slate-200 bg-white p-4">
        <div className="font-semibold text-slate-800 mb-3">机制链路</div>

        {/* ①构建 */}
        <StageBlock idx="①" name="构建" status="info">
          该对话共建成{" "}
          <span className="font-semibold">{sample.num_memory_entries ?? "—"}</span> 条记忆
          <span className="text-slate-400 text-xs ml-2">
            (全量记忆库逐条内容待接入,gold 是否在构建阶段丢失暂无法判定)
          </span>
        </StageBlock>

        {/* ②检索 */}
        <StageBlock
          idx="②"
          name="检索"
          status={sample.retrieval_hit ? "ok" : "fail"}
        >
          {sample.retrieval_hit ? (
            <>gold 证据被检索到 ✓{goldRankInRetrieved && `(排在第 ${goldRankInRetrieved} 位)`}</>
          ) : (
            <span className="text-red-600">gold 证据未被检索到 ✗ — 检索阶段失分</span>
          )}
          {/* top-k 检索候选明细 */}
          <div className="mt-2 space-y-1">
            {sample.retrieved_top.slice(0, 6).map((m, i) => {
              const isGold = (m.source_ids || []).some((s) => goldIds.has(s));
              return (
                <div
                  key={i}
                  className={`text-xs rounded px-2 py-1 border flex gap-2 ${
                    isGold ? "border-red-300 bg-red-50" : "border-slate-100 bg-slate-50"
                  }`}
                >
                  <span className="text-slate-400 shrink-0">
                    #{m.rank ?? i + 1}
                  </span>
                  {m.score !== null && (
                    <span className="text-slate-500 tabular-nums shrink-0">
                      {m.score.toFixed(3)}
                    </span>
                  )}
                  <span className="text-slate-700 truncate">{m.content}</span>
                  {isGold && <span className="text-red-500 shrink-0">← gold</span>}
                </div>
              );
            })}
          </div>
        </StageBlock>

        {/* ③注入 */}
        <StageBlock idx="③" name="注入" status={sample.prompt_hit ? "ok" : "fail"}>
          {sample.gold_in_prompt ? (
            <>gold 证据进入了 prompt ✓</>
          ) : (
            <span className="text-amber-600">gold 证据未进入 prompt ✗ — 注入阶段丢失</span>
          )}
        </StageBlock>

        {/* ④利用 / ⑤回答 */}
        <StageBlock
          idx="④⑤"
          name="利用 / 回答"
          status={sample.answer_correct ? "ok" : "fail"}
        >
          {sample.gold_in_prompt && !sample.answer_correct ? (
            <span className="text-purple-600">
              证据齐全却答错 — 利用阶段失败(不是检索的锅)
            </span>
          ) : sample.answer_correct ? (
            <span className="text-emerald-600">最终答对 ✓</span>
          ) : (
            <span className="text-slate-500">因前序阶段缺证据而答错</span>
          )}
        </StageBlock>
      </div>

      {/* 诊断 */}
      <div className="rounded-lg bg-slate-800 text-slate-100 p-4 text-sm">
        <span className="font-semibold">根因诊断:</span>{" "}
        {diagnose(sample, goldRankInRetrieved)}
      </div>
    </div>
  );
}

function StageBlock({
  idx,
  name,
  status,
  children,
}: {
  idx: string;
  name: string;
  status: "ok" | "fail" | "info";
  children: React.ReactNode;
}) {
  const dot =
    status === "ok" ? "bg-emerald-500" : status === "fail" ? "bg-red-500" : "bg-slate-400";
  return (
    <div className="flex gap-3 py-2 border-b border-slate-50 last:border-0">
      <div className="flex flex-col items-center shrink-0">
        <span className={`w-2.5 h-2.5 rounded-full ${dot} mt-1.5`} />
      </div>
      <div className="flex-1">
        <div className="text-xs text-slate-400">
          {idx} {name}
        </div>
        <div className="text-sm text-slate-700 mt-0.5">{children}</div>
      </div>
    </div>
  );
}

function diagnose(s: SampleTrace, goldRank: number | null): string {
  if (!s.retrieval_hit) {
    return `gold 证据没被检索到${goldRank ? `(仅排到第 ${goldRank} 位、未进 top-k)` : ""}。病灶在「检索」环节——检索机制没能把关键证据排进前列。`;
  }
  if (s.retrieval_hit && !s.gold_in_prompt) {
    return "gold 被检索到了,却没进 prompt。病灶在「注入」环节——检索到的证据在拼装 prompt 时被挤掉。";
  }
  if (s.gold_in_prompt && !s.answer_correct) {
    return "证据齐全(gold 已在 prompt 中),模型却答错。病灶在「利用」环节——不是检索问题,是模型没用好到手的证据。";
  }
  if (s.answer_correct) {
    return "全链路走通,最终答对。";
  }
  return "综合失败。";
}
