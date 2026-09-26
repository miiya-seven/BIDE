"use client";

import { Funnel, STAGES, FAILURE_LABELS, pct } from "@/lib/data";

// 五阶段漏斗:宽度=该阶段留存率,段间显示流失量。失败最大段高亮。
// 数据边界:build 阶段留存率暂缺(需全量记忆库),标注"待接入"。

// 五阶段固定配色(验证过的调色板):起点=灰 检索=橙 注入=黄 利用=violet 回答=绿
const STAGE_COLORS = ["#898781", "#eb6834", "#eda100", "#4a3aa7", "#008300"];

export default function FunnelChart({ funnel }: { funnel: Funnel }) {
  const values = STAGES.map((s) => (funnel as unknown as Record<string, number>)[s.key] ?? 0);

  // 找流失最大的阶段(诊断病灶)
  let worstDrop = 0;
  let worstIdx = -1;
  for (let i = 1; i < values.length; i++) {
    const drop = values[i - 1] - values[i];
    if (drop > worstDrop) {
      worstDrop = drop;
      worstIdx = i;
    }
  }

  const diagnosis = buildDiagnosis(funnel, worstIdx, worstDrop);

  return (
    <div className="w-full">
      {/* 漏斗主体 */}
      <div className="space-y-1">
        {STAGES.map((s, i) => {
          const v = values[i];
          const w = Math.max(v * 100, 2);
          const drop = i > 0 ? values[i - 1] - v : 0;
          const isWorst = i === worstIdx;
          return (
            <div key={s.key} className="flex items-center gap-3">
              <div className="w-24 text-right text-xs text-slate-500 shrink-0">
                {s.label}
              </div>
              <div className="flex-1 relative h-9">
                <div
                  className="h-full rounded flex items-center justify-end pr-2 transition-all"
                  style={{
                    width: `${w}%`,
                    background: STAGE_COLORS[i],
                    opacity: isWorst ? 1 : 0.82,
                    boxShadow: isWorst ? "0 0 0 2px #ef4444" : "none",
                  }}
                >
                  <span className="text-xs font-semibold text-white tabular-nums">
                    {pct(v)}
                  </span>
                </div>
              </div>
              <div className="w-28 text-xs shrink-0">
                {drop > 0.001 && (
                  <span className={isWorst ? "text-red-600 font-semibold" : "text-slate-400"}>
                    ↓ 流失 {pct(drop)}
                  </span>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* 证据齐全时的利用能力 */}
      {funnel.accuracy_given_gold !== null && (
        <div className="mt-3 text-xs text-slate-500">
          证据齐全(gold 已进 prompt)时的答对率:
          <span className="font-semibold text-slate-700">
            {pct(funnel.accuracy_given_gold)}
          </span>
          {funnel.accuracy_given_gold < 0.6 && (
            <span className="text-purple-600"> — 利用能力偏弱,证据到手也常答错</span>
          )}
        </div>
      )}

      {/* 自动诊断 */}
      <div className="mt-4 rounded-lg bg-slate-50 border border-slate-200 p-3 text-sm">
        <span className="font-semibold text-slate-700">诊断:</span>{" "}
        <span className="text-slate-600">{diagnosis}</span>
      </div>

      {/* 失败归因条 */}
      <FailureBar stages={funnel.failure_stages} n={funnel.n} />
    </div>
  );
}

function buildDiagnosis(funnel: Funnel, worstIdx: number, worstDrop: number): string {
  const stageName = ["", "检索", "注入", "利用", "回答"][worstIdx] || "";
  const parts: string[] = [];
  if (worstIdx > 0) {
    parts.push(`最大损失在「${stageName}」阶段(-${pct(worstDrop)})`);
  }
  // 判断利用能力
  if (funnel.accuracy_given_gold !== null && funnel.accuracy_given_gold < 0.6) {
    parts.push("即便证据齐全,利用能力也偏弱(常答错)");
  } else if (funnel.retrieval < 0.5) {
    parts.push("检索是主要瓶颈,大量 gold 证据没被捞到");
  }
  const ag = funnel.failure_stages["answer_generation"] || 0;
  const rt = funnel.failure_stages["retrieval"] || 0;
  if (ag > rt) {
    parts.push("失败主因是「证据齐全仍答错」而非检索——问题在利用而非检索");
  }
  return parts.join(";") || "各阶段较均衡";
}

function FailureBar({ stages, n }: { stages: Record<string, number>; n: number }) {
  const order = ["success", "retrieval", "retrieval_or_prompt_partial", "answer_generation", "unknown"];
  const items = order.filter((k) => stages[k]).map((k) => ({
    key: k,
    count: stages[k],
    ...FAILURE_LABELS[k],
  }));
  return (
    <div className="mt-4">
      <div className="text-xs text-slate-500 mb-1">失败归因(共 {n} 题)</div>
      <div className="flex h-6 rounded overflow-hidden">
        {items.map((it) => (
          <div
            key={it.key}
            className="h-full flex items-center justify-center text-[10px] text-white"
            style={{ width: `${(it.count / n) * 100}%`, background: it.color }}
            title={`${it.label}: ${it.count} (${pct(it.count / n)})`}
          >
            {it.count / n > 0.08 ? it.count : ""}
          </div>
        ))}
      </div>
      <div className="flex flex-wrap gap-x-4 gap-y-1 mt-2">
        {items.map((it) => (
          <div key={it.key} className="flex items-center gap-1 text-[11px] text-slate-600">
            <span className="inline-block w-2.5 h-2.5 rounded-sm" style={{ background: it.color }} />
            {it.label} · {pct(it.count / n)}
          </div>
        ))}
      </div>
    </div>
  );
}
