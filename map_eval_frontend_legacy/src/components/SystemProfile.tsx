"use client";

import { MechanismProfile, SystemMechanism, Funnel, pct } from "@/lib/data";

// 系统级机制画像:这个系统"整体上"怎么构建/检索/注入/回答。
// 不是单题,是全部题聚合出的机制特征。四段横向流水线卡片。

export default function SystemProfile({
  mech,
  profile,
  funnel,
}: {
  mech: SystemMechanism;
  profile: MechanismProfile;
  funnel: Funnel;
}) {
  const cards = [
    {
      stage: "①构建",
      color: "#3b82f6",
      how: mech.build,
      metrics: [
        {
          label: "平均建成记忆",
          value: profile.build.avg_memory_entries != null ? `${profile.build.avg_memory_entries} 条` : "—",
          hint: profile.build.min_memory_entries != null
            ? `范围 ${profile.build.min_memory_entries}–${profile.build.max_memory_entries}`
            : "",
        },
      ],
    },
    {
      stage: "②检索",
      color: "#f59e0b",
      how: mech.retrieve,
      metrics: [
        { label: "平均召回", value: profile.retrieval.avg_retrieved != null ? `${profile.retrieval.avg_retrieved} 条` : "—", hint: "" },
        {
          label: "gold 命中时平均排名",
          value: profile.retrieval.gold_avg_rank != null ? `第 ${profile.retrieval.gold_avg_rank} 位` : "—",
          hint: `命中 ${profile.retrieval.gold_hit_count}/${profile.n} 题`,
        },
        {
          label: "候选相似度",
          value: profile.retrieval.score_mean != null ? profile.retrieval.score_mean.toFixed(3) : "—",
          hint: profile.retrieval.score_min != null
            ? `${profile.retrieval.score_min}–${profile.retrieval.score_max}`
            : "分数不可得",
        },
      ],
    },
    {
      stage: "③注入",
      color: "#f97316",
      how: mech.inject,
      metrics: [
        { label: "平均注入", value: profile.inject.avg_injected != null ? `${profile.inject.avg_injected} 条` : "—", hint: "" },
        { label: "平均 prompt", value: profile.inject.avg_prompt_tokens != null ? `${profile.inject.avg_prompt_tokens} tok` : "—", hint: "" },
        { label: "gold 进 prompt 率", value: pct(funnel.utilize), hint: "" },
      ],
    },
    {
      stage: "④回答",
      color: "#22c55e",
      how: "LLM 基于注入的记忆生成答案",
      metrics: [
        {
          label: "证据齐全时答对率",
          value: pct(profile.answer.accuracy_given_gold),
          hint: profile.answer.accuracy_given_gold != null && profile.answer.accuracy_given_gold < 0.6 ? "利用能力偏弱" : "",
        },
        { label: "最终准确率", value: pct(funnel.answer), hint: "" },
      ],
    },
  ];

  return (
    <div>
      <p className="text-xs text-slate-500 mb-3">
        下面是该系统在全部 {profile.n} 题上聚合出的<b>整体机制画像</b>——它整体上如何构建、检索、注入、回答:
      </p>
      <div className="grid grid-cols-1 md:grid-cols-4 gap-2">
        {cards.map((c, i) => (
          <div key={c.stage} className="relative">
            <div
              className="rounded-lg border-2 p-3 h-full"
              style={{ borderColor: c.color, background: c.color + "0d" }}
            >
              <div className="text-sm font-semibold mb-1" style={{ color: c.color }}>
                {c.stage}
              </div>
              <div className="text-[11px] text-slate-500 mb-2 leading-snug min-h-[32px]">
                {c.how}
              </div>
              <div className="space-y-2">
                {c.metrics.map((m) => (
                  <div key={m.label}>
                    <div className="text-[10px] text-slate-400">{m.label}</div>
                    <div className="text-sm font-semibold text-slate-800 tabular-nums">
                      {m.value}
                    </div>
                    {m.hint && <div className="text-[10px] text-slate-400">{m.hint}</div>}
                  </div>
                ))}
              </div>
            </div>
            {i < cards.length - 1 && (
              <div className="hidden md:block absolute top-1/2 -right-1.5 text-slate-300 z-10">→</div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
