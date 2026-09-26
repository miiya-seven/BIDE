"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { MechanismProfile, SystemMechanism, Funnel, SampleTrace, pct } from "@/lib/data";

// 动画化机制图示:用卡片流动的视觉隐喻表达每阶段"在做什么动作"。
// 双模式:整体(系统聚合数)/ 单题(真实样例数据)。
// 四阶段场景:构建(压缩)→检索(相似度排序)→注入(装容器)→回答。

type Mode = "overall" | "sample";
type Stage = "build" | "retrieve" | "inject" | "answer";

const STAGE_TABS: { key: Stage; label: string; color: string }[] = [
  { key: "build", label: "① 构建", color: "#3b82f6" },
  { key: "retrieve", label: "② 检索", color: "#f59e0b" },
  { key: "inject", label: "③ 注入", color: "#f97316" },
  { key: "answer", label: "④ 回答", color: "#22c55e" },
];

export default function MechanismFlow({
  mech,
  profile,
  funnel,
  sample,
}: {
  mech: SystemMechanism;
  profile: MechanismProfile;
  funnel: Funnel;
  sample?: SampleTrace;
}) {
  const [stage, setStage] = useState<Stage>("build");
  const [mode, setMode] = useState<Mode>("overall");

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4">
      <div className="flex items-center gap-2 mb-4 flex-wrap">
        <span className="font-semibold text-slate-800 mr-1">机制图示</span>
        {STAGE_TABS.map((t) => (
          <button
            key={t.key}
            onClick={() => setStage(t.key)}
            className="px-2.5 py-1 rounded text-xs font-medium transition"
            style={
              stage === t.key
                ? { background: t.color, color: "white" }
                : { background: "#f1f5f9", color: "#64748b" }
            }
          >
            {t.label}
          </button>
        ))}
        <div className="ml-auto flex items-center rounded border border-slate-200 overflow-hidden text-xs">
          <button
            onClick={() => setMode("overall")}
            className={`px-2 py-1 ${mode === "overall" ? "bg-slate-800 text-white" : "text-slate-500"}`}
          >
            整体
          </button>
          <button
            onClick={() => setMode("sample")}
            disabled={!sample}
            className={`px-2 py-1 ${mode === "sample" ? "bg-slate-800 text-white" : "text-slate-500"} disabled:opacity-40`}
          >
            单题
          </button>
        </div>
      </div>

      <div className="min-h-[300px] relative overflow-hidden">
        <AnimatePresence mode="wait">
          <motion.div
            key={stage + mode}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.25 }}
          >
            {stage === "build" && <BuildScene mode={mode} profile={profile} mech={mech} sample={sample} />}
            {stage === "retrieve" && <RetrieveScene mode={mode} profile={profile} sample={sample} />}
            {stage === "inject" && <InjectScene mode={mode} profile={profile} funnel={funnel} sample={sample} />}
            {stage === "answer" && <AnswerScene mode={mode} profile={profile} funnel={funnel} sample={sample} />}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}

/* ============ ① 构建:对话 → LLM漏斗 → 记忆卡片堆(压缩隐喻) ============ */
function BuildScene({
  mode,
  profile,
  mech,
  sample,
}: {
  mode: Mode;
  profile: MechanismProfile;
  mech: SystemMechanism;
  sample?: SampleTrace;
}) {
  const memCount = mode === "overall"
    ? profile.build.avg_memory_entries ?? 0
    : sample?.num_memory_entries ?? 0;
  // 左侧对话气泡(示意数量),右侧记忆卡片(取真实内容或示意)
  const dialogues = mode === "sample" && sample
    ? sample.gold_evidence.map((g) => g.text).slice(0, 3)
    : ["用户: 我5月7日去了LGBTQ支持小组…", "用户: 7月加入了mentorship…", "…多轮对话…"];
  const memCards = mode === "sample" && sample
    ? sample.retrieved_top.slice(0, 4).map((m) => m.content)
    : ["Caroline 2023-05-07 参加LGBTQ支持组…", "Caroline 7月加入mentorship…", "…"];

  return (
    <div>
      <p className="text-xs text-slate-500 mb-4">
        {mech.build}。{mode === "overall" ? "整体上" : "这道题"}把对话压缩成{" "}
        <span className="font-semibold text-blue-600">{memCount || "—"}</span> 条记忆:
      </p>
      <div className="flex items-center gap-3">
        {/* 对话气泡 */}
        <div className="flex-1 space-y-2">
          {dialogues.map((d, i) => (
            <motion.div
              key={i}
              initial={{ x: -30, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              transition={{ delay: i * 0.15 }}
              className="rounded-2xl rounded-bl-sm bg-slate-100 px-3 py-1.5 text-xs text-slate-600"
            >
              {d.slice(0, 40)}
            </motion.div>
          ))}
        </div>
        {/* LLM 漏斗 */}
        <motion.div
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ delay: 0.4 }}
          className="shrink-0 flex flex-col items-center"
        >
          <div className="text-[10px] text-slate-400 mb-1">{mech.llm_extract ? "LLM抽取" : "直接索引"}</div>
          <div className="w-0 h-0 border-l-[28px] border-r-[28px] border-t-[36px] border-l-transparent border-r-transparent" style={{ borderTopColor: "#3b82f6" }} />
          <div className="text-lg text-blue-400">↓</div>
        </motion.div>
        {/* 记忆卡片堆 */}
        <div className="flex-1 relative">
          {memCards.map((m, i) => (
            <motion.div
              key={i}
              initial={{ x: 30, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              transition={{ delay: 0.6 + i * 0.15 }}
              className="rounded border border-blue-200 bg-blue-50 px-2 py-1.5 text-xs text-slate-700 mb-1.5 shadow-sm"
            >
              📇 {m.slice(0, 38)}
            </motion.div>
          ))}
          {mode === "overall" && (
            <div className="text-[10px] text-slate-400 mt-1">…等共 {memCount} 条</div>
          )}
        </div>
      </div>
      <div className="mt-4 text-xs text-slate-400">
        压缩隐喻:多轮原始对话 → {mech.llm_extract ? "LLM 抽取关键事实" : "逐条索引"} → 结构化记忆条目
      </div>
    </div>
  );
}

/* ============ ② 检索:问题 + 记忆按相似度浮动排序 ============ */
function RetrieveScene({
  mode,
  profile,
  sample,
}: {
  mode: Mode;
  profile: MechanismProfile;
  sample?: SampleTrace;
}) {
  // 单题用真实候选+分数;整体用聚合分数分布造示意条目
  const items = mode === "sample" && sample
    ? sample.retrieved_top.slice(0, 6).map((m) => ({
        text: m.content.slice(0, 40),
        score: m.score ?? 0,
        isGold: (m.source_ids || []).some((s) => sample.gold_evidence.some((g) => g.source_id === s)),
      }))
    : buildSyntheticScores(profile);
  const maxScore = Math.max(...items.map((i) => i.score), 0.01);
  const sorted = [...items].sort((a, b) => b.score - a.score);

  return (
    <div>
      <p className="text-xs text-slate-500 mb-4">
        问题向量化后,与记忆逐条算相似度并排序。
        {mode === "overall"
          ? ` 整体上 gold 命中时平均排在第 ${profile.retrieval.gold_avg_rank ?? "—"} 位,候选分数均值 ${profile.retrieval.score_mean ?? "—"}。`
          : " 下面是这道题的真实候选(gold 高亮):"}
      </p>
      <div className="space-y-2">
        {sorted.map((it, i) => (
          <motion.div
            key={i}
            layout
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.12 }}
            className="flex items-center gap-2"
          >
            <span className="w-6 text-right text-xs text-slate-400 tabular-nums">#{i + 1}</span>
            <div className="flex-1 h-8 bg-slate-100 rounded overflow-hidden relative">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${(it.score / maxScore) * 100}%` }}
                transition={{ delay: 0.2 + i * 0.12, duration: 0.5 }}
                className="h-full rounded flex items-center px-2"
                style={{ background: it.isGold ? "#ef4444" : "#60a5fa" }}
              >
                <span className="text-[10px] text-white font-semibold tabular-nums">{it.score.toFixed(3)}</span>
              </motion.div>
              <span className="absolute right-2 top-1/2 -translate-y-1/2 text-[11px] text-slate-500 truncate max-w-[60%]">
                {it.text}
              </span>
            </div>
            {it.isGold && <span className="text-xs text-red-500">← gold</span>}
          </motion.div>
        ))}
      </div>
      <div className="mt-3 text-xs text-slate-400">
        排序隐喻:相似度越高的记忆浮得越靠上;gold 若排在 top-k 之外就会被漏掉。
      </div>
    </div>
  );
}

function buildSyntheticScores(profile: MechanismProfile) {
  const mean = profile.retrieval.score_mean ?? 0.6;
  const max = profile.retrieval.score_max ?? 0.82;
  const min = profile.retrieval.score_min ?? 0.3;
  const goldRank = Math.round(profile.retrieval.gold_avg_rank ?? 2);
  const arr = [max, (max + mean) / 2, mean, mean * 0.95, (mean + min) / 2, min];
  return arr.map((s, i) => ({
    text: i === goldRank - 1 ? "(gold 记忆:平均排此位)" : `记忆候选 ${i + 1}`,
    score: s,
    isGold: i === goldRank - 1,
  }));
}

/* ============ ③ 注入:卡片装进 prompt 容器,溢出被挡 ============ */
function InjectScene({
  mode,
  profile,
  funnel,
  sample,
}: {
  mode: Mode;
  profile: MechanismProfile;
  funnel: Funnel;
  sample?: SampleTrace;
}) {
  const injCount = mode === "overall"
    ? Math.round(profile.inject.avg_injected ?? 0)
    : Math.min(sample?.retrieved_top.length ?? 5, 5);
  const tokens = mode === "overall" ? profile.inject.avg_prompt_tokens : null;
  const goldInPrompt = mode === "sample" && sample ? sample.gold_in_prompt : (funnel.utilize ?? 0) > 0.5;

  const cards = Array.from({ length: Math.min(injCount + 2, 7) });

  return (
    <div>
      <p className="text-xs text-slate-500 mb-4">
        排名靠前的记忆被装进 prompt 交给 LLM
        {tokens ? `(整体平均 ${tokens} token)` : ""}。容量有限,靠后的被挡在外:
      </p>
      <div className="flex gap-4 items-start">
        <div className="flex-1 space-y-1.5">
          {cards.map((_, i) => {
            const injected = i < injCount;
            return (
              <motion.div
                key={i}
                initial={{ x: 0, opacity: 1 }}
                animate={injected ? { x: 40, opacity: 1 } : { x: 0, opacity: 0.4 }}
                transition={{ delay: i * 0.1, duration: 0.4 }}
                className={`rounded border px-2 py-1.5 text-xs ${
                  injected ? "border-emerald-200 bg-emerald-50 text-slate-700" : "border-slate-200 bg-slate-50 text-slate-400"
                }`}
              >
                {injected ? "→ 进入 prompt" : "✗ 被截断"} · 记忆 {i + 1}
              </motion.div>
            );
          })}
        </div>
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
          className="shrink-0 w-40 rounded-lg border-2 border-orange-300 bg-orange-50 p-3"
        >
          <div className="text-xs font-semibold text-orange-600 mb-2">Prompt 容器</div>
          <div className="text-[11px] text-slate-500">已装入 {injCount} 条记忆</div>
          <div className={`mt-2 text-xs rounded px-2 py-1 ${goldInPrompt ? "bg-emerald-100 text-emerald-700" : "bg-red-100 text-red-700"}`}>
            gold 证据{goldInPrompt ? " ✓ 已进入" : " ✗ 未进入"}
          </div>
        </motion.div>
      </div>
    </div>
  );
}

/* ============ ④ 回答:证据 → LLM → 答案 ============ */
function AnswerScene({
  mode,
  profile,
  funnel,
  sample,
}: {
  mode: Mode;
  profile: MechanismProfile;
  funnel: Funnel;
  sample?: SampleTrace;
}) {
  const accGivenGold = profile.answer.accuracy_given_gold;
  const correct = mode === "sample" && sample ? sample.answer_correct : (funnel.answer ?? 0) > 0.5;

  return (
    <div>
      <p className="text-xs text-slate-500 mb-4">
        LLM 基于注入的记忆生成答案。
        {mode === "overall"
          ? ` 整体上,证据齐全时的答对率为 ${pct(accGivenGold)}——${accGivenGold != null && accGivenGold < 0.6 ? "利用能力偏弱,证据到手也常答错" : "利用能力尚可"}。`
          : ""}
      </p>
      <div className="flex items-center justify-center gap-4 py-6">
        <motion.div initial={{ x: -20, opacity: 0 }} animate={{ x: 0, opacity: 1 }} className="rounded border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-600">
          注入的记忆
        </motion.div>
        <motion.div initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ delay: 0.3 }} className="text-2xl">
          🤖
        </motion.div>
        <motion.div className="text-slate-300">→</motion.div>
        <motion.div
          initial={{ x: 20, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          transition={{ delay: 0.6 }}
          className={`rounded px-3 py-2 text-sm font-medium ${correct ? "bg-emerald-100 text-emerald-700" : "bg-red-100 text-red-700"}`}
        >
          {mode === "sample" && sample ? sample.pred_answer || "(空)" : correct ? "答对" : "答错"}
          {correct ? " ✓" : " ✗"}
        </motion.div>
      </div>
      {mode === "sample" && sample && !sample.answer_correct && (
        <div className="text-center text-xs text-slate-400">正确答案应为:{sample.gold_answer}</div>
      )}
    </div>
  );
}
