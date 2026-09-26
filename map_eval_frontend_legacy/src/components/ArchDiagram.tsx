"use client";

import { SystemMechanism } from "@/lib/data";

// 架构拓扑图(通用五阶段模板打底)。节点内容来自 systems.json 的真实机制画像。
// 后续可对差异大的系统(mem0/memorybank/dual_layer)特化专属拓扑。

const NODES = [
  { key: "build", label: "构建", color: "#3b82f6", desc: (m: SystemMechanism) => m.build },
  { key: "store", label: "存储", color: "#8b5cf6", desc: (m: SystemMechanism) => m.store },
  { key: "retrieve", label: "检索", color: "#f59e0b", desc: (m: SystemMechanism) => m.retrieve },
  { key: "inject", label: "注入", color: "#f97316", desc: (m: SystemMechanism) => m.inject },
  { key: "answer", label: "回答", color: "#22c55e", desc: () => "LLM 生成答案" },
];

export default function ArchDiagram({ mech }: { mech: SystemMechanism }) {
  return (
    <div className="w-full">
      <div className="flex items-stretch gap-1 overflow-x-auto pb-2">
        <FlowNode label="原始对话" color="#94a3b8" desc="多会话历史" />
        <Arrow />
        {NODES.map((n, i) => (
          <span key={n.key} className="contents">
            <FlowNode label={n.label} color={n.color} desc={n.desc(mech)} />
            {i < NODES.length - 1 && <Arrow />}
          </span>
        ))}
      </div>
      {mech.llm_extract && (
        <div className="mt-2 text-xs text-slate-400">
          ⚙ 构建阶段用 LLM 抽取
          {mech.build.includes("黑盒") && "(SDK 黑盒:只知输入输出,内部抽取不可见)"}
        </div>
      )}
      {mech.oracle && (
        <div className="mt-1 text-xs text-amber-600">
          ⚠ {mech.oracle === "upper" ? "oracle 参照上界,非真实系统" : "对照下界(无记忆)"}
        </div>
      )}
    </div>
  );
}

function FlowNode({ label, color, desc }: { label: string; color: string; desc: string }) {
  return (
    <div
      className="rounded-lg border-2 px-3 py-2 min-w-[110px] max-w-[150px] shrink-0"
      style={{ borderColor: color, background: color + "12" }}
    >
      <div className="text-xs font-semibold" style={{ color }}>
        {label}
      </div>
      <div className="text-[11px] text-slate-600 mt-1 leading-snug">{desc}</div>
    </div>
  );
}

function Arrow() {
  return (
    <div className="flex items-center shrink-0 text-slate-300 text-lg px-0.5">→</div>
  );
}
