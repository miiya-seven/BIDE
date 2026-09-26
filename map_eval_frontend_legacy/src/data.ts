// 数据类型 + 加载器。数据来自 public/data/*.json(由 scripts/build_data.py 生成)。
// 五阶段流水线:构建 build → 检索 retrieval → 注入 inject → 利用 utilize → 回答 answer

export type Benchmark = "locomo";

export const BENCHMARK_LABELS: Record<Benchmark, string> = {
  locomo: "LoCoMo",
};

export interface OverviewRow {
  system: string;
  n: number;
  answer_accuracy: number;
  retrieval_hit: number | null;
  prompt_hit: number | null;
  gold_in_prompt: number | null;
  avg_total_tokens: number | null;
  avg_latency: number | null;
  oracle?: "upper" | "lower" | null;
}

export interface Funnel {
  gold_exists: number | null;
  build: number | null;
  retrieval: number | null;
  inject: number | null;
  utilize: number | null;
  answer: number | null;
  accuracy_given_gold: number | null;
  failure_stages: Record<string, number>;
  n: number;
}

export interface SystemMechanism {
  build: string;
  store: string;
  retrieve: string;
  inject: string;
  llm_extract: boolean;
  oracle?: "upper" | "lower";
}

// 系统级机制画像(整体特征,非单题)
export interface MechanismProfile {
  build: {
    avg_memory_entries: number | null;
    min_memory_entries: number | null;
    max_memory_entries: number | null;
  };
  retrieval: {
    avg_retrieved: number | null;
    gold_avg_rank: number | null;
    gold_hit_count: number;
    score_mean: number | null;
    score_min: number | null;
    score_max: number | null;
  };
  inject: {
    avg_injected: number | null;
    avg_prompt_tokens: number | null;
  };
  answer: {
    accuracy_given_gold: number | null;
  };
  n: number;
}

export interface RetrievedMemory {
  content: string;
  score: number | null;
  rank: number | null;
  source_ids: string[];
}

export interface SampleTrace {
  sample_id: string;
  question: string;
  gold_answer: string;
  pred_answer: string;
  category: string;
  answer_correct: boolean;
  failure_stage: string;
  retrieval_hit: boolean;
  prompt_hit: boolean;
  gold_in_prompt: boolean;
  num_memory_entries: number | null;
  gold_evidence: { source_id: string; text: string }[];
  retrieved_top: RetrievedMemory[];
  retrieved_source_ids: string[];
  retrieved_gold_ids: string[];
}

// 五阶段定义(展示用)
export const STAGES = [
  { key: "gold_exists", label: "gold 信息存在", short: "起点" },
  { key: "retrieval", label: "检索命中", short: "检索" },
  { key: "inject", label: "注入 prompt", short: "注入" },
  { key: "utilize", label: "证据齐全", short: "利用" },
  { key: "answer", label: "最终答对", short: "回答" },
] as const;

// 失败阶段的中文标签 + 颜色(遵循诊断语义)
export const FAILURE_LABELS: Record<string, { label: string; color: string }> = {
  success: { label: "成功", color: "#22c55e" },
  retrieval: { label: "检索失败(没捞到)", color: "#ef4444" },
  retrieval_or_prompt_partial: { label: "部分证据(注入残缺)", color: "#f59e0b" },
  answer_generation: { label: "证据齐全仍答错(利用失败)", color: "#a855f7" },
  unknown: { label: "未知", color: "#94a3b8" },
};

// 加载器:静态导出下用相对路径 fetch public/data
async function loadJSON<T>(path: string): Promise<T> {
  const res = await fetch(`/data/${path}`);
  if (!res.ok) throw new Error(`加载失败: ${path}`);
  return res.json();
}

export const loadOverview = () =>
  loadJSON<Record<Benchmark, OverviewRow[]>>("overview.json");
export const loadFunnel = () =>
  loadJSON<Record<Benchmark, Record<string, Funnel>>>("funnel.json");
export const loadSystems = () =>
  loadJSON<Record<string, SystemMechanism>>("systems.json");
export const loadMechanism = () =>
  loadJSON<Record<Benchmark, Record<string, MechanismProfile>>>("mechanism.json");
export const loadSamples = (bench: Benchmark, sys: string) =>
  loadJSON<SampleTrace[]>(`samples/${bench}__${sys}.json`);

export function pct(v: number | null | undefined): string {
  if (v === null || v === undefined) return "—";
  return (v * 100).toFixed(1) + "%";
}
