# 当前主实验：实现与方案思想

更新：2026-09-26。本文档只描述当前 Full 主实验实现。

## 1. 方案思想

BIDE 的核心不是把原始问题直接当作唯一检索向量，而是先把问题转换为可用于记忆访问的结构化查询表示。该表示保留实体、角色、关系、事件、值、时间、状态/模态等条件，使不同检索通道能够从不同角度寻找支持问题的历史记录。

记忆侧在回答前离线组织原始对话：Raw 保留原始文本、说话人、时间和图片来源；更高层视图提供命题、结构化断言、来源关联、实体/事件信息和索引。回答时，候选最终仍回到 Raw 记录，以保留可读的原文和来源关系。

在线流程是“问题解析 → 多路候选获取 → 候选融合 → 统一重排 → 证据整理 → Reader”。它强调候选证据的互补性和事件/时间范围，但当前 Full 没有单独执行一个稳定的逐需求闭合器，也没有将三态 evidence state 作为已验证的独立模块输入 Reader。

## 2. 实际流程

```text
LoCoMo question Q
  → canonical Query Frame / query_text
  → six retrieval families over organized memory
  → raw-ID projection
  → family-internal best-rank aggregation
  → cross-family reciprocal-rank fusion
  → 128 unique Raw candidates
  → Qwen3 explicit-pair reranker
  → Top-10 or Top-20 Raw records
  → compact source-aware evidence serialization
  → extract-first Reader
  → official Judge
  → frozen repair/adjudication selection for the retained final result
```

## 3. 输入与查询表示

数据集为 LoCoMo Cat1–Cat4，共1540题。每个问题有 conversation-local Raw corpus。当前查询文件为 `runs/current_answer_inputs/V41_QUERIES_REBUILT.jsonl`，其中同时保存原始 `question`、结构化 `query_v41` 和序列化 `query_text`。

Query Frame 的字段包括但不限于：目标实体、角色、关系短语、事件短语、值/属性短语、时间短语、query kind、modality 和 polarity。它是从问题得到的检索表示，不是答案或 gold evidence；生成过程不读取参考答案。

## 4. 记忆与检索家族

Full 使用六个检索家族，并将各家族结果映射回原始 Raw ID：

- Clause：原始记录的短句/视图级匹配；包括问题和结构化查询视图，以及说话人/时间视图。
- Occurrence：词法与稠密结果的组合，并使用记录/来源关联产生 occurrence 相关排序。
- Structural：会话片段、角色事实、实体匹配和结构一跳等排序。
- Multivector：问题/结构化查询与记录局部文本的词元级语义匹配。
- SPLADE：稀疏语义检索；当前运行 receipt 记录其实际 backend。
- Typed L2：在类型化结构断言上使用 semantic、owner、value 视图，再映射回 Raw。

Clause、Multivector、SPLADE 等通道仍可直接使用 Raw 文本视图；结构化家族额外使用离线构建的 L1/L2/L3 视图。不同家族不是串行六步，而是并行候选来源。

## 5. 候选融合与重排

各家族内部对同一 Raw 的多个 lane 取最好名次；Typed L2 先进行族内 RRF。跨家族的当前基线使用 reciprocal-rank fusion：

```text
score(r) = sum_f 1 / (20 + rank_f(r))
```

候选按分数降序、Raw ID 升序打破平局，保留128条唯一 Raw 记录。随后将结构化 `query_text` 与128条候选的 `retrieval_text` 送入 Qwen3 explicit-pair reranker，得到最终排序。Top10 和 Top20 是同一排序的前缀，不是两次独立检索。

当前 Full 的近期复现检索指标（1536道有标注来源题）为：Candidate128 Any/Exact = 98.37%/93.10%；Top10 = 93.23%/81.90%；Top20 = 95.05%/86.39%。Any 表示至少命中一条标注来源，Exact 表示全部命中；它们不是 QA 准确率。

## 6. 重排后的证据组织与 Reader

TopK 记录被转换为紧凑的原文包，保留 Raw ID、原文、speaker、time、image caption、retrieval rank，并按来源 session/turn 顺序组织，使连续回复、时间关系和不同事件仍可辨别。该阶段的作用是减少冗余、保留来源和上下文关系、提供可读证据。

extract-first Reader 接收问题与组织后的证据包，负责跨记录整合、主体/事件/时间范围判断、列表或计数回答以及最终答案生成。它输出答案文本，不输出一个经过独立验证的 evidence certificate。后续历史修复与争议复核只用于当前保留结果的最终选择，不能被解释为单次 Full Reader 的模块增益。

## 7. 当前结果口径

| 条件 | Top10 | Top20 |
|---|---:|---:|
| Full 单次 Reader | 1346/1540 = 87.40% | 1366/1540 = 88.70% |
| Full 最终保留 | 1388/1540 = 90.13% | 1409/1540 = 91.49% |

最终保留结果来自 Full 单次答案之后的既定成功修复和正式复核；Top20 由单次1366题、修复/选择增量和复核增量形成最终1409题。主论文采用用户确认的最终 Full Top20 91.49%。单次 Reader 结果用于统一消融基线。

## 8. 与论文抽象的对应边界

论文中的 `D(Q)` 可对应 Query Frame，`C_Q` 可对应融合后的 Candidate128，`E_Q` 可对应 TopK 后的紧凑证据包。当前 `Z_Q` 更准确地表示证据组织/支持说明，而不是已经稳定实现并独立验证的 satisfied/unmet/conflicting 闭合状态。

因此可以声称：BIDE 使用需求感知的多路记忆访问、候选融合、统一重排和来源保持的证据组织来回答问题。不能仅凭当前 Full 结果声称每个需求都被独立检索、显式三态闭合、或由 alignment 状态严格控制最终预算选择。

## 9. 关键代码与产物

- 查询与原文输入：`runs/current_answer_inputs/V41_QUERIES_REBUILT.jsonl`、`V41_RAW_VIEWS.jsonl`。
- 当前 Full 排名：`runs/current_answer_inputs/RANKINGS.json`。
- 检索协议：`runs/locomo_answer_judge_20260922/full/cost_and_ablations/upstream/PROTOCOL.json`。
- 检索构建：`src/best_memory/retrieval/{clause,occurrence,structural,multivector,splade,typed_l2,fusion}.py`。
- Reader/回答协议：`src/best_memory/answering/full/PROTOCOL.json` 及 `cost_and_ablations/compact_answer_only_full_coherent_extract_full/`。
- 当前 Full 单次和最终结果：`compact_answer_only_full_coherent_extract_full/results/` 与 `hypothetical_fusion_analysis/SUMMARY.json`。

