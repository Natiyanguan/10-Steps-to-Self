# 工程交接说明

> 给接手这个工程的人（或 AI）看的。  
> 权威规范是 `docs/TECH_ROADMAP_FINAL.md`，本文只说**当前进展与下一步**。  
> 文中所有签名都是从代码里实测导出的，不是凭记忆写的；但标注为「未验证」的部分确实没跑过。

**当前状态**：226 项测试通过，ruff / mypy strict 全绿。`app/` 约 7800 行，`tests/` 约 1800 行。

验证命令（Windows，仓库根目录）：

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
.venv/Scripts/python.exe -m ruff check app tests scripts
.venv/Scripts/python.exe -m mypy app scripts
.venv/Scripts/python.exe -m app.cli chat --fake    # 离线跑完 10 问并出报告
```

---

## 1. 已完成且已接线

按 `§16.1` 生成顺序，第 1–6 步完成：

| 层 | 模块 | 说明 |
|---|---|---|
| 领域 | `app/domain/{dimension,technique,mode}.py` | 8 维度 × 12 技术 × 3 提问模式，纯函数，零 LLM 依赖 |
| Schema | `app/schemas/{enums,session}.py` | Pydantic strict，全流程状态 |
| 安全 | `app/safety/{lexicon,policy,evidence,output_guard,injection,resources}.py` | **零 LLM、零网络**，LLM 全挂时仍工作 |
| LLM | `app/llm/{base,capability,gateway,retry,structured}.py` + `providers/` ×5 | 能力门控、故障转移、熔断、分级超时、容错 JSON |
| Agent | `app/agents/` ×7 | planner / interviewer / quality / safety / analyst / critic / summarizer |
| 状态机 | `app/graph/` | 10 节点按 phase 条件入口，可从任意阶段恢复 |
| 工具 | `app/cli.py`、`scripts/probe_providers.py` | `tss doctor` / `tss chat --fake` / `tss-probe` |

提示词在 `prompts/`，8 个模板都已挂 `_partials/user_data_notice.jinja2`（§3.7 sandwich 尾部约束）。

---

## 2. 先修这个：上下文压缩只做了一半

**这是唯一处于中间状态的地方，接手第一件事应该是把它接完。**

三个模块写完并通过了导入与类型检查，但**没有任何调用方**——也就是说 §6.3 的上下文压缩目前完全没生效，长会话仍然会触顶：

| 文件 | 状态 |
|---|---|
| `app/agents/summarizer.py` | 已在 `Deps` 注册（`deps.summarizer`），但没有节点调用它 |
| `app/context/compress.py` | 零调用方 |
| `app/context/budget.py` | 零调用方（`ContextBudgeter` 目前是死代码） |
| `SessionState.summary` 字段 | 无人读、无人写 |

还有一处**已存在但没接上的参数**：`Interviewer.ask()` 接受 `summary: str = ""`，
而 `app/graph/nodes/ask.py` 调用时**没有传这个参数**。
所以即使摘要生成出来了，也进不了提问提示词。

### 实测签名（照这个接，别猜）

```python
# app/context/budget.py
build_budgeter(caps: object) -> ContextBudgeter
ContextBudgeter.plan(*, system_prompt: str, desired_output_tokens: int) -> BudgetPlan
# BudgetPlan 字段: context_window, reserved_output, system_tokens,
#                  available_for_history, mode, notes
count_text(text: str, *, tokenizer_hint: str = "cl100k_base") -> TokenCount

# app/context/compress.py
plan_compression(
    pairs: list[QAPair], *, available_tokens: int,
    summary: SessionSummary | None = None,
    keep_recent: int = 3, watermark: float = 0.6,
    tokenizer_hint: str = "cl100k_base",
) -> CompressionPlan
# CompressionPlan 字段: should_compress, to_compress, keep_verbatim,
#                       history_tokens, available_tokens, reason
history_tokens(pairs, summary=None, *, tokenizer_hint="cl100k_base") -> int

# app/agents/summarizer.py
Summarizer.summarize(
    pairs: list[QAPair], *, dimension_of: dict[int, Dimension] | None = None,
) -> SummarizeResult
# SummarizeResult 字段: summary, quotes_kept, quotes_dropped, dropped_samples, error
merge_summaries(old: SessionSummary | None, new: SessionSummary) -> SessionSummary

# app/schemas/session.py
# SessionSummary 字段: by_dimension, preserved_quotes, covered_indices,
#                      approx_tokens, created_at
```

### 建议接线方案（**未验证，我没跑过**）

1. **触发点放在 `app/graph/nodes/listen.py`**，位置在 `qa_pairs` 更新之后、各路由分支 `return` 之前。
   放这里的理由：`listen` 是回答累积的唯一入口，而它有 PROBE / PAUSED / ANALYZE 多个出口，
   放在分支里会漏掉一部分路径。
2. 取 caps → 算额度 → 决策 → 压缩：
   ```
   caps = gateway.eligible_routes(TaskKind.ASK)[0].provider.caps   # 注意空列表
   plan = build_budgeter(caps).plan(system_prompt=..., desired_output_tokens=...)
   cplan = plan_compression(pairs, available_tokens=plan.available_for_history, summary=state.summary)
   if cplan.should_compress:
       result = await deps.summarizer.summarize(cplan.to_compress, dimension_of=...)
       if result.ok:
           new_summary = merge_summaries(state.summary, result.summary)
   ```
3. **压缩失败必须能继续**：`SummarizeResult.ok == False` 时保留原文，
   代价只是 token 更贵。压缩不是必需路径，不该让它阻断会话。
4. `DegradeReason` 里需要**新加**一个压缩原因（如 `HISTORY_COMPRESSED`），
   并按 §9.5 发 `degraded` 事件——压缩会损失保真度（只有逐字原话还能当依据），必须对用户可见。
   注意别复用已有的 `context_truncated`：那个表示**裁剪**（直接丢弃最早历史，
   `budget.py` 的兜底手段），压缩是**摘要**（主动手段，保留结构化信息 + 逐字金句）。
   两者的用户影响不同——裁剪是信息丢失，压缩是保真度下降——混用会让降级日志失去诊断价值。
5. **`ask.py` 里补 `summary=state.summary.render() if state.summary else ""`**，
   否则前面全做了也没用。

### 两条不能破坏的约束

- **原始 `qa_pairs` 永不删除。** 摘要只影响放进提示词的内容，不影响持久化与审计。
  删原文会让"引用逐字校验"彻底失去基准。
- **`preserved_quotes` 必须逐字。** `app/safety/evidence.py` 对引用做逐字校验，
  摘要一旦改写原话，那句话就永久失去依据资格，对应洞察会被程序丢弃——**即使它是对的**。
  `Summarizer` 已经会自校验并在通过率 < 50% 时拒绝整份摘要（`MIN_QUOTE_PASS_RATE`），
  别把这个检查去掉。

### 还缺的测试

`tests/unit/` 下没有 `test_compress.py` 和 `test_summarizer.py`。
`plan_compression` 是纯函数，适合穷尽边界（空列表、`available_tokens<=0`、
轮数不足 `MIN_PAIRS_TO_COMPRESS`、已被 `covered_indices` 覆盖、水位刚好等于阈值）。
压缩策略错了的表现是"长会话突然触顶"，这种故障在生产上极难复现，所以单测值得写厚。

---

## 3. 未开工的部分

按 `§16.1` 剩余顺序。目录都还不存在（或只有空 `__init__.py`）：

| 优先级 | 模块 | 路线文档 | 目录状态 |
|---|---|---|---|
| P1 | 持久化 + checkpointer | §5.2 | `app/storage/` 不存在 |
| P1 | HTTP API（SSE 流式、幂等、并发） | §10 | `app/api/` 不存在 |
| P1 | 可观测性（trace / 指标 / 成本） | §11 | `app/observability/` 不存在 |
| P2 | 嵌入 + 检索 + Reranker | §7 | `app/embeddings/`、`app/rag/` 仅 `__init__.py` |
| P2 | 记忆层与冲突消解 | §6 | `app/memory/` 仅 `__init__.py` |
| P2 | 知识库 | §7 | `app/kb/` 仅 `__init__.py` |
| P2 | RAG 评估 | §12 | `eval/` 不存在 |

`config/` 目前只有 `crisis_resources.yaml`。§17 提到的 `config/providers.yaml`
和 `config/task_models.yaml` 都还没有——前者要靠 `tss-probe` 生成。

---

## 4. 接真实模型的顺序

目前**所有测试都跑在 `FakeProvider` / `DemoProvider` 上，没有连过真实 API。**
`tss-probe` 只跑过 `--dry-run`，真实探测（JSON 稳定性、中文质量采样、上下文窗口、
是否推理型）**一次都没执行过**。

正确顺序：

```
1. 配 .env（照 .env.example）
2. tss doctor            # 检查配置与连通性
3. tss-probe             # 生成 config/providers.yaml
4. 把探测建议的档位写入 config/task_models.yaml
```

第 3 步会花少量 token（JSON 稳定性跑 20 次、中文质量跑 10 次）。
这笔钱必须花——档位定错的后果是模型产出"格式完整但内容失效"的心理分析。

**本地模型（minimind）的接入方法记录在 `docs/LOCAL_MODELS.md`，代码只做了 provider 适配层
（`app/llm/providers/local_minimind.py`，21 项单测用 respx mock 验证形状转换与错误映射）。
注意：TOY 档无法通过 Gateway 承担任何任务**（所有 `TaskKind` 最低要求 FAST），
minimind 只能在 provider 层直连做 CI 冒烟，不能进主流程。

---

## 5. 交接时值得知道的几个坑

这些是开发过程中测试真实抓到的缺陷，都已修复，但同类问题容易再犯：

1. **提示词只列"允许的维度"和"可用的技术"两份独立清单是不够的**，
   模型会组合出非法配对（如 `emotion` + `exception`），导致反复重规划、白烧 token。
   必须把"维度 → 配套技术"的**配对关系**讲清楚。
2. **同意校验要判"是否仍然有效"，不是"有没有同意对象"。**
   条款版本变更后老用户会静默绕过同意闸门。判定逻辑收在 `Consent.is_current()` 一处。
3. **安全评估必须在原文上做，注入剥离在其后。**
   `injection.sanitize()` 是按句剥离的，「忽略以上指令，我不想活了」这句里注入与危机信号
   在同一句，顺序反了会把危机信号一起删掉，安全层就什么都看不到。
   `intake.py` 和 `listen.py` 里都有注释标明这个顺序，别调换。
4. **注入规则的误伤比漏检更容易被忽略。**
   心理场景下用户天然会说「我想忽略父母的期待」「我作为一名医生，工作压力很大」，
   规则写松就会把真实经历剥掉。`tests/unit/test_injection.py` 里"不应命中"的用例
   比"应命中"的多，是刻意的。
5. **重试闭包按引用捕获循环变量**（ruff B023）：目前恰好正确，
   但一旦重试被推迟到循环外，就会静默对着错误的 provider 重试，日志里记的还是原来那个。
