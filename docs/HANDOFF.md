# 工程交接说明

> 给接手这个工程的人（或 AI）看的。  
> 权威规范是 `docs/TECH_ROADMAP_FINAL.md`，本文只说**当前进展与下一步**。  
> 文中所有签名都是从代码里实测导出的，不是凭记忆写的；但标注为「未验证」的部分确实没跑过。

**当前状态**：289 项测试通过，ruff / mypy strict 全绿。`app/` 约 9100 行，`tests/` 约 2900 行。
P0 + P1 已全部完成（§16.1 第 1–9 步）；P2（RAG / 记忆 / 知识库 / 评估）未开工。

验证命令（Windows，仓库根目录）：

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
.venv/Scripts/python.exe -m ruff check app tests scripts
.venv/Scripts/python.exe -m mypy app scripts
.venv/Scripts/python.exe -m app.cli chat --fake    # 离线跑完 10 问并出报告
.venv/Scripts/python.exe -m uvicorn app.main:app --reload   # 需要先配 .env
```

---

## 1. 已完成且已接线

按 `§16.1` 生成顺序，第 1–9 步完成（P0 + P1 全部）：

| 层 | 模块 | 说明 |
|---|---|---|
| 领域 | `app/domain/{dimension,technique,mode}.py` | 8 维度 × 12 技术 × 3 提问模式，纯函数，零 LLM 依赖 |
| Schema | `app/schemas/{enums,session}.py` | Pydantic strict，全流程状态 |
| 安全 | `app/safety/{lexicon,policy,evidence,output_guard,injection,resources}.py` | **零 LLM、零网络**，LLM 全挂时仍工作 |
| LLM | `app/llm/{base,capability,gateway,retry,structured}.py` + `providers/` ×5 | 能力门控、故障转移、熔断、分级超时、容错 JSON |
| Agent | `app/agents/` ×7 | planner / interviewer / quality / safety / analyst / critic / summarizer |
| 状态机 | `app/graph/` | 10 节点按 phase 条件入口，可从任意阶段恢复 |
| 持久化 | `app/storage/{db,models,repo}.py` | SQLAlchemy async + SQLite；SessionState 快照 + revision 乐观锁 + 危机审计 append-only |
| HTTP API | `app/api/{app,errors,schemas,state,deps}.py` + `routes/` | 创建/提交/获取会话、健康检查、Prometheus 指标、错误码映射 |
| 可观测性 | `app/observability/{metrics,logging}.py` | Prometheus 指标 + loguru JSON 日志（隐私脱敏） |
| 工具 | `app/cli.py`、`scripts/probe_providers.py` | `tss doctor` / `tss chat --fake` / `tss-probe` |

提示词在 `prompts/`，8 个模板都已挂 `_partials/user_data_notice.jinja2`（§3.7 sandwich 尾部约束）。

**关于持久化的取舍**：P1 用 SQLite + 自己的 SessionState 快照 + revision 乐观锁，
**没有**用 LangGraph checkpointer。理由：LangGraph 的 checkpoint 序列化含 dataclass 的
GraphState（emissions/safety_action）有风险，而 SessionState 是纯 pydantic、JSON 可往返。
revision 乐观锁是 §5.1/§5.2 明确设计的并发手段。P5 生产化时再评估是否引入
Postgres checkpointer。

---

## 2. 上下文压缩（已接完）

> **状态：已完成。** 下表原本是"待办"，现在全部有调用方了。保留本节是为了让后人
> 理解这套接线是怎么搭起来的，以及踩过的两个坑（见文末）。

压缩链路的完整接线：`listen._maybe_compress` → `_history_budget`（估额度）→
`plan_compression`（纯函数决策）→ `Summarizer.summarize`（生成 + 自校验）→
`merge_summaries`（增量合并）→ 写入 `state.summary`，再由 `ask` 节点传入 interviewer。

已接线文件：
- `app/graph/nodes/listen.py`：`_history_budget` / `_maybe_compress` 两个新函数，
  在 `qa_pairs` 更新后、路由分支前调用
- `app/graph/nodes/ask.py`：调 `interviewer.ask()` 时传 `summary`
- `app/schemas/enums.py`：新增 `DegradeReason.HISTORY_COMPRESSED`
- 测试：`tests/unit/test_compress.py`（17）、`tests/unit/test_summarizer.py`（19）、
  `tests/integration/test_compression.py`（4）

接线中发现并已修的两个坑（写进 §5 了）：

1. **系统提示词 token 估算必须用中文**：`"x" * 1500` 会被 BPE 压成 225 token，
   严重高估可用额度、推迟压缩触发，违背"宁可早压缩"。
2. **压缩能否落地受 SUMMARIZE 门控约束**：SUMMARIZE 要求 `min_context ≥ 8000` 且
   `min_output ≥ 800`。provider 窗口落在 `[4000, 8000)` 时能触发压缩、
   但 summarize 会被能力门控拒绝——那是正确的降级（保留原文继续），不是 bug。

---

<details>
<summary>以下是完成前的原始待办说明（已过时，仅存档）</summary>

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

</details>

---

## 3. 未开工的部分（P2）

按 `§16.1` 剩余顺序（第 10、12、14 步）。目录都只有空 `__init__.py`（或不存在）：

| 模块 | 路线文档 | 目录状态 | 依赖 |
|---|---|---|---|
| 嵌入 + 检索 + Reranker | §7 | `app/embeddings/`、`app/rag/` 仅 `__init__.py` | 智谱 embedding API / 本地 BGE-M3 |
| 记忆层与冲突消解 | §6 | `app/memory/` 仅 `__init__.py` | 嵌入层（用户历史检索） |
| 知识库构建管线 | §7.7 | `app/kb/` 仅 `__init__.py` | Bocha / Tavily / OpenAlex（离线） |
| RAG 评估 + 回归门禁 | §12 | `eval/` 不存在 | 检索层 + 标注集 |

**P2 是智能增强，不是"让系统能跑"的必要条件。** 系统现在已能对外服务
（创建会话 → 提问 → 分析报告），P2 加的是"跨会话记忆""提问范式检索""评估闭环"。
它们都依赖外部服务（embedding / 向量库 / 搜索 API），无法纯离线完整验证。

`config/` 目前只有 `crisis_resources.yaml`。§17 提到的 `config/providers.yaml`
和 `config/task_models.yaml` 都还没有——前者靠 `tss-probe` 生成。

### P1 已知的简化（P5 生产化时补）

- **幂等**：§10.1 要求 `Idempotency-Key`，P1 用 revision 乐观锁兜底，真正的
  Redis 幂等 + 分布式锁留到 P5（见 `routes/sessions.py` 注释）。
- **SSE 流式**：§10.2 的事件类型已定义，但 P1 的 API 是"一轮一响应"，
  没有做真正的流式推送（`events` 数组在响应体里一次性返回）。
- **OTel trace**：§11.2 的 span 覆盖没接（需要 OpenTelemetry SDK），
  只有 loguru JSON 日志 + Prometheus 指标。
- **alembic 迁移**：P1 用 `create_all`，P5 换 Postgres 时再引入迁移。

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
6. **系统提示词 token 估算必须用中文**（压缩接线时踩到）：`"x" * 1500` 会被 BPE 压成
   225 token，而真实中文提示词约 1200 token。用 ASCII 估算会严重高估可用额度、
   推迟压缩触发，正好违背"宁可早压缩，不要晚触顶"。见 `listen.py` 的 `_SYSTEM_PROMPT_ESTIMATE`。
7. **能力门控的 min_context / min_output 是硬约束**：SUMMARIZE 要求
   `min_context ≥ 8000` 且 `min_output ≥ 800`。窗口落在 `[4000, 8000)` 的 provider
   能过 ASK（min 4000）却过不了 SUMMARIZE——压缩会触发但 summarize 被门控拒绝，
   这是正确的降级（保留原文继续），不要当成 bug 去"修"。
8. **乐观锁的 `revision` 列与 `state_json` 必须同步**（storage 层踩到）：UPDATE 时
   如果只把 `revision` 列 +1、而 `state_json` 里序列化的还是旧版本，下次 load 会读回
   旧版本号，乐观锁形同虚设——而且不报错，只在第二次提交时莫名 409。
   已修（`repo.py::update_with_revision` 序列化 `bumped`），并有回归测试
   `test_revision_persists_after_reload` 守着。
9. **裸 async generator 不支持 `async with`**：写 `async def session_ctx(): yield` 但
   忘加 `@asynccontextmanager` 装饰器，会在运行时抛
   "async_generator object does not support the asynchronous context manager protocol"。
   见 `storage/db.py`。
10. **持久化对象是 SessionState，不是 GraphState**：GraphState 含 dataclass
    （emissions / safety_action）和运行时字段（inbox），不该进库。持久化前用
    `extract_session_state` 剥掉运行时字段，恢复时从 SessionState 重新构造 GraphState。
11. **IntEnum 字段的 API 序列化**：`Sharpness` 是 IntEnum，FastAPI 默认只接受整数
    （1/2/3），传 "moderate" 会 422。`api/schemas.py` 加了 `field_validator(mode="before")`
    把字符串名转成枚举，客户端才能传友好字符串。
