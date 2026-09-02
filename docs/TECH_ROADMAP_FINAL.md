# 10-Steps-to-Self 技术路线文档（终版 v3.0）

> 一个基于「苏格拉底式反向提问」的自我探索 Agent。
> 本文档合并了两份前置技术路线（`TECH_ROADMAP.md` 与 `技术路线FromDeepSeek.txt`），
> 并在 v3.0 中**按实际可用的 API 与本地模型条件定稿**，
> 做了显式取舍并保留理由，**用于直接驱动下游 AI 编码工具生成工程代码**。

**v3.0 相对 v2.0 的变化**（由实际供应商条件驱动，见 §0.1 决策 #15-#24）：
云端 LLM 定为 DeepSeek，嵌入定为智谱 embedding-3，二者均可切本地模型；
新增**模型能力分级与门控**（防止弱模型静默劣化输出）、**三档运行档位**（cloud/hybrid/local）、
**知识库构建管线**（Bocha/Tavily/OpenAlex，离线运行）、**供应商能力探测清单**；
修订了结构化输出策略、向量库选型与维度、报告分段生成。

- 目标环境：Python 3.12（项目现状） / Windows 11 开发 + Linux 容器部署
- 文档约定：`MUST` 必须实现 · `SHOULD` 建议实现 · `LATER` 明确延后 · `REJECTED` 明确不做
- 每个 `###` 小节自包含，可单独作为一次代码生成任务的输入
- `【生成指令】` 是给编码工具的硬约束（文件路径、可测性要求）
- `【决策】` 标记两份前置文档冲突处的最终选择与理由，**下游工具不得擅自改回**

---

## 0. 两版差异与合并决策（先读这节）

两份文档的分歧不是细节，而是架构哲学。**不能叠加，必须选择。**

| # | 议题 | 我方版本 | DeepSeek 版本 | 终版决策 | 理由 |
|---|---|---|---|---|---|
| 1 | 服务架构 | 模块化单体 + LangGraph | 微服务 + RabbitMQ/Kafka + Saga | **模块化单体起步，接口按可拆分设计** | 产品未验证提问质量前，分布式带来的最终一致性、Saga 补偿、跨服务调试成本是纯负债 |
| 2 | 编排方式 | LangGraph StateGraph + Checkpointer | 对话管理服务 + 事件驱动 | **LangGraph** | 提问流程需要中断/续跑/回溯/条件短路，状态机原生支持；事件驱动要自己实现这些 |
| 3 | 异步任务 | 无（同步 + SSE） | MQ 全链路异步 | **仅报告生成走任务队列** | 60s 级任务必须异步；但每一步都过 MQ 是过度设计 |
| 4 | 向量库 | Chroma | Milvus / Qdrant / Weaviate | **pgvector 起步 → Qdrant（规模化时）** | 已有 Postgres，pgvector 少一个组件、支持事务内一致写；抽象层保证可切换 |
| 5 | 提问模式 | 仅渐进式 | 一次性 / 渐进两种 | **三种模式：一次性 / 分批(3-4-3) / 渐进，默认分批** | 原始方法论明确说"分批或一次性"；分批兼顾深度与延迟 |
| 6 | 报告引用 | 禁止引用文献给用户下结论 | 标注「根据《焦点解决短期治疗》…」 | **保留我方立场，但允许"参考框架"式弱标注** | 权威引用会把探索工具包装成诊断权威，且模型极易编造书名页码 |
| 7 | 危机处置 | 提供本地化热线资源 | 转人工或热线 | **默认只给外部资源；转人工仅在真有值班人力时开启（feature flag）** | 承诺人工而无人接，比不承诺更危险 |
| 8 | 合规声明 | GDPR 式权利 | GDPR + HIPAA | **GDPR / 中国个保法；明确不声称 HIPAA** | 本产品非 HIPAA covered entity，误声称合规是实质法律风险，见 3.8 |
| 9 | 反幻觉主手段 | Evidence 原话子串程序化校验 | RAG 约束 + 辅助 LLM 后验证 | **两者叠加，程序化校验为不可绕过底线** | 模型校验模型仍可能同时出错；字符串匹配不会 |
| 10 | 免责声明频次 | 会话级一次 | 每次响应末尾 | **会话开始 + 报告显著位置 + 危机分支必带；不逐条附加** | 逐条附加导致声明疲劳，用户会完全忽略 |
| 11 | 预生成 | 投机预取检索片段 | 预生成常见困惑的问题模板 | **只预生成参考范式，不预生成最终问题文本** | 预制问题文本直接摧毁个性化，是本产品的核心价值 |
| 12 | 部署 | 容器 + 水平扩容 | K8s + HPA + Serverless 可选 | **容器 → K8s（P5）；Serverless 明确 REJECTED** | 长会话有状态 + SSE 流式 + 冷启动，Serverless 不匹配 |
| 13 | Python 版本 | 3.12 | 3.11 | **3.12** | 项目 `pyproject.toml` 已锁定 `>=3.12` |
| 14 | Reranker 分数接近 | 相对差距判据 + 校准 + 弃检索 | LLM 终审打分 | **两者结合：先用相对差距，仅在关键路径且候选≤5 时才 LLM 终审** | LLM 终审有效但贵且慢，不能作为默认路径 |

### 0.1 v3.0 新增决策（由实际 API 与本地模型条件驱动）

| # | 议题 | 决策 | 理由 |
|---|---|---|---|
| 15 | 云端 LLM | **DeepSeek（OpenAI 兼容协议）**，强档 / 快档双档 | 已有可用 key；中文强、成本低；OpenAI 兼容协议让本地模型复用同一 adapter |
| 16 | 本地 LLM 定位 | **minimind 仅用于开发/CI/离线冒烟，不承担 Planner / Interviewer / Analyst / Critic** | 见 §2.4：minimind 是从零训练的小参数教学模型，不具备本产品所需的长上下文、指令遵循与结构化输出能力。让它跑核心链路会产出"格式完整但内容失效"的结果 |
| 17 | 能力门控 | **Provider 必须声明能力；Gateway 对能力不足的任务拒绝路由并显式降级，禁止"试一试"** | #16 的工程保障。没有门控，切弱模型只会静默劣化，而静默劣化是本文档全篇最想避免的失败模式（§9.5） |
| 18 | 向量库（修订 #4） | **Chroma 承担 P0-P3 → pgvector 在 Postgres 落地后接入，二者同在 `VectorStore` 抽象下** | 新信息推翻原判断：embedding-3 为 2048 维，而 pgvector 的 HNSW/IVFFlat 索引上限为 2000 维，`vector(2048)` 无法建索引（见 §2.6） |
| 19 | 嵌入维度 | **统一收敛到 1024 维**（embedding-3 若支持 `dimensions` 参数则请求降维；否则保留 2048 走 Chroma，pgvector 侧用 `halfvec`） | 1024 同时满足 pgvector 索引上限与本地 BGE-M3 原生维度，使云/本地切换不必改表结构 |
| 20 | 结构化输出（修订 §6.2） | **JSON 模式 + schema 校验 + 修复重试为主路径；function calling 仅在探测确认支持后启用** | DeepSeek 系与本地模型对 function calling 支持不一致，推理型模型常不支持工具调用或温度。把 function calling 当唯一路径会在换模型时整链失效 |
| 21 | 报告生成 | **分段生成（诉求 → 冲突 → 建议）并可续写，不一次生成整份报告** | 实际配置 `LLM_MAX_TOKENS=4096`，推理型模型的思维链还要占额度。整份报告极易触顶截断，且截断发生在 JSON 中间会直接解析失败 |
| 22 | 联网搜索边界 | **Bocha/Tavily 仅用于离线知识库构建与资源核验，绝不接收用户原话** | 把用户心理独白发给搜索引擎是不可逆的隐私泄露，且搜索结果对"提问质量"无帮助。见 §7.7 |
| 23 | 学术检索 | **OpenAlex 用于离线构建带真实出处的技术库，只入库真实 `doi`/`work_id`** | 正面解决决策 #6 的"模型编造书名页码"：引用必须来自库内真实记录，可程序化核验 |
| 24 | 跨厂商降级 | **必须再接一家云厂商作为熔断兜底；智谱同一 key 若可调对话模型则为首选兜底（待实测）** | 当前只有 DeepSeek 一家对话供应商，§9.2 的"备用供应商"是空的——单厂故障即全站不可用 |

**关于 v3.0 的一处说明：**
用户提供的模型标识与"embedding-3 为 2048 维"均按用户实测结论采用。
本文档因此**不在任何地方硬编码模型名与维度**，
而要求启动时按 §17 探测清单实测能力并落盘 ——
这样无论标识与实际能力如何变化，工程都不会建立在过期假设之上。

**两版各自的独有价值，终版全部吸收：**

- 来自我方版本：理论→构件映射与指标、提问技术库与维度矩阵、锐度治理、Evidence 程序化校验、
  Critic 只否决不改写、上下文硬预算与优先级裁剪、记忆冲突不消解、会话级锁+乐观锁+幂等、
  安全层不依赖 LLM 可用性、降级可见性、FakeLLM 与混沌/红队测试、产品级评估与回归门禁。
- 来自 DeepSeek 版本：明确的技术选型表、两种提问模式、物理数据模型（DDL 级）、
  静态/传输加密与合规、内容审核 API、多路召回、LLM 终审、用户反馈闭环、
  专家参与与定期审计、Prometheus/Grafana/Sentry 具体监控栈、HPA 扩缩容依据。
- **两版都缺、终版新增**：知情同意版本化与未成年人处理、供应商数据留存条款（对心理数据是硬门槛）、
  数据留存与删除策略、中文特化（BM25 分词 / embedding / token 计数偏差）、
  显式 API 契约与错误码、抽象 Provider 层、成本模型、会话跨天恢复的状态重确认、
  评估集伦理（不得用真实用户数据）、feature flag、人工反馈回流评估集的闭环。

---

## 1. 定位与理论映射

### 1.1 一句话定义

系统扮演一位「精通苏格拉底提问术的生涯规划 / 心理咨询取向的引导者」，
**不给答案，而是通过 10 个递进式问题把用户潜意识里的真实诉求"接生"出来**，
最后输出「潜意识诉求分析 + 可执行建议」的结构化复盘报告。

核心资产不是"回答能力"，而是**提问质量**与**长期记忆的连贯性**。所有技术选型服务于这两点。

### 1.2 理论到工程的映射（每个理论必须落成可测量构件）

| 理论依据 | 工程构件 | 可观测指标 |
|---|---|---|
| 苏格拉底产婆术 | `QuestionPlanner`：递进式提问策略引擎 + 禁止给建议的输出约束 | 提问深度递进率、"直接给答案"违规率 |
| 乔哈里视窗 | `BlindSpotProbe`：维度覆盖检查器，强制打到用户未提及的维度 | 四象限覆盖度、盲区维度命中数 |
| 焦点解决短期治疗 | `TechniqueLibrary`：奇迹/例外/量尺/循环问句库 | 技术使用分布、单一技术过载告警 |
| 边界对象与认知卸载 | `SessionState` + 单问单答节奏 + 进度可视化 | 单轮认知负荷、中途放弃率 |

### 1.3 提问技术库 — MUST

代码化为枚举 + 模板 + 适用条件，而非散落在提示词里的自然语言：

```python
class Technique(StrEnum):
    MIRACLE      = "miracle"        # 奇迹问句：假设问题已解决，你会注意到什么不同？
    EXCEPTION    = "exception"      # 例外问句：什么时候这个困扰没有出现？
    SCALING      = "scaling"        # 量尺问句：0-10 分你现在在几分？1 分之差是什么？
    CIRCULAR     = "circular"       # 循环问句：如果最亲近的人来回答，他会怎么说？
    COST_OF_LOSS = "cost_of_loss"   # 代价问句：做这个决定你最怕失去什么？
    REFRAME      = "reframe"        # 重构问句：换个角度这件事在保护你什么？
    VALUE_PROBE  = "value_probe"    # 价值观追问：这之所以重要，是因为你在意什么？
    TIME_SHIFT   = "time_shift"     # 时间线：5 年后的你会如何评价现在的犹豫？
```

### 1.4 提问维度矩阵 — MUST

10 问在**维度矩阵**上做覆盖规划，而非随机生成：

`事实层` → `情绪层` → `价值层` → `恐惧层` → `资源层` → `代价层` → `关系层` → `身份层`

规划约束：前 3 问偏事实与情绪（建立安全感）；中间 4 问打价值/恐惧/代价（核心挖掘）；
后 3 问收敛到资源与行动。**任一维度连续两问重复 = 规划失败，触发重规划。**

### 1.5 提问模式（吸收 DeepSeek，三种可配置）— MUST

【决策 #5】原始方法论明确提到"分批或一次性"，我方前版只做渐进式是缺失。

| 模式 | 行为 | 优势 | 代价 | 适用 |
|---|---|---|---|---|
| `ONE_SHOT` | 一次性生成全部 10 问 | 用户可通览、可自选顺序、离线思考 | 无法根据回答调整，深度受限 | 用户明确要求一次性看全 |
| `BATCHED`（默认） | 3 → 4 → 3 分三批 | 兼顾适应性与等待感，批间可重规划 | 实现稍复杂 | 默认路径 |
| `PROGRESSIVE` | 每次 1 问，完全动态 | 深度最佳，可即时追问 | LLM 调用最多、总时长最长 | 高投入用户 / 深度模式 |

- 模式影响 `QuestionPlanner` 的调用节奏，**但不影响维度矩阵约束**：三种模式都必须满足覆盖规划。
- `ONE_SHOT` 下无法基于回答调整，因此 `PlanSlot.depends_on` 必须为空；由校验器强制。
- `BATCHED` 在每批结束后允许**重规划剩余槽位**（用户回答可能揭示新维度），
  但已发出的问题不可撤回。

【生成指令】1.3 / 1.4 实现为 `app/domain/technique.py` 与 `app/domain/dimension.py`，
纯数据 + 纯函数、零 LLM 依赖、100% 单测覆盖。提问模式为 `app/domain/mode.py` 中的枚举 + 策略函数。

---

## 2. 技术选型总览（吸收 DeepSeek 的显式选型，按本项目现状定稿）

| 类别 | 选型 | 备选 / 演进 | 说明 |
|---|---|---|---|
| 语言运行时 | **Python 3.12** | — | 项目 `pyproject.toml` 已锁 `>=3.12`【决策 #13】 |
| Web 框架 | **FastAPI + uvicorn** | — | async 原生、SSE 支持、Pydantic 一体化 |
| 编排 | **LangGraph** + checkpoint | — | 状态机、中断续跑、条件边【决策 #2】 |
| LLM 抽象 | **自建 `LLMGateway`** 封装 | 直接用 LangChain | 需要精细控制超时/熔断/降级/计量，LangChain 封装不够 |
| 强档模型（规划/提问/分析/Critic） | **DeepSeek 强档（OpenAI 兼容协议）** | 第二家云厂商（兜底，决策 #24） | 【决策 #15】能力要求见 §2.4；**代码中不硬编码模型名** |
| 快档模型（安全/质量/摘要/改写） | **DeepSeek 快档** | 本地 Qwen2.5-7B-Instruct 等 | 成本与延迟优化，见 §9.4 |
| 本地模型 | **minimind（仅开发/CI/冒烟）**；生产级本地用 Ollama/vLLM 托管指令模型 | — | 【决策 #16】能力门控见 §2.4 |
| 结构化输出 | **JSON 模式 + schema 校验 + 修复重试** | function calling（探测确认后启用） | 【决策 #20】见 §6.2 |
| 关系库 | **PostgreSQL 16**（开发可 SQLite） | — | 会话、消息、记忆、审计 |
| 向量库 | **Chroma（P0-P3）** | **pgvector**（Postgres 落地后）· Qdrant（规模化） | 【决策 #18】2048 维触及 pgvector 索引上限，见 §2.6 |
| 关键词检索 | **rank-bm25 + jieba（进程内）** | PostgreSQL 全文检索（迁 pgvector 时） | 中文必须先分词，见 §2.2 |
| Embedding（云） | **智谱 `embedding-3`**（实测 2048 维） | — | 目标维度统一 1024，见 §2.6【决策 #19】 |
| Embedding（本地） | **BGE-M3**（原生 1024 维，中文强） | bge-large-zh-v1.5 | 与云端同维，切换不改表结构 |
| Reranker | **bge-reranker-v2-m3（本地）** | 无（降级到 RRF 排名） | 无云端 rerank API，本地是唯一选项，见 §7.4 |
| 缓存 / 锁 | **Redis 7** | — | 分布式锁、幂等键、限流令牌桶、会话热态 |
| 任务队列 | **Redis + arq**（轻量） | Celery / RabbitMQ | 【决策 #3】仅报告生成等长任务用，不做全链路事件驱动 |
| 配置 | **pydantic-settings + .env** | — | 启动即校验，缺失快速失败 |
| 提示词管理 | **jinja2 外置文件 + 版本化** | LangSmith Hub | 改配置即切版本，无需部署 |
| 日志 | **loguru → JSON** | — | 结构化字段固定，隐私红线见 §11.1 |
| Trace | **OpenTelemetry** | LangSmith / Langfuse | LangSmith 便于 LLM 专项调试，SHOULD |
| 指标 | **Prometheus + Grafana** | — | 吸收 DeepSeek |
| 错误追踪 | **Sentry** | — | 吸收 DeepSeek；必须配 PII 脱敏钩子 |
| 内容审核 | **自建词表为主**（供应商 moderation 仅作可选第三信号） | — | 词表层不可省，见 §3.4 |
| 测试 | **pytest + pytest-asyncio + respx** | — | respx mock httpx |
| 质量 | **ruff + mypy strict** | — | CI 门禁 |
| 容器 | **Docker + docker-compose**（开发） | **K8s + HPA**（P5） | 【决策 #12】Serverless REJECTED |
| MCP | **mcp SDK** | — | 对外暴露工具能力，见 2.3 |

### 2.1 Provider 抽象层 — MUST（两版都缺，终版新增）

**模型名、供应商、endpoint 一律不出现在业务代码里。** 定义统一协议：

```python
class ChatProvider(Protocol):
    async def complete(self, req: ChatRequest) -> ChatResponse: ...
    async def stream(self, req: ChatRequest) -> AsyncIterator[Chunk]: ...
    @property
    def capabilities(self) -> ProviderCaps: ...   # 上下文长度、是否支持结构化输出、定价
```

- 任务到模型的映射放**配置**：`task_models: {ask: tier_strong, safety: tier_fast, ...}`。
- `ProviderCaps.context_window` 驱动 `ContextBudgeter`，**不允许硬编码 token 上限**。
- 换供应商 = 改配置 + 新增一个 adapter，业务代码零改动。这是应对模型快速迭代的唯一可维护方式。
- Embedding 与 Reranker 同样抽象（`EmbeddingProvider` / `RerankProvider`），
  因为它们的替换频率同样高，且切换必须触发重建索引（见 7.2）。

### 2.2 中文特化 — MUST（两版都缺，终版新增）

产品是中文交互，以下每一项不处理都会直接降低效果：

1. **BM25 分词**：中文无空格，必须接分词器（jieba / pkuseg）后再建索引。
   直接用英文 tokenizer 做中文 BM25 等于关键词检索失效。
2. **Embedding 选型**：云端用智谱 `embedding-3`，本地用 BGE-M3（均为中文强模型）。
   **不要默认用英文优化模型**；跨语言模型在中文口语表述上召回明显更差。
3. **Token 计数偏差**：中文 token/字符比与英文差异大（约 1 字 ≈ 0.6-1.5 token，随模型而变）。
   `ContextBudgeter` **必须用目标模型自己的 tokenizer 实测**，
   不可用"字符数 ÷ 4"这类英文经验公式，否则预算估算会系统性偏低并导致意外截断。
4. **Evidence 子串匹配**：中文需归一化处理（全角/半角、中英标点、空白、繁简），
   否则用户原话校验会因标点差异误判失败，见 6.3。
5. **危机词表**：必须覆盖中文谐音、拆字、拼音缩写、火星文变体，见 §3.4。

### 2.3 MCP 定位 — SHOULD

项目依赖里有 `mcp`。明确其用途，避免误用：

- **作为 MCP Server**：把「提问技术库检索」「维度覆盖检查」等**无状态能力**暴露给外部客户端。
- **作为 MCP Client**：接入外部工具（如查询本地心理援助资源目录）。
- **明确不做**：不通过 MCP 暴露用户记忆或会话状态。用户心理数据不得跨信任边界流出。

### 2.4 模型能力分级与门控 — MUST（v3.0 核心新增）

用户要求"尽可能多用本地模型且可切换"。**可切换是架构目标，等价可用不是事实。**
本节把这件事写成代码约束，避免下游工具把弱模型接到核心链路上。

#### 2.4.1 任务对模型的真实要求

| 任务 | 上下文需求 | 指令遵循 | 结构化输出 | 中文表达 | 最低档位 |
|---|---|---|---|---|---|
| Planner（规划 10 问） | 高（画像+检索+历史） | **极高**（多约束同时满足） | **必需**（嵌套 JSON） | 中 | `STRONG` |
| Interviewer（渲染问题） | 中 | **极高**（不许给建议） | 必需（简单 JSON） | **极高**（措辞就是产品本体） | `STRONG` |
| Analyst（分析报告） | **极高**（全量问答） | 极高 | **必需**（深层嵌套 + 原文引用） | 极高 | `STRONG` |
| Critic（复核依据） | 高 | 高 | 必需 | 中 | `STRONG` |
| Safety（风险判级） | 低（单条输入） | 中 | 必需（枚举 + 短文本） | 中 | `FAST` |
| AnswerQuality（质量评估） | 低 | 中 | 必需（布尔 + 分数） | 低 | `FAST` |
| Summarizer（摘要压缩） | 中 | 中 | 必需（含 `preserved_quotes` 原文回填） | 中 | `FAST` |
| Query 改写（HyDE） | 低 | 低 | 可选 | 中 | `FAST` |
| LLM 终审（重排） | 低 | 中 | 必需（0-3 打分） | 低 | `FAST` |

#### 2.4.2 档位定义

```python
class ModelTier(StrEnum):
    STRONG = "strong"   # 长上下文 + 强指令遵循 + 稳定 JSON：Planner/Interviewer/Analyst/Critic
    FAST   = "fast"     # 中等能力、低延迟低成本：Safety/Quality/Summary/改写/终审
    TOY    = "toy"      # 仅开发冒烟与 CI：能返回合法响应，内容质量不作保证

class ProviderCaps(BaseModel):
    tier: ModelTier
    context_window: int
    max_output_tokens: int
    supports_json_mode: bool
    supports_function_calling: bool
    supports_temperature: bool          # 推理型模型常忽略或拒绝该参数
    supports_streaming: bool
    zh_quality: Literal["low", "medium", "high"]
    tokenizer_hint: str                 # 用于选择计数器，见 §6.3
    cost_per_1k_in: float | None
    cost_per_1k_out: float | None
    probed_at: datetime | None           # 由 §17 探测清单实测填充
```

#### 2.4.3 门控规则（硬约束）— MUST

【决策 #17】

1. `config/task_models.yaml` 写死**任务 → 所需最低档位**；
   Gateway 路由时校验 `provider.caps.tier >= required_tier`。
2. **档位不足则拒绝路由**，而不是"先试试看"：
   抛 `InsufficientCapabilityError` → 走降级链 → 全链耗尽则返回 `LLM_UNAVAILABLE`。
3. **`TOY` 档永远不允许承担 `STRONG` 任务**，无论配置怎么写。
   这条在代码里是无条件断言，不读配置——配置写错不应让用户拿到劣质心理分析。
4. 任务要求 `supports_json_mode` 而 provider 不支持，同样视为能力不足。
5. 门控拒绝必须计入 `llm_capability_rejected_total{task,provider}` 并打日志。

> **为什么单独立一节：** §9.5 反复强调"静默降级最危险"，
> 而模型可切换恰恰是静默降级的最大入口 —— 换个 endpoint 就能跑通，
> 但输出从"有依据的洞察"退化成"通顺的空话"，接口依然返回 200。
> 能力门控是把这种退化变成显式失败的唯一手段。

### 2.5 minimind 的定位与用法 — MUST

用户希望用自己训练的 minimind。它在本项目里的正确位置是**测试替身**，不是生产模型。

**事实基础**：minimind 是从零训练的小参数（约 0.02B–0.1B 量级）中文语言模型教学项目，
上下文通常 512–1024 token，未经专门微调时**不具备稳定的复杂指令遵循与嵌套 JSON 生成能力**。

**因此**（决策 #16）：

| 用途 | 是否允许 | 说明 |
|---|---|---|
| CI / 本地冒烟：验证 adapter、流式、超时、重试链路真的通 | ✅ **推荐** | 比 FakeLLM 更真实（真走 HTTP、真有 tokenizer 行为），且免费离线 |
| 开发期跑通 Graph 全流程而不烧 token | ✅ 允许 | 只断言"流程走完、schema 未崩"，不断言内容质量 |
| 演示"可切换本地模型"的架构能力 | ✅ 允许 | 这是架构演示，不是质量演示 |
| Safety 风险判级 | ❌ **禁止** | 安全不能建立在能力不足的模型上；规则层 + `FAST` 档才是安全路径 |
| Planner / Interviewer / Analyst / Critic | ❌ **禁止**（`TOY` 硬阻断） | 见 §2.4.3 第 3 条 |

**接入方式**：minimind 若有 OpenAI 兼容的 serve 接口，直接复用 `OpenAICompatProvider`
（与 DeepSeek 同一 adapter，仅 `base_url` / `model` 不同），配置中标 `tier: toy`。
若只有原生推理脚本，写一个薄 HTTP 包装 `app/llm/providers/local_minimind.py`
把它变成 OpenAI 兼容形状 —— **不要为它在业务层开特例分支**。

**若确实想让本地模型承担生产任务**：正确路径不是提升 minimind，
而是用 Ollama / vLLM 托管一个**指令微调过的 7B 级中文模型**（如 Qwen2.5-7B-Instruct），
经 §17 探测确认 `supports_json_mode` 与中文质量后，标为 `FAST` 档承接安全/质量/摘要。
`STRONG` 档在消费级硬件上难以本地满足（需长上下文 + 强指令遵循），
本文档建议 `STRONG` 保持云端；除非合规要求数据不出境，
此时接受质量下降并**在 §7.5 评估集上量化这个下降幅度**，而不是假装没有下降。

### 2.6 向量库与嵌入维度（修订决策 #4）— MUST

**新信息推翻了 v2.0 的判断：**
`embedding-3` 实测 **2048 维**，而 pgvector 的 **HNSW 与 IVFFlat 索引上限是 2000 维**
（`vector` 列本身能存到 16000 维，但**无法为超过 2000 维的列建这两种索引**，
只能顺序扫描——那等于没有向量库）。v2.0 写的 `vector(1024)` 与"pgvector 起步"在 2048 维前提下不成立。

| 方案 | 做法 | 取舍 |
|---|---|---|
| **1. 降维到 1024（目标）** | 若 `embedding-3` 支持 `dimensions` 参数则直接请求 1024；否则做**确定性投影 + 重新归一化** | 与本地 BGE-M3 原生 1024 维**完全对齐**，云/本地切换无需改表；pgvector 可正常建 HNSW |
| **2. 保留 2048 + Chroma（P0-P3 实际路径）** | Chroma 无 2000 维索引限制，`CHROMA_COLLECTION` 已按 2048 命名配好 | 零基建、立即可用；与 pgvector 路线不通，后期需重建 |
| **3. pgvector `halfvec`** | `halfvec` 支持到 4000 维建 HNSW（pgvector ≥ 0.7） | 可行但精度减半，作兜底而非首选 |

**定稿（决策 #18 / #19）：**

- **P0-P3 用 Chroma**（已配置、零基建），一律经 `VectorStore` 抽象访问。
- **目标维度 1024**（`EMBED_TARGET_DIM=1024`）。
  若供应商不支持降维请求，由 `app/rag/projection.py` 做确定性降维，
  **且必须在 §7.5 检索评估集上验证 Recall@10 降幅 < 3%**，否则回退方案 2 并记录该决定。
- `chunks` 除 `embed_model` 外**再记 `embed_dim` 与 `projection_version`**：
  维度或投影方式变化与换模型等价，**同样要求全量重建**（见 §7.2）。
- Postgres 落地后再评估是否迁 pgvector。**迁移不是必须** ——
  本产品数据量级（技术库千级块 + 每用户百级块）Chroma 完全够用，
  迁移动机应当来自真实瓶颈而非架构洁癖。

**`VectorStore` 抽象 — MUST：**

```python
class VectorStore(Protocol):
    async def upsert(self, scope: str, owner_id: str | None, chunks: list[Chunk]) -> None: ...
    async def search(self, scope: str, owner_id: str | None, vec: list[float],
                     k: int, filters: dict | None = None) -> list[Hit]: ...
    async def delete_by_owner(self, owner_id: str) -> int: ...   # 级联删除，见 §3.6
    async def delete_by_ids(self, ids: list[str]) -> int: ...
    @property
    def dim(self) -> int: ...                                    # 启动时校验与配置一致
```

`delete_by_owner` 必须存在于抽象层 —— 它是"假删除"合规缺陷的唯一防线（§3.6 / §12）。
`dim` 属性用于启动自检：**维度与配置不符必须快速失败**，
否则写入会静默产生不可检索的数据。

### 2.7 供应商用途边界与密钥治理 — MUST

| 资源 | 用途 | 是否接触用户原话 |
|---|---|---|
| DeepSeek（`LLM_*`） | 全部 LLM 任务 | **是**（受 §3.3 约束） |
| 智谱 `embedding-3`（`EMBED_*`） | 向量化用户历史与知识库 | **是**（受 §3.3 约束） |
| 本地 minimind / Ollama | 开发冒烟 / 未来 `FAST` 档 | 是，但不出本机（隐私最优） |
| Bocha（`BOCHA_API_KEY`） | **仅**离线知识库构建（中文网） | **绝不**（决策 #22） |
| Tavily（`TAVILY_API_KEY`） | **仅**离线知识库构建（国际网） | **绝不**（决策 #22） |
| OpenAlex（`OPENALEX_*`） | **仅**离线学术出处采集 | **绝不**（决策 #23） |

**为什么搜索类 API 绝不能接用户原话**（决策 #22）：
把用户的心理独白发给搜索引擎是**不可逆的隐私泄露**（会进入其日志与缓存），
而且搜索结果对"提问质量"没有帮助 —— 本产品需要的是提问范式，不是网页事实。
工程上强制：`app/kb/` 下的搜索客户端**不从 `SessionState` 取任何字段**，
只接受 CLI 参数与配置文件输入；此约束需有测试用例覆盖。

**密钥治理 — MUST：**

- **用户在对话中提供的 key 已视为泄露，正式部署前必须全部轮换。**
  本文档只描述变量名，不落任何真实 key。
- `.env` 必须在 `.gitignore` 中（需确认覆盖 `.env`、`.env.*`、`*.key`）；
  `.env.example` 只写变量名与说明，**不写真实值**。
- OpenAlex 配额：**10000 credits/日，每请求固定扣 10 credits ≈ 1000 请求/日**。
  知识库构建脚本必须**带本地缓存与断点续跑**，避免重跑耗尽当日配额（见 §7.7）。
- 所有外部调用走 `httpx.AsyncClient` 且**必须设超时**；
  搜索类调用失败不得阻断任何主流程（它们本就只在离线管线中）。

---

## 3. 安全、隐私与合规（最高优先级，先于功能实现）

这是触及心理状态的产品。**安全模块必须在第一个可运行版本里就存在**，不能作为后期补丁。
以下规则在架构上强制，而非仅写在提示词里——提示词可被绕过，代码分支不能。

### 3.1 产品定位声明 — MUST

- 系统是**自我探索与思维梳理工具**，不是医疗器械，不做诊断、不做治疗、不开处方。
- 输出中禁止"诊断""你患有""疗程""处方""确诊"等表述，由 `OutputGuard` 正则 + 语义双重拦截。
- 【决策 #10】免责声明出现在：**会话开始**、**报告显著位置**、**危机分支**。
  不逐条附加——逐条附加导致声明疲劳，用户会完全忽略，反而降低实际保护效果。

### 3.2 知情同意（两版都不完整，终版强化）— MUST

DeepSeek 提到 `consent_given BOOLEAN`，但布尔值不够：**条款会变，需要版本化。**

```python
class Consent(BaseModel):
    user_id: str
    policy_version: str        # 同意的是哪一版条款
    granted_at: datetime
    scopes: set[ConsentScope]  # {store_memory, cross_session_recall, quality_review, eval_use}
    withdrawn_at: datetime | None
```

- **分项授权**：存储记忆、跨会话调用、人工质量抽检、用于评估集，**各自独立可拒绝**。
  拒绝跨会话调用的用户仍能正常使用单次会话——功能必须为此降级而非拒绝服务。
- 条款版本升级需**重新征求同意**，旧版同意不自动继承。
- **撤回同意**必须立即生效并触发数据删除流程（见 3.6）。
- **未成年人**：若用户声明或系统合理推断为未成年人，
  强制 `Sharpness=GENTLE`、关闭深度恐惧维度探索、提示监护人与专业资源。
  这是不可配置的硬约束。

### 3.3 供应商数据留存（两版都缺，最关键的遗漏）— MUST

**用户的心理独白会离开你的服务器进入模型供应商。** 这是本产品最大的隐私风险点，
两份文档都只谈了自己的存储加密，却没管数据出境后的处理。

- 选择供应商时**必须确认**：默认不用于训练、零留存或短留存（zero/short retention）选项、
  数据处理地域。这些条款直接决定产品能否合法运营。
- 与供应商签署 DPA（数据处理协议）；隐私政策中**明示**使用了第三方模型服务及其地域。
- 若合规要求数据不出境 → 走**自托管开源模型**路径（架构上由 2.1 Provider 抽象层支持，
  切换只需新增 adapter）。**这是 Provider 抽象层的核心价值之一，不是可选优化。**
- 送往模型前对可识别 PII 做脱敏（见 §3.5），减少出境敏感信息。

**本项目的具体情况（v3.0 补充）：**

当前用户原话会流向**两家**供应商，而非一家 —— 这点容易被忽略：

| 供应商 | 收到什么 | 上线前必须确认 |
|---|---|---|
| DeepSeek | 困惑描述、全部回答原文、摘要 | 是否默认用于训练、留存期、能否申请零留存 |
| 智谱（embedding-3） | **被向量化的每一段用户表述原文** | 同上。嵌入调用同样是数据出境，常被误认为"只是算向量" |

- **嵌入服务也是数据出境通道。** 只关注 LLM 而忽略 embedding 是常见疏漏：
  为了做跨会话检索，用户的历史表述会被逐条送去向量化，累积暴露量甚至超过单次对话。
- **本地嵌入是隐私上明显更优的选择**：BGE-M3 本地推理让"用户历史"这一最敏感的数据
  完全不出本机。若硬件允许，建议**生产环境用本地 BGE-M3 做 `user` scope 向量化，
  云端 embedding-3 只用于 `kb` scope（公共知识库，无隐私问题）**。
  这是一个**按 scope 分流 provider** 的设计，`EmbeddingProvider` 抽象需支持按 scope 选择实现。
- 两家的条款都需在隐私政策中明示，并纳入 §3.2 的 `policy_version`。

### 3.3.1 按 scope 分流嵌入 provider — SHOULD（强烈建议）

```yaml
# config/task_models.yaml
embeddings:
  kb:    cloud_zhipu      # 公共知识库：用云端，省本地算力
  user:  local_bge_m3     # 用户历史：留在本机，隐私最优
```

**约束**：两者**必须输出同一维度**（§2.6 的 1024 目标维度正是为此），
否则 `kb` 与 `user` 的向量空间不可比。
即使两个 scope 各自独立检索、从不跨库比较，同维仍是必要的 ——
它保证切换任一侧时不必改表结构，也让降级（本地不可用时临时回落云端）成为可能。
**降级回落云端必须记 `degraded` 并计入审计**（因为它改变了数据流向，是隐私相关事件，
而非单纯的质量事件）。

### 3.4 危机识别与转介 — MUST

**双层检测，串行短路：**

1. **第一层：规则前置过滤器**（`app/safety/lexicon.py`）
   - 自伤/自杀意念、伤害他人、严重物质滥用、未成年人受虐等关键词与变体。
   - **中文变体必须覆盖**：谐音（如"自 sha"）、拆字、拼音缩写、空格/符号插入、繁简、火星文。
   - 零 LLM 依赖，延迟 < 1ms，**LLM 不可用时仍然生效**。这是最后一道防线，必须能独立工作。
2. **第二层：轻量分类器**（小模型，低温度，结构化输出）
   - 输出 `risk_level ∈ {none, low, medium, high, imminent}` + 依据片段。
   - 仅在第一层未命中但语义可疑时调用，控制成本。
   - 【吸收 DeepSeek】可叠加供应商 Moderation API 作为第三信号，但**不可替代词表层**
     （Moderation 面向内容合规，不面向自伤风险，二者目标不同）。

**处置矩阵（硬编码为状态机分支，不由 LLM 自由决定）：**

| risk_level | 系统行为 |
|---|---|
| none / low | 正常提问流程 |
| medium | 中断提问节奏，切换关怀性回应，锐度自动降级，提示专业资源 |
| high / imminent | **立即终止提问流程**，输出危机资源信息，不再生成任何深挖性问题 |

- 危机分支**不允许**继续"犀利提问"。产品调性在此让位于安全。
- 危机资源信息**不由模型生成**，只从 `config/crisis_resources.yaml` 按地区读取。
  模型编造错误的求助电话，后果不可接受。
- 【决策 #7】默认只提供外部资源。**"转人工"仅在真有值班人力时用 feature flag 开启**：
  承诺人工介入而无人响应，比不承诺更危险。
- 危机事件落**独立审计日志**（append-only、单独权限、脱敏），用于人工复核与阈值调优。
- **假阴性远比假阳性可怕**：词表阈值设计上偏保守（宁可多触发 medium）。
  假阳性的代价是一次略显谨慎的回应，假阴性的代价不可逆。

【生成指令】`app/safety/` 下实现 `lexicon.py`、`classifier.py`、`policy.py`。
`policy.decide(risk, session) -> SafetyAction` 必须是纯函数、可完全离线单测。
危机路径单测覆盖率 100%，且红队用例集 `tests/redteam/crisis_cases.jsonl` **只增不减**。

### 3.5 提问锐度控制（Sharpness Governor）— MUST

用户要求"犀利问题"，但犀利与伤害只隔一层。

```python
class Sharpness(IntEnum):
    GENTLE = 1   # 温和探索
    MODERATE = 2 # 适度挑战（默认）
    SHARP = 3    # 犀利直击（需用户显式开启）
```

**自动降级触发条件**（任一命中即降级并记录原因）：
`risk_level >= medium` · 用户回答长度骤降（回避信号）· 情绪强度指标上升 ·
用户明确表达不适 · 判定为未成年人（强制 GENTLE 且不可上调）。

锐度作为参数注入提示词模板，同时作为 `OutputGuard` 的校验基线。

### 3.6 隐私与数据治理 — MUST

【吸收 DeepSeek 的加密与合规，并补两版都缺的留存策略】

- **传输加密** TLS 1.2+；**静态加密** AES-256（磁盘/卷级）；
  会话原文字段建议应用层加密，密钥走 KMS/环境隔离，**不与数据同库**。
- **PII 最小化**：不主动索取真实姓名、身份证、住址、单位全称。
  用户主动提供则存储前脱敏（`app/safety/pii.py`）；日志中一律以键名代替值。
- **数据留存策略**（两版都缺）：
  | 数据 | 默认留存 | 说明 |
  |---|---|---|
  | 会话原文 | 90 天（可配置） | 用户可提前删除 |
  | 结构化记忆 | 用户主动删除前保留 | 长期价值所在 |
  | 报告 | 同记忆 | 用户可导出 |
  | 普通日志 | 30 天 | 不含原话 |
  | 危机审计日志 | 按当地法规（通常更长） | 单独权限 |
- **用户权利**：查看 / 导出（JSON）/ 更正 / 删除（含向量库中的对应块）。
  **删除必须级联到向量库和缓存**，否则"已删除"是假的——
  这是最常见的合规实现缺陷，务必写进删除流程的测试用例。
- 【决策 #8】**声明遵循 GDPR 与中国个人信息保护法；明确不声称 HIPAA**。
  本产品不是 HIPAA covered entity，错误声称合规本身构成法律风险。

### 3.7 越权、注入与滥用防护 — MUST

- **禁止代入真实身份**：不扮演具名真实心理医生、不声称持有执业资质。
- **提示词注入防护**：
  用户输入永远作为数据传入，以明确分隔符包裹；系统指令置于用户内容之前，
  并在末尾复述关键约束（sandwich 结构）；检测并忽略"忽略以上指令""你现在是另一个 Agent"类内容。
  **注入检测命中不应直接报错**，而是剥离可疑指令后继续（用户可能只是好奇试探，报错体验差）。
- **输出侧兜底**（`OutputGuard`）拦截：医疗断言、绝对化承诺（"一定会好"）、
  越界建议（用药、停药、法律、投资）、**给建议而非提问**（提问阶段的核心违规）。
  命中则重生成；重生成仍失败降级到安全模板回复。
- **成本滥用**：单会话 token 上限 + 单用户日配额，超限熔断（见 9.2）。

### 3.8 专家参与与定期审计 — SHOULD

【吸收 DeepSeek，这是我方前版的实质缺失】

技术手段无法验证「提问是否真的有心理学品质」，必须有人类专业判断入环：

- **上线前**：由具备心理咨询背景的人员审阅提问技术库、提示词、评估集标注标准。
- **运行中**：定期（如每月）抽样审阅会话（**仅限授权 `quality_review` 的用户**），
  重点看：是否有引导性预设、是否越界给建议、危机识别是否有漏检。
- **审计结论回流**：新增红队用例、调整词表阈值、修订提示词，形成闭环。
- 抽检必须走**独立权限**与脱敏视图，且抽检记录本身可审计。

---

## 4. 系统架构

### 4.1 分层视图（模块化单体，按未来可拆分设计）

```
┌────────────────────────────────────────────────────────────────┐
│ 接入层  FastAPI · REST + SSE · 鉴权 · 限流 · 幂等 · Trace 注入  │
├────────────────────────────────────────────────────────────────┤
│ 编排层  LangGraph StateGraph + Checkpointer（中断/续跑/回溯）    │
│  Intake→Consent→Plan→Ask→Listen→Probe→Analyze→Critic→Report    │
│                          ↘ Crisis（任意节点可短路）              │
├────────────────────────────────────────────────────────────────┤
│ Agent 层  Planner · Interviewer · Safety · Analyst · Critic     │
├────────────────────────────────────────────────────────────────┤
│ 能力层  LLMGateway · RAG(检索/融合/重排) · Memory · Context · MCP│
├────────────────────────────────────────────────────────────────┤
│ 存储层  Postgres(+pgvector) · Redis(锁/缓存/队列) · 对象存储(导出)│
├────────────────────────────────────────────────────────────────┤
│ 可观测  结构化日志 · OTel Trace · Prometheus 指标 · Sentry · 评估│
└────────────────────────────────────────────────────────────────┘
```

【决策 #1】**模块化单体，不是微服务。** 但每个能力层模块以**接口**暴露，
禁止跨模块直接访问对方的数据表——这样 P5 需要拆分时，
把模块换成 HTTP/gRPC 客户端即可，而不必重写业务逻辑。

**未来拆分优先序**（真到了需要拆的时候，按此顺序）：
1. Embedding / Rerank 推理服务（GPU 资源特性不同，最该独立）
2. 报告生成 worker（长任务，扩缩容节奏不同）
3. RAG 检索服务（可被多产品复用）

对话编排**最后拆**，因为它持有状态，拆开代价最高、收益最低。

### 4.2 为什么用 LangGraph 而不是事件驱动微服务

【决策 #2】提问流程本质是**有状态、可中断、需回溯**的过程：
用户可能中途离开三天、答得敷衍需要追问、触发安全分支需立即跳转。

- **可恢复**：Checkpointer（开发 SQLite → 生产 Postgres），用户三天后回来接着第 6 问继续。
- **可审计**：每次状态迁移落库，出问题能重放。
- **可插拔安全边**：安全检查作为条件边而非节点内 if，任何节点后都能短路到危机分支。
- 事件驱动 + Saga 要自己实现上述全部能力，且**分布式调试成本极高**——
  当用户抱怨"问题很奇怪"时，你需要在一个 trace 里看完整推理链，而不是跨 5 个服务拼日志。

### 4.3 状态机定义 — MUST

```python
class Phase(StrEnum):
    INTAKE   = "intake"    # 收集困惑描述
    CONSENT  = "consent"   # 知情同意 + 模式与锐度选择（新增，见 3.2）
    PLAN     = "plan"      # 生成维度规划（不是问题文本）
    ASK      = "ask"       # 生成并发出问题（按模式一次/一批/一个）
    LISTEN   = "listen"    # 接收回答 + 质量评估
    PROBE    = "probe"     # 敷衍/回避时追问（不占用 10 问配额）
    ANALYZE  = "analyze"   # 潜意识诉求分析
    CRITIC   = "critic"    # 分析结果自检（依据与幻觉校验）
    REPORT   = "report"    # 输出结构化报告
    CRISIS   = "crisis"    # 安全终止分支
    PAUSED   = "paused"    # 会话中断待恢复（新增，见 4.5）
    DONE     = "done"
```

**关键条件边：**

- `LISTEN → PROBE`：质量分低于阈值且追问未超上限（每问最多 2 次）。
- `LISTEN → ASK`：质量合格，推进下一问。
- `LISTEN → PLAN`：`BATCHED` 模式下每批结束，重规划剩余槽位。
- `任意节点 → CRISIS`：安全检查命中 `high/imminent`。**此边优先级最高，覆盖其他所有边。**
- `ANALYZE → CRITIC → ANALYZE`：自检不通过最多重做 2 次；
  超限则输出**带不确定性标注**的报告——宁可承认依据不足，也不编造洞察。

### 4.4 Agent 职责与协作契约 — MUST

多 Agent 的失败模式几乎都来自**职责重叠**与**隐式契约**。因此：
每个 Agent 只有一个职责，Agent 间**只通过 Pydantic 模型通信，禁止传自由文本**。

| Agent | 唯一职责 | 输入 | 输出 | 禁止 |
|---|---|---|---|---|
| Planner | 规划维度与技术序列 | 困惑 + 画像 + 检索片段 | `QuestionPlan` | 生成问题文本 |
| Interviewer | 把槽位渲染成自然问题 | `PlanSlot` + 历史摘要 | `Question` | 给建议、下判断 |
| Safety | 风险判级 | 用户最新输入 | `RiskAssessment` | 修改流程（判级归它，处置归 policy） |
| Analyst | 潜意识诉求分析 + 建议 | 全量问答对 | `InsightReport` | 引用未出现的内容 |
| Critic | 校验 Analyst 输出 | `InsightReport` + 问答对 | `CritiqueResult` | 自己改写报告 |

**三条协作铁律：**

1. **单写者**：任一状态字段只有一个 Agent 可写，从根上消除并发写冲突。
2. **Critic 只否决不改写**：改写权保留给 Analyst。若两个 Agent 都能改，会出现互相覆盖的振荡。
3. **无自由文本传递**：全部结构化，避免"上一个 Agent 的措辞影响下一个 Agent 的判断"这类隐性耦合。
   Critic 尤其**不接收 Analyst 的推理过程**，只看结论与原始问答，防 anchoring。

### 4.5 会话恢复的状态重确认（两版都缺）— MUST

LangGraph 能恢复**技术状态**，但恢复不了**人的状态**。用户三天后回来，心境可能已变。

- `PAUSED` 超过阈值（默认 24h）后恢复时，必须先做**轻量重确认**：
  简要回顾已聊内容 + 询问"这几天有变化吗？"，而非直接抛出第 6 问。
- 重确认的回答**可触发重规划**（剩余槽位），也可能触发新的安全检查。
- 超过更长阈值（默认 14 天）建议**开启新会话**，旧会话结论转入长期记忆作为背景。
  强行续接两周前的情绪脉络，往往比重新开始更差。

【生成指令】编排放 `app/graph/`（一节点一文件）；Agent 放 `app/agents/`；
跨节点模型放 `app/schemas/`。节点签名统一 `async def node(state: SessionState) -> dict`，
只返回增量更新字段（便于 LangGraph 做状态合并与最小化写入）。

---

## 5. 数据模型

模型即契约。下游工具应**先生成本节全部模型并通过类型检查**，再实现逻辑。

### 5.1 领域模型（Pydantic v2）— MUST

```python
class PlanSlot(BaseModel):
    index: int = Field(ge=1, le=10)
    dimension: Dimension
    technique: Technique
    intent: str                      # 这一问要挖什么（内部说明，不展示给用户）
    depends_on: list[int] = []       # ONE_SHOT 模式下必须为空

class QuestionPlan(BaseModel):
    slots: list[PlanSlot]
    mode: AskMode
    rationale: str
    @model_validator(mode="after")
    def check_coverage(self):
        # 维度覆盖 + 无连续重复 + ONE_SHOT 时 depends_on 为空；失败即重规划
        ...

class Question(BaseModel):
    index: int
    text: str = Field(max_length=300)     # 硬约束：防认知过载
    technique: Technique
    sharpness: Sharpness

class AnswerQuality(BaseModel):
    score: float = Field(ge=0, le=1)
    is_evasive: bool
    is_too_brief: bool
    missing_aspects: list[str]             # 供 PROBE 生成追问

class Answer(BaseModel):
    question_index: int
    text: str
    quality: AnswerQuality | None = None

class RiskAssessment(BaseModel):
    level: RiskLevel
    matched_rules: list[str] = []
    evidence: str | None = None
    source: Literal["lexicon", "classifier", "moderation"]

class Evidence(BaseModel):
    """报告中每条洞察必须挂依据 —— 反幻觉的结构性约束。"""
    question_index: int
    quote: str                             # 必须是用户原话子串（归一化后校验）

class Insight(BaseModel):
    statement: str
    evidence: list[Evidence] = Field(min_length=1)   # 无依据不许成条
    confidence: Literal["low", "medium", "high"]

class Suggestion(BaseModel):
    action: str
    first_step: str                        # 必须 24h 内可启动，防空泛建议
    linked_insights: list[int]

class InsightReport(BaseModel):
    surface_concern: str                   # 用户自述的表层困惑
    underlying_needs: list[Insight]        # 潜意识诉求
    inner_conflicts: list[Insight]         # 内在冲突（记忆冲突是重要来源，见 8.2）
    suggestions: list[Suggestion]
    uncertainty_notes: list[str] = []      # 明确写出依据不足之处
    degraded: bool = False                 # 生成过程是否发生降级（见 9.5）

class SessionState(BaseModel):
    session_id: str
    user_id: str
    phase: Phase
    mode: AskMode
    sharpness: Sharpness
    concern: str
    plan: QuestionPlan | None
    qa_pairs: list[tuple[Question, Answer]]
    probe_counts: dict[int, int]
    risk_history: list[RiskAssessment]
    report: InsightReport | None
    consent: Consent | None
    paused_at: datetime | None             # 见 4.5 恢复重确认
    revision: int                          # 乐观锁版本号
```

**四个关键设计点：**

- `Evidence.quote` 必须是用户原话子串 —— 把"不许编造"从提示词约束升级为**可程序化验证的约束**。
  **这是全系统最重要的一条反幻觉设计。**
- `Suggestion.first_step` 强制 24h 可启动，逼出具体动作而非"多沟通""调整心态"。
- `uncertainty_notes` 是**允许模型说不知道的出口**。没有这个字段，模型只会编。
- `revision` 用于乐观锁并发控制（见 9.1）。

### 5.2 物理数据模型（PostgreSQL）— MUST

【吸收 DeepSeek 的 DDL 级设计，并修正三处会导致信息丢失的字段设计】

```sql
-- 用户
CREATE TABLE users (
  user_id        UUID PRIMARY KEY,
  created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
  last_seen      TIMESTAMPTZ,
  is_minor_flag  BOOLEAN DEFAULT FALSE,      -- 见 3.2 未成年人硬约束
  anonymized     BOOLEAN DEFAULT FALSE,
  deleted_at     TIMESTAMPTZ                 -- 软删除，级联清理见 3.6
);

-- 知情同意（修正 #1：DeepSeek 用单个 BOOLEAN，无法表达版本与分项授权）
CREATE TABLE consents (
  consent_id     UUID PRIMARY KEY,
  user_id        UUID NOT NULL REFERENCES users,
  policy_version TEXT NOT NULL,
  scopes         JSONB NOT NULL,             -- {"store_memory":true,"eval_use":false,...}
  granted_at     TIMESTAMPTZ NOT NULL,
  withdrawn_at   TIMESTAMPTZ
);

-- 会话
CREATE TABLE sessions (
  session_id     UUID PRIMARY KEY,
  user_id        UUID NOT NULL REFERENCES users,
  status         TEXT NOT NULL,              -- active/paused/completed/aborted/crisis_stopped
  mode           TEXT NOT NULL,
  sharpness      SMALLINT NOT NULL,
  phase          TEXT NOT NULL,
  concern        TEXT,
  summary        TEXT,
  revision       INT NOT NULL DEFAULT 0,     -- 乐观锁
  start_time     TIMESTAMPTZ NOT NULL DEFAULT now(),
  paused_at      TIMESTAMPTZ,
  end_time       TIMESTAMPTZ
);
CREATE INDEX ON sessions (user_id, start_time DESC);

-- 消息
CREATE TABLE messages (
  message_id     UUID PRIMARY KEY,
  session_id     UUID NOT NULL REFERENCES sessions,
  seq            INT NOT NULL,               -- 会话内单调序号，保证顺序可靠
  role           TEXT NOT NULL,              -- user/assistant/system
  content        TEXT NOT NULL,              -- 应用层加密候选字段
  metadata       JSONB,                      -- 提问技术、维度、质量分、prompt 版本
  created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (session_id, seq)
);

-- 长期记忆（修正 #2：DeepSeek 无冲突关系字段，无法表达"想法变了"）
CREATE TABLE memories (
  memory_id      UUID PRIMARY KEY,
  user_id        UUID NOT NULL REFERENCES users,
  category       TEXT NOT NULL,              -- 价值观/恐惧/偏好/已尝试方案/关系
  topic          TEXT NOT NULL,              -- 冲突检测按 topic 聚类
  content        TEXT NOT NULL,
  confidence     REAL NOT NULL,
  status         TEXT NOT NULL DEFAULT 'active',  -- active/superseded/invalidated
  supersedes     UUID REFERENCES memories,   -- 时间演化链
  conflict_group UUID,                       -- 矛盾并存型共享同一 group
  asserted_at    TIMESTAMPTZ NOT NULL,       -- 用户说这话的时间（≠写库时间）
  source_session UUID REFERENCES sessions,
  created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at     TIMESTAMPTZ
);
CREATE INDEX ON memories (user_id, topic, status);

-- 报告
CREATE TABLE reports (
  report_id      UUID PRIMARY KEY,
  session_id     UUID NOT NULL REFERENCES sessions,
  payload        JSONB NOT NULL,             -- InsightReport 序列化
  degraded       BOOLEAN NOT NULL DEFAULT FALSE,
  created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 用户反馈（吸收 DeepSeek 的反馈闭环，见 12.4）
CREATE TABLE feedback (
  feedback_id    UUID PRIMARY KEY,
  session_id     UUID REFERENCES sessions,
  target_type    TEXT NOT NULL,              -- question/report/suggestion
  target_ref     TEXT,
  rating         SMALLINT,                   -- 1-5
  comment        TEXT,
  created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 安全审计（修正 #3：两版都未把危机事件独立存储）
CREATE TABLE safety_audit (
  audit_id       UUID PRIMARY KEY,
  session_id     UUID,
  user_id_hash   TEXT NOT NULL,              -- 哈希而非明文 user_id
  risk_level     TEXT NOT NULL,
  source         TEXT NOT NULL,
  matched_rules  JSONB,
  action_taken   TEXT NOT NULL,
  created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
-- 仅追加：撤销 UPDATE/DELETE 权限，单独角色可读

-- 向量块元数据（P0-P3：向量存 Chroma，此表存元数据与来源，见 §2.6【决策 #18】）
CREATE TABLE chunks (
  chunk_id          UUID PRIMARY KEY,
  scope             TEXT NOT NULL,           -- 'kb' | 'user'
  owner_id          UUID,                    -- scope='user' 时必填，强制过滤
  text              TEXT NOT NULL,
  embed_model       TEXT NOT NULL,           -- 见 §7.2：三字段任一变化即需全量重建
  embed_dim         INT  NOT NULL,
  projection_version TEXT,                   -- 降维方案版本（§2.6 方案 1）
  vector_ref        TEXT NOT NULL,           -- Chroma collection + id
  metadata          JSONB NOT NULL,
  created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX ON chunks (scope, owner_id);
CREATE INDEX ON chunks (embed_model, embed_dim);

-- 迁 pgvector 时（Postgres 落地后再评估，非必须）追加：
--   ALTER TABLE chunks ADD COLUMN embedding vector(1024);
--   CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops);
--   CREATE INDEX ON chunks USING gin (tsv);
-- 注意：HNSW/IVFFlat 索引上限 2000 维。
--   1024 维可直接建索引；2048 维必须改用 halfvec（≤4000 维）或先降维，见 §2.6。
```

**三处修正的原因：**

1. `consents` 独立成表：布尔值无法表达"同意了哪一版、授权了哪些范围"，条款变更时无法合规。
2. `memories` 增加 `supersedes` / `conflict_group` / `asserted_at`：
   没有这些字段就只能覆盖旧记忆，而**用户想法变化本身是最有价值的提问素材**（见 8.2）。
   `asserted_at` 与 `created_at` 必须分开——用户说话时间和入库时间不同，时间衰减要用前者。
3. `safety_audit` 独立且仅追加：危机记录与业务数据混存，权限无法收紧，也不满足审计要求。

---

## 6. 提示词工程、上下文预算与幻觉治理

### 6.1 提示词版本化与外置 — MUST

- 提示词**不写在 Python 字符串里**，放 `prompts/{agent}/{version}.jinja2`，
  由 `PromptRegistry` 按配置加载。
- 文件头注明：适用 Agent、期望输出 schema、变更原因。
- 切换版本 = 改配置，**无需改代码、无需重新部署**，便于 A/B 与快速回滚。
- 渲染后记录 hash 到 trace，问题复现时能定位到确切版本。
- 【吸收 DeepSeek】模板分区管理：`固定指令` / `动态用户信息` / `检索结果` 三段分离，
  便于按段裁剪（见 6.3）与命中前缀缓存（见 9.4）。

### 6.2 结构化输出（对抗格式漂移）— MUST

【决策 #20】v2.0 写的"一律用 function calling"在多 provider 现实下站不住：
DeepSeek 系与本地模型对 function calling 的支持程度不一致，
推理型模型常不支持工具调用（也常忽略 `temperature`），
而 minimind 级模型两者都不支持。**把 function calling 当唯一路径，换模型时整链失效。**

**因此改为分层策略（按 provider 能力自动选择，见 §2.4.2）：**

| 优先级 | 手段 | 启用条件 |
|---|---|---|
| 1 | `response_format={"type":"json_object"}`（JSON 模式） | `caps.supports_json_mode` |
| 2 | function calling / tool schema | `caps.supports_function_calling`（探测确认后） |
| 3 | 纯提示词约束 + 严格解析 | 兜底（仅 `TOY` 档开发场景） |

**无论走哪条路，Pydantic 校验都是唯一裁判** ——
provider 声称支持 JSON 模式不等于输出一定符合我们的 schema。

**解析失败三级降级（保留 v2.0 设计）：**
1. 直接解析 → 2. 容错修复（去 markdown 围栏、修尾逗号、补全被截断的括号）
→ 3. 带错误信息重试一次（若 `supports_temperature` 则降到 0）
→ 仍失败抛 `StructuredOutputError`。

- **三次失败必须显式失败**，不允许猜测填充默认值继续跑。
  静默降级会产出"看起来正常但内容错误"的报告，比报错危险得多。
- **`supports_temperature=False` 时不得传该参数**：部分推理型模型会直接报 400。
  Gateway 必须按 caps 裁剪请求参数，而不是无条件全带上。
- **JSON 模式下仍需在提示词里给出 schema 与示例**：
  JSON 模式只保证"是合法 JSON"，不保证"是你要的那个 JSON"。
- 输出被截断是结构化失败的最常见原因，见 §6.5 的输出预算约束。

### 6.3 上下文预算与截断策略 — MUST

长会话（10 问 + 追问，可能 30+ 轮）的核心工程问题。**绝不能靠"祈祷不超长"。**

```
总预算 = min(ProviderCaps.context_window, 配置上限)
分配：系统指令 15% | 会话摘要 20% | 近期原文 35% | 检索片段 20% | 输出预留 10%
```

**分层记忆压缩：**

1. **近期窗口**：最近 3 轮问答保留**完整原文**（追问依赖细节，压缩即失效）。
2. **中期摘要**：更早问答按「维度 + 关键事实 + 情绪信号 + 用户原话金句」结构化摘要。
   **金句必须原样保留** —— 后续 `Evidence` 校验依赖原文。
3. **长期画像**：跨会话稳定特征（结构化字段，非自由文本）。

**实施要点：**

- 每次 LLM 调用前 `ContextBudgeter.fit()` 精确计数并按优先级裁剪。
  **超预算时先砍检索片段 → 再砍中期摘要 → 永不砍系统指令与安全约束。**
- token 计数**必须用目标模型自己的 tokenizer**（中文偏差大，见 §2.2）。
  **现实约束（v3.0）**：`tiktoken` 只覆盖 OpenAI 系模型，**对 DeepSeek 与本地模型并不精确**。
  策略：优先用 provider 官方 tokenizer；不可得时用 `tiktoken` 的 `cl100k_base` 做**保守估算**
  并乘安全系数（中文建议 1.15-1.25），宁可低估可用预算也不要超限。
  `ProviderCaps.tokenizer_hint` 用于选择计数实现；
  **估算模式必须在日志与 trace 中标记**，否则截断问题无法归因。
- 摘要器输出必须含 `preserved_quotes` 字段，且压缩后金句仍可被 Evidence 校验命中。
- 结构化摘要支持**选择性加载**：Analyst 需全维度，Interviewer 只需相邻维度。
- 截断事件必须记指标（`context_truncation_total`），持续触发说明预算配置有问题。

【生成指令】实现 `app/context/budgeter.py` 与 `summarizer.py`。
`budgeter.fit()` 必须是纯函数、可单测（输入预算与片段列表 → 输出裁剪后列表），
**裁剪逻辑内不得调用 LLM**。

### 6.4 幻觉治理（四层叠加）— MUST

【决策 #9】两版方案叠加，程序化校验为不可绕过底线。单一手段防不住幻觉。

| 层级 | 手段 | 拦截对象 |
|---|---|---|
| 生成前 | 检索增强 + "仅基于用户所说"指令 + 低温度（分析 ≤0.3） | 无依据推断 |
| 生成时 | 结构化输出强制每条 Insight 挂 Evidence | 空口洞察 |
| **生成后** | **`verify_evidence()` 程序化校验 quote 是否为用户原话子串** | **伪造引用** |
| 复核 | Critic 独立复核依据与结论的逻辑关联 | 引用真实但推论跳跃 |

- **第三层是最硬的**：不依赖任何模型判断，纯字符串校验。
  归一化后模糊匹配（阈值 0.9），中文需处理全角/半角、中英标点、繁简、空白。
  **校验不通过的 Insight 直接丢弃**，不保留待人工判断。
- Critic 用**独立提示词 + 更低温度**，只接收「报告 + 问答对」，
  不接收 Analyst 的推理过程（防 anchoring）。
- 【决策 #6】分析阶段**禁止引用心理学文献给用户下结论**。
  可以说"有一种参考框架认为…"，不可以说"根据《XX》你的情况属于…"。
  两个原因：会把探索工具包装成诊断权威；模型极易编造书名与页码。
  如需引用，只能引用**知识库中真实存在的条目 id**，并同样走 Evidence 式校验。

### 6.5 输出长度预算与报告分段生成 — MUST（v3.0 新增）

【决策 #21】实际配置 `LLM_MAX_TOKENS=4096`。这个值对本产品的影响是结构性的：

- 若目标模型是**推理型**，思维链会占用输出额度，留给 JSON 正文的可能远少于 4096。
- `InsightReport` 是深层嵌套结构（多条 `Insight` × 多条 `Evidence` 原文引用 + 建议 + 不确定性说明），
  中文原话引用尤其占 token。**一次生成整份报告极易触顶。**
- 输出截断发生在 JSON 中间 → 直接解析失败 → 触发重试 → **再次截断**。
  这会变成稳定失败而非偶发失败，重试链救不回来。

**因此报告必须分段生成：**

```
Analyze 阶段拆为三次调用（各自独立 schema、独立 Evidence 校验）：
  1. underlying_needs   (潜意识诉求)
  2. inner_conflicts    (内在冲突)
  3. suggestions        (建议，输入含前两步已确认的 insights)
最后由代码组装成 InsightReport，uncertainty_notes 由各段合并
```

**要点：**

- 分段让每次输出远离上限，也让**部分失败可局部重试**（诉求成功、冲突失败只重做冲突段）。
- 建议段以**前两段已通过 Evidence 校验的结论**为输入，保证 `linked_insights` 指向真实存在的洞察。
- 分段后必须校验 `Suggestion.linked_insights` 的索引有效性（代码校验，非模型自觉）。
- 若某段最终失败：**报告仍然出，但该段留空并在 `uncertainty_notes` 明确写出缺失原因**，
  同时置 `degraded=true`。
  这比整份报告失败更好，也比悄悄少一段危险得多（§9.5）。
- 同理，**Planner 一次规划 10 个 slot 也接近输出上限**：
  `ONE_SHOT` 模式下若 10 问一次生成失败，允许拆成两批各 5 问（维度约束仍全局校验）。

**通用约束 — MUST**：
`ContextBudgeter` 必须同时约束**输入预算**与**输出预留**：
`max_output_tokens = min(caps.max_output_tokens, 配置上限)`，
且输入预算裁剪后要为输出留足空间（§6.3 的 10% 输出预留在 4096 上限下需按绝对值重算，
而非按比例——比例分配在小上限时会失效）。

---

## 7. RAG 设计

### 7.1 检索什么 —— 先明确边界

本系统 RAG **不用于回答用户问题**，只用于两个目的：

1. **提问技术库检索**：为 Planner 提供适用于当前情境的提问范式与案例。
2. **用户历史检索**：跨会话召回该用户过去的相关表述（需 `cross_session_recall` 授权）。

**明确不做**：检索心理学文献去给用户下结论（见 6.4 决策 #6）。
知识库内容以**提问范式**为主，而非诊断标准。这既是安全边界，也避免"看似权威实则套用"。

### 7.2 索引设计与 Embedding 版本管理 — MUST

- **两个 scope 严格隔离**：`kb`（公共提问技术库）与 `user`（私有历史）。
  Chroma/Qdrant 按 collection 隔离；pgvector 按 `scope + owner_id` 过滤，
  且**应用层强制注入 owner_id 条件**（双重保险，防漏过滤导致跨用户数据泄露）。
- **分块策略**：技术库按「一个技术 + 一个示例」为一块（语义完整）。
  **不用固定长度切分** —— 提问范式被切断就失去参考价值。
- 元数据必须含 `source_type`、`created_at`、`asserted_at`、`dimension`、`technique`。
- **Embedding 版本管理（v3.0 强化）**：
  记录三个字段而非一个：`embed_model`、`embed_dim`、`projection_version`。
  **任一变化都要求全量重建索引**，新旧向量**绝不可混用**
  （不同模型/维度/投影的向量空间不可比，混用让检索静默失效——最难发现的一类故障）。
  重建流程：双写新 collection → 在 §7.5 评估集上校验指标不降 → 切流 → 下线旧索引。
  Chroma 按 collection 名带上版本（如 `kb_zhipu1024_v1`），
  这样重建是"建新 collection"而非"改旧 collection"，回滚只需切配置。
- **按 scope 分流 provider（见 §3.3.1）**：
  `kb` 用云端 embedding、`user` 用本地 BGE-M3 是隐私最优组合。
  **两侧必须同维**，且各自独立记录上述三字段 ——
  `kb` 换模型不应触发 `user` 侧重建，反之亦然。

### 7.3 混合检索与多路召回 — MUST

【吸收 DeepSeek 的多路召回】

```
                      ┌→ 向量检索 top50 ─────────┐
Query 改写(HyDE/多查询)                          ├→ RRF 融合 → Reranker(本地) → top5
                      └→ BM25(jieba 分词) top50 ─┘
                      (可选)第二 embedding 模型 top50
```

**本项目的具体形态：**
向量侧 = 智谱 `embedding-3` 或本地 BGE-M3（按 scope 分流，§3.3.1）；
BM25 侧 = `rank-bm25` + `jieba`，**进程内**构建（数据量级千级块，无需 ES）；
Reranker = 本地 `bge-reranker-v2-m3`（无云端 rerank API 可用，见 §7.4）。

- 向量检索在"用户口语化表述"上召回差，**必须配 BM25 补关键词匹配**。
- 融合用 **RRF（倒数排名融合）**：只依赖排名不依赖分数，
  天然规避两路分数不可比的问题——这比 DeepSeek 提的"归一化后加权融合"更稳，
  因为归一化权重需要调参且随数据分布漂移。
- **多路 embedding 召回是 SHOULD 而非 MUST**：成本翻倍，
  必须用 7.5 的消融实验证明增益 > 3% 才保留。

### 7.4 Reranker 分数过近 / 不可靠 —— 专项方案 — MUST

RAG 落地最常见的坑。**核心原则：分数绝对值不可信，用"分布特征"而非"固定阈值"决策。**

1. **看相对差距，不看绝对值**
   `gap = (s1 - s2) / (s1 + ε)`。`gap < 0.05` 说明模型无法区分，**不要强行选 top1**。

2. **低区分度时的三种处置**（配置化，按场景选）
   - **全都要**（首选）：分数接近的前 N 条一起入上下文。
     提问场景容忍多个参考范式，这是最自然的解法。
   - **换维度决策**：用元数据二次排序（时间新近度、技术使用均衡性、维度匹配度）。
     "技术使用均衡性"尤其有用——避免 10 个问题全用同一种技术。
   - **弃用检索**（最重要）：若 top1 低于绝对下限，说明知识库无相关内容，
     **直接走无检索路径**。宁可不给参考，不给错参考——
     塞不相关片段进上下文比空着更糟，它会主动误导模型。

3. **分数校准（Calibration）**
   离线用标注集拟合分数→相关概率的映射（isotonic regression / Platt scaling），
   线上用**校准后的概率**做阈值判断，让阈值有可解释含义（"P(相关) > 0.7"而非"score > 0.42"）。
   模型或知识库更新后**必须重跑校准**。

4. **LLM 终审（吸收 DeepSeek，但严格限定）**
   【决策 #14】仅在以下**全部**条件满足时启用：
   `gap < 0.05` **且** 候选数 ≤ 5 **且** 处于关键路径（Planner 规划）。
   用小模型 + 0-3 分制 + 结构化输出。
   **不作为默认路径**：LLM 终审有效但增加延迟与成本，滥用会让检索链变得不可预测。

5. **不确定性显式传递**
   检索结果携带 `confidence` 与 `is_ambiguous` 进入提示词；
   低置信时提示词明确写"以下参考可能不适用，请优先基于用户原话"。
   **让下游知道上游不确定，而不是假装确定。**

6. **降级链**
   `Reranker → RRF 排名 → 纯向量 top-k → 无检索`
   任一环节超时（>800ms）或异常自动降级并记录。
   **降级是常态而非异常**，指标里必须能看到降级率。

7. **本项目的 Reranker 现实约束（v3.0）**
   当前**没有可用的云端 rerank API**，Reranker 只能本地跑。这带来三个后果：
   - **必须在 executor / 独立进程中执行**：cross-encoder 推理是 CPU/GPU 密集，
     直接在事件循环里跑会卡死整个 worker（§9.1 的典型陷阱）。
   - **模型未加载或加载失败时，降级到 RRF 排名是正常路径**，不是错误。
     首次加载耗时较长，应在启动预热而非首个请求时加载。
   - **P3 之前完全可以不做 Reranker**：先跑 §7.5-A 的消融实验，
     若 RRF 与 RRF+Rerank 的 Recall 差距 < 2%，直接砍掉（决策见 §7.5）。
     本产品知识库规模小、候选集质量高，Reranker 的边际收益很可能不值它的延迟与部署成本。

8. **LLM 终审用 `FAST` 档**（§2.4）：这是低难度打分任务，不需要 `STRONG`。
   用强档跑终审是成本浪费；用 `TOY` 档跑则毫无意义（打分不可信）。

【生成指令】实现 `app/rag/retriever.py`、`fusion.py`、`reranker.py`、`calibration.py`。
Reranker 封装在带超时与降级链的 wrapper 内。
`decide_from_scores()` 必须是纯函数，单测覆盖三类边界分布：全部接近 / 全部偏低 / 单峰突出。

### 7.5 RAG 效果评估方法 — MUST

**不建评估集就等于没有 RAG 质量概念。** 三层评估，成本递增：

**A. 检索层（零 LLM，成本极低，可每次提交跑）**
- 标注集：50-100 条 `query → 相关文档 id`（由具心理背景人员标注，见 3.8）。
- 指标：`Recall@k`、`MRR`、`nDCG@k`、`Hit Rate`。
- **消融对比必做**：纯向量 / 纯 BM25 / RRF / RRF+Rerank / +多路召回，量化每层增益。
  **Reranker 增益 < 2% 就砍掉**（省延迟省钱）；多路召回 < 3% 同理。用数据决策，不靠直觉堆组件。

**B. 生成层（RAGAS 式，需 LLM 评判）**
- `Faithfulness`（本系统更关键的是**是否忠于用户原话**而非忠于检索内容）、
  `Answer Relevancy`、`Context Precision`、`Context Recall`。
- LLM-as-judge 必须：**固定评判模型与版本 + 固定提示词 + 用人工标注校准**。
  否则评分漂移会让跨版本指标失去可比性——这是 LLM 评估最容易踩的坑。

**C. 端到端产品层（本系统特有，最重要）**
通用 RAG 指标衡量不了"提问质量"：
- **提问质量评分集**：20-30 个典型困惑场景，人工评分维度：
  深度（是否触及价值/恐惧层）、**非引导性（是否暗含预设答案）**、
  自然度、锐度适配、**是否违规给建议**。
  "非引导性"是苏格拉底方法的命门：一个暗含预设答案的问题不是提问，是伪装成问题的建议。
- **可程序化计算的零成本指标**：维度覆盖率、Evidence 校验通过率、
  技术分布均衡度、Critic 否决率。
- **在线指标**（吸收 DeepSeek）：用户对问题的"扎心度/有用度"评分、报告有用性评分、
  完成率、放弃点分布。
- **A/B 测试**（吸收 DeepSeek）：提示词版本、检索策略、提问模式分流对比。
  样本量需足够；心理产品的主观评分方差大，小样本结论不可信。

**回归门禁 — MUST**：提示词或模型变更后必须跑 A + C 的程序化部分，
关键指标下降超阈值则**阻断上线**。

### 7.6 评估集伦理（两版都缺）— MUST

- 评估集**默认使用合成/专家编写案例**，不得直接使用真实用户会话。
- 仅当用户显式授权 `eval_use` 且经**充分脱敏与改写**后，方可纳入。
- 评估集本身按敏感数据管理（访问控制 + 加密），纳入版本控制时**不得含真实个人信息**。

【生成指令】评估代码放 `eval/`（与 `app/` 同级），
提供 `eval/run.py` 一键跑全部评估并输出对比报告（markdown + json）。
数据放 `eval/datasets/*.jsonl`，纳入版本控制。

### 7.7 知识库构建管线（v3.0 新增）— SHOULD

用户可用的 Bocha / Tavily / OpenAlex 在本产品里有一个**明确且有限**的位置：
**离线构建提问技术库**。它们不参与任何在线请求路径。

#### 7.7.1 铁律：完全离线，不接触用户数据

【决策 #22】

- 管线是**独立 CLI**（`app/kb/build.py`），手动或定时运行，**不由 API 请求触发**。
- 搜索客户端**不得从 `SessionState` 读取任何字段**，只接受 CLI 参数与配置文件。
  **此约束需有测试用例**：注入一个假 `SessionState` 并断言搜索模块无法访问它。
- 产出物是**版本化的知识库文件**（`data/kb/techniques_v*.jsonl`），
  经**人工/专家审核**后才入向量库（§3.8）。
  **不允许"抓取即入库"** —— 未经审核的网络内容进提问技术库，
  等于让互联网决定系统怎么问用户问题。

#### 7.7.2 三个数据源的分工

| 源 | 用途 | 检索什么 | 不做什么 |
|---|---|---|---|
| **OpenAlex** | 提供**真实可核验的学术出处** | 焦点解决短期治疗、苏格拉底式提问、动机式访谈等的论文元数据（`doi`、标题、年份、摘要） | 不下载全文；不用摘要生成"诊断标准" |
| **Bocha** | 中文网检索 | 中文咨询实务中的提问范式表述、本地化措辞 | 不采集个案故事（隐私与版权） |
| **Tavily** | 国际网检索 | 英文提问技术清单、公开培训材料 | 同上 |

**OpenAlex 配额纪律（决策 #23）**：
10000 credits/日、每请求固定 10 credits ≈ **1000 请求/日**。
因此：**必须本地缓存响应**（按查询 hash 存盘）、**支持断点续跑**、
**每请求固定扣费与页大小无关 → 尽量用大页而非多次小页请求**。
配额耗尽时脚本应优雅退出并提示明日续跑，而不是不断重试。

#### 7.7.3 产出物的 schema（入库前必须满足）

```python
class TechniqueEntry(BaseModel):
    technique: Technique              # 必须映射到 §1.3 的枚举，不允许自由新增
    dimension: list[Dimension]        # 适用维度
    prompt_pattern: str               # 提问范式模板（中文）
    example: str                      # 一个具体示例
    applicable_when: str              # 适用条件
    contraindication: str | None      # 禁用条件（如高风险状态下不宜使用）
    sources: list[Source]             # 至少一条，见下
    reviewed_by: str                  # 审核人标识，未审核不得入库
    reviewed_at: datetime

class Source(BaseModel):
    kind: Literal["openalex", "web", "book"]
    ref: str                          # doi / work_id / URL
    title: str
    year: int | None
```

**`sources` 与 `reviewed_by` 是硬性非空** —— 这是决策 #23 的落地点：
§6.4 禁止模型编造文献，而正面的解法是**让引用只能来自库内真实记录**。
`Source.ref` 必须可程序化核验（doi 能解析、URL 能访问），
校验脚本 `app/kb/verify.py` 在入库前跑一遍。

**`contraindication` 字段值得单独说明**：某些提问技术在高风险状态下不宜使用
（例如对处于危机中的用户使用尖锐的代价问句）。
把禁用条件结构化，Planner 才能在 `risk_level >= medium` 时**程序化过滤**掉这些技术，
而不是依赖提示词里的一句叮嘱。这是 §3.4 处置矩阵在 RAG 侧的延伸。

#### 7.7.4 与决策 #6 的一致性

§6.4 决策 #6 禁止"引用文献给用户下结论"。本管线**不违反**该决策，因为：
知识库存的是**怎么问**（提问范式），不是**用户是什么**（诊断标准）。
`sources` 用于**内部可追溯与专家审核**，
**默认不展示给用户**；即便展示也只以"参考框架"形式，绝不作为对用户状态的判断依据。

【生成指令】`app/kb/` 下实现 `build.py`（CLI）、`sources/openalex.py`、`sources/bocha.py`、
`sources/tavily.py`、`verify.py`、`ingest.py`。
所有源客户端必须带本地缓存、超时、配额计数。
**`app/kb/` 不得 import `app/graph/`、`app/agents/`、`app/schemas/session*`** ——
用导入方向约束保证它无法接触会话数据（可用 lint 规则或架构测试强制）。

---

## 8. 记忆系统与记忆冲突

### 8.1 四层记忆 — MUST

| 层 | 内容 | 存储 | 生命周期 |
|---|---|---|---|
| 工作记忆 | 当前会话近 3 轮原文 | LangGraph State（Redis 热态） | 会话内 |
| 会话记忆 | 本次会话结构化摘要 | Postgres `sessions.summary` | 90 天默认 |
| 长期画像 | 稳定特征、价值观、反复主题、已尝试方案 | Postgres `memories`（结构化） | 用户删除前 |
| 语义记忆 | 可检索的历史表述片段 | pgvector `chunks(scope='user')` | 同上，带时间衰减 |

**长期画像用结构化字段而非自由文本**：自由文本画像会越写越长、无法校验、冲突时无法定位。

### 8.2 记忆冲突 —— 专项方案 — MUST

用户的想法**本来就会变**（三个月前想离职，现在想留下）。这不是数据错误，是**真实的时间演化**。

> **核心原则：不覆盖、不删除、不"纠正"用户 —— 记忆是带时间戳的观察序列，不是唯一真值。**

DeepSeek 提的"时间衰减 + 新信息权重更高"方向对，但**"更新画像"的思路仍隐含覆盖语义**。
终版明确：写入是 append，冲突是**关系**而非**错误**。

**冲突检测**：写入新条目时检索同 `topic` 既有条目，
用 NLI（蕴含/矛盾判断）或结构化字段比对识别矛盾。

**三种冲突类型，处置完全不同：**

1. **时间演化型**（想法确实变了）
   → 两条都保留，新条目标 `supersedes`，旧条目 `status='superseded'`。
   检索默认取最新，但**矛盾本身是绝佳的提问素材**：
   "你三个月前提到 X，现在倾向 Y，这个变化里发生了什么？"
   —— 这恰好是最好的苏格拉底式问题。**冲突在本产品里是资产，不是缺陷。**

2. **矛盾并存型**（同时持有冲突信念，如"想要自由" + "想要稳定"）
   → 两条都保留，共享 `conflict_group`，**绝不可消解**。
   这是 `InsightReport.inner_conflicts` 的核心输入，也是最有价值的洞察来源。

3. **提取错误型**（系统抽取错了）
   → `status='invalidated'`，软删除保留审计痕迹。
   **判定为提取错误必须有证据（用户显式否认），不能由系统自行判断**——
   系统自判会把"用户改变主意"误杀成"我记错了"，丢掉最有价值的演化信息。

**实施要点：**

- 检索按 `recency × confidence × relevance` 三因子加权，**不做硬性覆盖**。
  `recency` 用 `asserted_at` 而非 `created_at`。
- 【吸收 DeepSeek】**在报告中显式指出矛盾点并邀请澄清** —— 这与苏格拉底方法天然契合。
- **跨会话引用历史前必须先确认**：
  "我记得你之前提到过 X，还是这样吗？"
  直接断言记忆内容是体验灾难；记错了会瞬间摧毁信任，而信任是这个产品的全部基础。
- 用户可见的记忆管理入口：**可查看、可更正、可删除**（合规要求，也是信任基础）。
- 记忆写入需 `store_memory` 授权；跨会话调用需 `cross_session_recall` 授权。
  **未授权时功能降级而非报错**（单次会话仍完整可用）。

【生成指令】实现 `app/memory/store.py`（CRUD + 状态机）、`conflict.py`（检测与分类）、
`retrieval.py`（三因子加权）。`conflict.classify()` 必须可单测，
测试用例需完整覆盖上述三种类型 + 无冲突基线。

---

## 9. 并发、可靠性与延迟

### 9.1 并发模型 — MUST

- **全链路 async**：FastAPI + httpx.AsyncClient + async LLM client + asyncpg。
  LLM 调用是 IO 密集，任一处同步阻塞会让并发能力直接崩掉。
- **CPU 密集任务隔离**：tokenizer 计数、本地 embedding/rerank 推理、BM25 建索引、中文分词
  放 `run_in_executor` 或独立进程，避免阻塞事件循环。
  （这是 async 服务最常见的性能陷阱：一次同步 embedding 就能卡死整个 worker。）
- **无状态服务 + 外部状态**（吸收 DeepSeek）：会话状态在 Postgres/Redis，
  应用实例无状态，可水平扩容。
- **每会话串行化**：同一 `session_id` 的请求必须串行。
  Redis 分布式锁（`SET NX PX` + 唯一 token 释放，防误删他人锁；
  锁超时需大于最长节点耗时 + 余量，并支持看门狗续期）。
  并发提交同一会话的两个回答会导致状态错乱与问题序号错位。
- **乐观锁兜底**：`revision` 每次写入 +1，`UPDATE ... WHERE revision = :expected`，
  影响行数 0 则冲突，返回 409。
  **锁 + 版本号双保险**：Redis 抖动导致锁失效时，版本号仍能防脏写。
- **幂等**：写接口接受 `Idempotency-Key`，Redis 记录 键→结果，TTL 24h。
  网络重试不得造成重复推进问题序号。

### 9.2 LLM 调用可靠性 — MUST

集中封装 `app/llm/gateway.py`，**业务代码禁止直接调 SDK**。Gateway 负责：

1. **超时分级**（不同任务用同一超时值是常见错误）：
   安全分类 3s / 质量评估 5s / 生成问题 15s / 摘要 20s / 报告分段 60s。
   **与 `LLM_TIMEOUT_SECONDS=180` 的关系**：180s 是 **HTTP 客户端的外层上限**
   （给推理型模型留足余量），上述值是**每任务的内层截止时间**，由 Gateway 强制。
   两者不是重复配置：外层防连接悬挂，内层防单个任务拖垮整体节奏。
   **若目标模型是推理型**（输出前有较长思维链），
   快档任务的内层值需按 §17 探测得到的实测 P95 上调，**不要照抄本表**。
2. **选择性重试**：指数退避 + 抖动（`base=1s, factor=2, jitter=±30%, max=3`）。
   仅对 429/5xx/超时/连接错误重试；
   **4xx 参数错误与内容策略拒绝不重试**（重试必然再失败，纯浪费配额与时间）。
3. **熔断器**：滑动窗口失败率 > 50% 触发熔断，半开试探恢复。
   避免下游故障时请求堆积拖垮自己。
4. **降级链**：`主模型 → 备用模型 → 备用供应商 → 模板/缓存兜底`。
   【关键】降级必须**用户可感知**（"当前使用简化模式"），不可静默降低质量。
   **本项目的现实缺口（决策 #24）**：当前只有 DeepSeek 一家对话供应商，
   "备用供应商"这一环**是空的** —— 单厂故障即全站不可用。
   补齐方案按优先级：
   ① 智谱同一 key 若可调对话模型（GLM 系），作为**首选兜底**（已有凭据，接入成本最低，需实测确认）；
   ② 本地 Ollama 托管的 7B 指令模型作 `FAST` 档兜底（`STRONG` 任务仍无法兜底，只能排队等恢复）；
   ③ 再接一家云厂商。
   **在兜底补齐之前，`STRONG` 任务的降级链末端只能是"明确失败 + 保留会话状态待恢复"** ——
   不允许用 `FAST` 或 `TOY` 档顶替 `STRONG` 任务（§2.4.3 第 3 条），
   宁可让用户稍后继续（会话可续跑，§4.2），也不给出劣质的心理分析。
5. **限流适配**：本地令牌桶按 **RPM 与 TPM 双维度**预先削峰，
   而非等供应商返回 429 再退避。
6. **并发闸门**：全局 semaphore 限制在途请求数（吸收 DeepSeek 的"每模型最大并发"），
   防突发流量打爆配额。
7. **成本计量**：每次调用记录 prompt/completion tokens 与预估成本，按 session 与 user 聚合，
   支持**单会话成本上限熔断**与**单用户日配额**（防恶意刷量，见 §3.7）。
8. **参数按能力裁剪**：`supports_temperature=False` 时不传 `temperature`；
   不支持 `response_format` 时不传该字段。
   **无条件全带参数会在换模型时收到 400**，而 400 不可重试（第 2 条），直接失败。
9. **嵌入服务的可靠性同等对待**：`EmbeddingProvider` 也要有超时、重试、熔断与降级。
   嵌入失败的降级路径是**跳过向量检索走 BM25**，而不是让整个请求失败 ——
   检索是增强项，不是必需项（§7.4 的"无检索"是合法路径）。
   `EMBED_BATCH_SIZE=32` 需在批量索引时遵守；**在线单条查询不要走批量接口**。

### 9.3 关键设计：安全层不依赖 LLM 可用性 — MUST

**LLM 全挂时，规则层危机检测必须仍然工作。**

若安全分类器不可用且规则层未命中，系统采取**保守策略**：
不进入深度提问（因为无法评估风险），转温和模式或告知稍后重试。
**绝不在无法评估安全的情况下继续犀利追问。**

这条是两版文档里最容易被"降级到备用模型就行"的思路掩盖的：
备用模型也可能同时不可用（同供应商故障），而安全不能有单点依赖。

**v3.0 补充：本项目当前的单点风险是真实存在的。**
只有一家对话供应商（决策 #24），意味着"LLM 全挂"不是理论场景。因此：

- **规则层词表必须能独立完成危机拦截**，这是 §3.4 第一层存在的全部理由。
- 向量库 / 嵌入服务不可用时，安全判定**不受影响**（安全不依赖检索）。
  这要求 Safety 节点**不得**把检索结果作为判级输入 —— 保持它的输入只有用户最新输入。
- 本地 `FAST` 档模型（若已部署）可作为安全分类器的兜底，
  这是**本地模型在本项目中唯一真正有价值的生产用途**：
  它不追求质量最优，而是追求**在云端不可用时安全能力不为零**。

### 9.4 延迟优化 — SHOULD

| 手段 | 说明 |
|---|---|
| SSE 流式 | 首 token 时间是感知延迟关键；问题文本边生成边推送 |
| 并行化 | 安全检查与检索并行（`asyncio.gather`），二者无依赖 |
| 前缀缓存 | 系统指令与技术库放提示词头部保持稳定，命中供应商 prompt 缓存 |
| 分批返回 | `BATCHED` 模式先返回前 3 问，后台生成后续（吸收 DeepSeek） |
| 投机预取 | 用户答题时后台预取下一问所需检索片段 |
| 小模型分流 | 安全分类、质量评估、摘要用小模型；仅提问与分析用强模型 |
| 报告异步 | 60s 级任务转后台队列 + 完成通知（SSE/轮询），避免请求挂死 |
| 语义缓存 | **仅缓存技术库检索结果**；绝不缓存针对用户的问题与分析 |

【决策 #11】**预生成只做参考范式，不做最终问题文本。**
DeepSeek 提的"预生成常见困惑的提问模板"若直接返回给用户，
会摧毁个性化——而个性化提问正是本产品的唯一核心价值。
模板可以作为 Planner 的输入，不能作为 Interviewer 的输出。

**分析报告明确接受更高延迟**：宁可 30 秒给出有依据的分析，
不要 3 秒给出泛泛而谈。用进度反馈掩盖延迟，而非牺牲质量。

### 9.5 失败可见性 — MUST

- 每个降级、重试、熔断、截断、检索弃用事件都要打结构化日志并计入指标。
- **静默失败最危险**：报告少了一半依据但接口返回 200，用户会当成真实结论使用。
- 任何影响输出质量的降级必须在响应中带 `degraded: true` + 原因，
  并持久化到 `reports.degraded`（便于事后分析哪些报告不可靠）。

---

## 10. API 契约（两版都缺，终版新增）

下游生成前后端代码需要明确契约，否则会各自发明接口。

### 10.1 端点 — MUST

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/v1/sessions` | 创建会话（body: concern, mode, sharpness, consent） |
| GET | `/api/v1/sessions/{id}` | 获取会话状态与进度 |
| POST | `/api/v1/sessions/{id}/answers` | 提交回答（需 `Idempotency-Key`） |
| GET | `/api/v1/sessions/{id}/stream` | SSE：问题流式输出、阶段变更、降级通知 |
| POST | `/api/v1/sessions/{id}/report` | 触发报告生成（异步，返回 task_id） |
| GET | `/api/v1/tasks/{task_id}` | 查询异步任务状态 |
| POST | `/api/v1/sessions/{id}/pause` | 主动暂停 |
| POST | `/api/v1/feedback` | 提交反馈（评分/评论） |
| GET | `/api/v1/me/memories` | 查看记忆 |
| DELETE | `/api/v1/me/memories/{id}` | 删除单条记忆（级联向量库） |
| POST | `/api/v1/me/export` | 数据导出（异步） |
| DELETE | `/api/v1/me` | 账号与数据删除 |
| GET | `/health` `/ready` | 健康检查（ready 需校验 DB/Redis/Provider 连通） |
| GET | `/metrics` | Prometheus 指标 |

### 10.2 SSE 事件类型 — MUST

```
event: phase        data: {"phase":"ask","progress":{"current":4,"total":10}}
event: question     data: {"index":4,"delta":"你最怕失去的"}      # 流式增量
event: question_end data: {"index":4,"technique":"cost_of_loss"}
event: safety       data: {"level":"medium","message":"..."}      # 关怀性介入
event: degraded     data: {"reason":"reranker_timeout","impact":"参考质量下降"}
event: error        data: {"code":"LLM_UNAVAILABLE","retryable":true}
event: done         data: {"phase":"done"}
```

### 10.3 错误码 — MUST

| Code | HTTP | 可重试 | 说明 |
|---|---|---|---|
| `CONSENT_REQUIRED` | 403 | 否 | 未同意或条款版本过期 |
| `SESSION_CONFLICT` | 409 | 是 | 乐观锁冲突，重取后重试 |
| `SESSION_LOCKED` | 423 | 是 | 同会话有请求在处理中 |
| `RATE_LIMITED` | 429 | 是 | 含 `Retry-After` |
| `BUDGET_EXCEEDED` | 402 | 否 | 会话/日成本上限 |
| `LLM_UNAVAILABLE` | 503 | 是 | 全部降级链耗尽 |
| `STRUCTURED_OUTPUT_FAILED` | 502 | 是 | 三次解析失败 |
| `CRISIS_STOPPED` | 200 | 否 | **非错误**：安全终止，返回资源信息 |
| `SESSION_EXPIRED` | 410 | 否 | 超 14 天建议新建会话 |

`CRISIS_STOPPED` 用 200 而非错误码：它是**正常的产品路径**，
前端不应按错误渲染（弹红色报错框会加重危机中用户的负面体验）。

---

## 11. 可观测性

### 11.1 结构化日志 — MUST

- loguru + JSON sink，固定字段：`ts` `level` `trace_id` `session_id` `user_id_hash`
  `node` `agent` `event` `duration_ms` `tokens` `cost` `degraded` `prompt_version` `model`。
- **隐私红线（最重要）**：
  **用户原话不进普通日志。** 仅在显式开启的调试模式 + 用户授权下记录，且必须过 PII 脱敏。
  `user_id` 一律哈希。**Sentry 必须配 `before_send` 钩子剥离原文与 PII** ——
  异常上报是最容易泄露用户心理独白的通道，两版文档都没提这点。
- 危机事件走独立审计日志（见 5.2 `safety_audit`）。

### 11.2 Trace — MUST

- OpenTelemetry span 覆盖：每个 graph 节点、每次 LLM 调用、每次检索/重排/embedding。
- span 属性：模型名、prompt 版本 hash、token 数、是否降级、检索命中数与**分数分布**。
- 一次会话的完整 trace 必须可回放 —— 这是排查"为什么问了个奇怪的问题"的唯一途径。
- SHOULD：接 LangSmith / Langfuse 做 LLM 专项调试（可与 OTel 并存）。

### 11.3 核心指标（Prometheus）— MUST

**技术**：分节点 P50/P95/P99 延迟、LLM 错误率/重试率/熔断次数、
降级率（按类型）、token 与成本（按会话/用户）、上下文截断触发率、锁等待时长。

**质量**：维度覆盖率、Evidence 校验通过率、Critic 否决率、追问触发率、
结构化输出解析失败率、Reranker 低区分度触发率、检索弃用率。

**产品**：会话完成率、**放弃点分布（第几问流失最多）**、
报告有用性评分、问题"扎心度"评分、跨会话回访率。

**安全**：各风险等级触发次数、锐度自动降级次数、OutputGuard 拦截次数与类型、
注入检测命中次数。

> **「第几问流失最多」是最有价值的产品指标** —— 它直接指出哪一问设计得太重或太突兀，
> 是优化提问序列的第一手依据。

### 11.4 告警 — MUST

| 条件 | 级别 | 动作 |
|---|---|---|
| 危机事件触发 | 高 | 立即通知（若开启人工值班） |
| Evidence 校验通过率骤降 | 高 | 疑似模型或提示词回退，考虑回滚 |
| LLM 全降级链耗尽 | 高 | 服务实质不可用 |
| 熔断开启 | 中 | 关注下游供应商状态 |
| 成本小时环比翻倍 | 中 | 排查滥用或死循环 |
| 检索弃用率 > 30% | 低 | 知识库覆盖不足 |

---

## 12. 测试策略

| 层级 | 范围 | 要点 |
|---|---|---|
| 单元 | 纯函数与领域逻辑 | technique/dimension/mode/budgeter/conflict/policy/scores，**零 LLM 依赖，100% 覆盖** |
| 契约 | Pydantic 模型 | 边界值、校验器、序列化往返 |
| 集成 | Graph 流转 | **FakeLLM** 跑完整状态机：三种模式 + 危机分支 + 追问 + Critic 否决重做 + PAUSED 恢复 |
| 数据 | 存储层 | 迁移、级联删除（**必须验证向量库与缓存同步清理**）、乐观锁冲突 |
| 评估 | 提示词质量 | 见 7.5，上线门禁 |
| 混沌 | 故障注入 | LLM 超时/429/非法 JSON、向量库不可用、Redis 宕机、Reranker 超时、**供应商全挂** |
| 红队 | 对抗 | 提示词注入、危机词中文变体、诱导越界建议、诱导医疗诊断、诱导暴露他人隐私 |
| 负载 | 并发 | 同会话并发提交、突发流量、长会话上下文压测 |
| **能力门控** | Provider 能力矩阵 | `TOY` 档被拒绝承担 `STRONG` 任务；缺 `json_mode` 被拒；参数按 caps 裁剪（§2.4.3） |
| **档位冒烟** | 三档运行 | `local` 档全流程跑通（CI 必跑，用 minimind/Ollama，零云端成本）；`cloud`/`hybrid` 按需 |
| **架构隔离** | 导入方向 | `app/kb/` 不得 import 会话相关模块；注入假 `SessionState` 断言搜索模块无法访问（§7.7.1） |
| **维度一致性** | 向量库 | collection 维度与 `EMBED_TARGET_DIM` 不符时启动失败；换模型后拒绝混用（§7.2） |
| **输出截断** | 报告分段 | 模拟输出触顶，验证分段生成与局部重试；验证某段失败时报告仍出且 `degraded=true`（§6.5） |

**四条硬要求：**

1. **FakeLLM 是整套测试的地基，必须最先实现。** 它让状态机测试确定、快速、免费。
   FakeLLM 需支持：预设返回序列、模拟超时、返回非法 JSON、模拟拒答。
2. 危机红队用例集 `tests/redteam/crisis_cases.jsonl` **只增不减**（回归防护）。
   每发现一个绕过案例就永久加入。
3. **级联删除必须有测试**：向量库残留是最常见的"假删除"合规缺陷。
4. CI 门禁：单元 + 契约 + 集成 + ruff + mypy strict 全绿；
   评估测试按需/定时触发（因需真实 LLM 调用）。
5. **`local` 档冒烟进 CI**：这是 minimind 在本项目最有价值的用途（§2.5）——
   它让"provider adapter / 流式 / 超时 / 重试链路真的通"这件事**每次提交都被验证**，
   且完全离线免费。**但断言只针对流程与 schema，不针对内容质量。**
6. **不要用 FakeLLM 替代能力门控测试**：FakeLLM 可以声明任意 caps，
   因此必须单独测"caps 不足时 Gateway 确实拒绝"，而不是测"FakeLLM 返回了正确结果"。

---

## 13. 工程规范与部署

### 13.1 目录结构 — MUST

```
app/
  main.py                FastAPI 入口
  config.py              pydantic-settings（启动即校验，缺失快速失败）
  api/                   路由 / 依赖 / 中间件（鉴权、限流、幂等、trace）/ 错误码 / SSE
  graph/                 LangGraph 节点与图装配（一节点一文件）
  agents/                planner / interviewer / safety / analyst / critic
  domain/                technique.py / dimension.py / mode.py（纯逻辑，零依赖）
  schemas/               Pydantic 模型
  llm/                   gateway.py / retry.py / breaker.py / budget.py / capability.py
    providers/           openai_compat.py（DeepSeek 与本地共用）/ local_minimind.py / fake.py
  embeddings/            provider.py（按 scope 分流，§3.3.1）/ zhipu.py / local_bge.py / projection.py
  rag/                   retriever / fusion / reranker / calibration / indexer / vector_store.py
  kb/                    build.py（离线 CLI）/ sources/ / verify.py / ingest.py —— 见 §7.7
                         禁止 import graph/agents/session（架构测试强制）
  memory/                store / conflict / retrieval
  context/               budgeter / summarizer
  safety/                lexicon / classifier / policy / pii / output_guard / injection
  storage/               repositories / migrations（alembic）
  tasks/                 arq worker（报告生成、导出、索引重建）
  observability/         logging / tracing / metrics
  mcp_server/            MCP 能力暴露（无状态能力，见 2.3）
prompts/                 外置提示词（jinja2，按 agent/version）
config/                  crisis_resources.yaml / task_models.yaml / feature_flags.yaml
eval/                    datasets/ + run.py + metrics/
tests/                   unit / contract / integration / chaos / redteam / load
docs/                    本文档 + ADR
```

### 13.2 配置与 Feature Flag — MUST

- 所有密钥只从环境变量读，**不入库不入仓**；提供 `.env.example` 列全变量与注释。
- 启动时校验必需配置，缺失**快速失败**（不允许带空 API key 启动后运行时才炸）。
- 环境分离 `dev/staging/prod`：模型档位、日志级别、锐度上限、留存期按环境配置。
- **Feature Flag**（两版都缺）：`human_handoff`（转人工）、`sharp_mode`、
  `cross_session_recall`、`multi_route_recall`、`llm_final_judge`、
  `local_embeddings`、`reranker_enabled`。
  高风险能力必须可**一键关闭**而不需发版——出问题时这是唯一的快速止血手段。

### 13.2.1 三档运行档位（v3.0 新增）— MUST

用户要求"尽可能多用本地模型且可切换"。把这件事做成**三个预设档位**，
而不是十几个互相牵连的开关 —— 单个开关的组合空间太大，测不完也说不清。

| 档位 | LLM | Embedding | Reranker | 适用场景 |
|---|---|---|---|---|
| `cloud` | DeepSeek 强/快档 | 智谱 `embedding-3`（两 scope 都用） | 本地或关闭 | 质量优先，起步默认 |
| `hybrid`（**推荐生产**） | DeepSeek 强/快档 | `kb`=智谱、`user`=**本地 BGE-M3** | 本地 | **隐私最优的可行组合**：最敏感的用户历史不出本机（§3.3.1） |
| `local` | 本地 Ollama 指令模型（`FAST`）+ minimind（`TOY`） | 本地 BGE-M3 | 本地 | 离线开发 / CI / 数据不出境的合规场景 |

**关键约束：**

- `local` 档下**没有 `STRONG` 能力**（消费级硬件难以满足长上下文 + 强指令遵循）。
  因此该档**只允许用于开发、CI、演示**；
  若要在生产用 `local`，必须先在 §7.5 评估集上量化质量下降，
  并明确告知用户"当前为本地模式，分析深度受限"（§9.5 的 `degraded` 语义）。
- 切换档位**只改配置**，不改代码。
  **切换 embedding 档位必须触发索引重建**（§7.2），启动时按 `embed_model + embed_dim`
  校验 collection 一致性，**不一致就快速失败**，不允许混用。
- 每个档位都必须有一条 CI 冒烟用例（至少跑通 `local` 档全流程）。

### 13.2.2 环境变量清单 — MUST

`.env.example` 必须列全以下变量（**只写变量名与说明，不写真实值**，§2.7）：

```bash
# ---- 运行档位 ----
RUNTIME_PROFILE=cloud            # cloud | hybrid | local

# ---- LLM（OpenAI 兼容协议；DeepSeek 与本地模型共用同一 adapter）----
LLM_API_KEY=
LLM_BASE_URL=                    # 如 https://api.deepseek.com/v1
LLM_MODEL_STRONG=                # 强档模型标识（Planner/Interviewer/Analyst/Critic）
LLM_MODEL_FAST=                  # 快档模型标识（Safety/Quality/Summary）
LLM_TEMPERATURE=0.3              # 分析阶段应更低，见 §6.4；provider 不支持时自动省略
LLM_MAX_TOKENS=4096              # 输出上限，驱动 §6.5 的报告分段
LLM_TIMEOUT_SECONDS=180          # HTTP 外层上限；任务内层截止见 §9.2

# ---- 备用供应商（决策 #24，兜底，缺失时降级链末端为显式失败）----
LLM_FALLBACK_BASE_URL=
LLM_FALLBACK_API_KEY=
LLM_FALLBACK_MODEL=

# ---- 本地 LLM ----
LOCAL_LLM_BASE_URL=              # Ollama/vLLM/minimind serve 的 OpenAI 兼容地址
LOCAL_LLM_MODEL=
LOCAL_LLM_TIER=toy               # toy | fast —— 绝不允许填 strong（§2.4.3）

# ---- Embedding（云）----
EMBED_API_KEY=
EMBED_BASE_URL=
EMBEDDING_MODEL=
EMBED_DIM_NATIVE=2048            # 供应商原生维度（实测值）
EMBED_TARGET_DIM=1024            # 目标统一维度，见 §2.6【决策 #19】
EMBED_BATCH_SIZE=32

# ---- Embedding（本地）----
LOCAL_EMBED_MODEL=BAAI/bge-m3    # 原生 1024 维
LOCAL_EMBED_DEVICE=cpu           # cpu | cuda

# ---- Reranker（仅本地）----
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RERANKER_ENABLED=false           # 先做 §7.5 消融再决定是否开启
RERANKER_TIMEOUT_MS=800

# ---- 向量库 ----
VECTOR_BACKEND=chroma            # chroma | pgvector
CHROMA_PERSIST_DIR=./data/chroma
CHROMA_COLLECTION_KB=            # 名称须含模型与维度版本，如 kb_zhipu1024_v1
CHROMA_COLLECTION_USER_PREFIX=

# ---- 存储 ----
DATABASE_URL=                    # 开发可 sqlite+aiosqlite，生产 postgresql+asyncpg
REDIS_URL=

# ---- 知识库构建（离线 CLI 专用，绝不接触用户数据，§7.7）----
BOCHA_API_KEY=
TAVILY_API_KEY=
OPENALEX_BASE=https://api.openalex.org
OPENALEX_API_KEY=
OPENALEX_DAILY_CREDITS=10000     # 每请求扣 10 → 约 1000 请求/日
KB_CACHE_DIR=./data/kb_cache

# ---- 安全与合规 ----
POLICY_VERSION=                  # 见 §3.2 同意版本化
CRISIS_RESOURCES_FILE=config/crisis_resources.yaml
SESSION_RETENTION_DAYS=90

# ---- 可观测 ----
SENTRY_DSN=                      # 必须配 before_send 脱敏，见 §11.1
OTEL_EXPORTER_OTLP_ENDPOINT=
LOG_LEVEL=INFO
DEBUG_LOG_RAW_TEXT=false         # 记录用户原话，仅调试且需授权，见 §11.1
```

**启动自检 — MUST**（缺一项就 fail fast，不允许"先跑起来再说"）：

1. 必需变量非空；`RUNTIME_PROFILE` 合法。
2. `LOCAL_LLM_TIER != strong`（硬断言）。
3. `EMBED_TARGET_DIM` 与向量库 collection 实际维度一致（§2.6 的 `dim` 属性）。
4. `VECTOR_BACKEND=pgvector` 且 `EMBED_TARGET_DIM > 2000` → **直接报错**并提示 §2.6。
5. `crisis_resources.yaml` 存在且可解析、条目非空 ——
   **危机资源缺失时不允许启动**，这比任何功能都优先（§3.4）。

### 13.3 演进路径（每阶段可独立交付）

| 阶段 | 目标 | 交付内容 |
|---|---|---|
| **P0 骨架** | 端到端跑通 10 问 | schemas + domain + **规则层安全** + Graph 状态机 + FakeLLM + CLI |
| **P1 可用** | 真实 LLM | LLMGateway（超时/重试/降级/**能力门控**）+ `OpenAICompatProvider`（DeepSeek）+ **JSON 模式结构化输出** + Interviewer/Analyst（**报告分段**）+ SQLite |
| **P2 可靠** | 抗故障与并发 | 熔断 + 锁 + 幂等 + 上下文预算 + Critic + OutputGuard + API 契约 + 可观测 |
| **P3 智能** | RAG 与记忆 | pgvector + 混合检索 + Reranker（低区分度处置）+ 四层记忆 + 冲突处理 |
| **P4 质量** | 评估驱动 | 评估集 + 回归门禁 + 提示词 A/B + 校准 + 红队集 + 反馈闭环 |
| **P5 规模** | 生产化 | Postgres + Redis + 容器 + K8s/HPA + 成本治理 + （必要时）服务拆分 |

**强制顺序约束（下游工具不得调整）：**

- **安全规则层在 P0 必须存在**，不可延后。这是唯一不能"以后再说"的模块。
- **P2 之前不对外开放**（缺可靠性与输出防护）。
- **RAG 放 P3**：先把提问流程本身做好。
  过早引入 RAG 会掩盖提问设计的问题，且没有评估集时无法判断 RAG 是否真有帮助。
- **知情同意与数据删除随 P1 一起上**（一旦开始存真实用户数据就必须具备）。
- **能力门控随 P1 一起上**（§2.4.3）：P1 就要接真实 provider，
  此时若无门控，后续任何模型切换都可能静默劣化，而且越晚补越难发现已被污染的输出。
- **本地模型接入放 P1 末尾**：先用 DeepSeek 把链路跑通再加本地 adapter，
  否则会把"模型能力不足"与"代码有 bug"两类问题混在一起排查。
- **§17 探测清单在 P1 开始前跑一次**，把实测结果写进配置 ——
  本文档所有"待实测"的地方都依赖它。

### 13.4 部署 — MUST

- Docker 多阶段构建；`docker-compose`（app + postgres + redis + worker）用于开发与小规模。
- P5 迁 K8s：**HPA 依据 = CPU + 请求队列长度 + 在途 LLM 请求数**
  （吸收 DeepSeek；但仅 CPU 不够，因为瓶颈在等待外部 API 而非 CPU）。
- SSE 长连接需注意：ingress/LB 超时需大于最长流式时长，禁用响应缓冲。
- 【决策 #12】**Serverless 明确 REJECTED**：长会话有状态 + SSE 流式 + 冷启动 + 本地模型缓存
  都与 Serverless 模型冲突。
- 数据库迁移用 alembic，**向前兼容**（先加字段 → 双写 → 切读 → 删旧字段）。

---

## 14. 成本模型与反馈闭环

### 14.1 成本模型（两版都缺）— MUST

单会话成本必须**可估算、可上限、可归因**，否则规模化时会失控。

**单会话 LLM 调用次数估算（BATCHED 模式）：**

| 用途 | 次数 | 模型档 |
|---|---|---|
| 规划 / 重规划 | 3-4 | 强 |
| 生成问题 | 10 | 强 |
| 安全分类 | ~12（仅可疑时触发） | 弱 |
| 回答质量评估 | 10 | 弱 |
| 追问生成 | 0-5 | 强 |
| 摘要压缩 | 2-4 | 弱 |
| 分析报告 | 1（长输出） | 强 |
| Critic 复核 | 1-3 | 中 |
| **合计** | **~40-50 次** | 混合 |

- **PROGRESSIVE 模式成本约为 BATCHED 的 1.5-2 倍**，应作为付费档或限量功能。
- 控制手段：小模型分流（弱档承担约一半调用量）、前缀缓存、
  `Question.text` 长度上限、上下文硬预算、单会话 token 熔断。
- 每阶段都要记录实测成本；**P4 评估阶段必须产出"成本 vs 质量"曲线**，
  用于决定各环节用哪个档位的模型——这个决策不能靠猜。

**v3.0 修订：报告分段改变了调用次数**

§6.5 把报告拆成 3 段，Critic 相应也按段复核，因此：

| 用途 | v2.0 估算 | v3.0 修订 | 档位 |
|---|---|---|---|
| 分析报告 | 1（长输出） | **3**（诉求/冲突/建议，各自更短） | `STRONG` |
| Critic 复核 | 1-3 | **3-6**（按段复核 + 重做） | `STRONG` |
| **合计** | ~40-50 | **~45-58** | 混合 |

调用次数上升但**单次输出显著变短**，总 token 大致持平，
而**结构化失败率明显下降** —— 这是用调用次数换可靠性，值得。

**本地模型对成本的真实影响：**

- `hybrid` 档把 `user` scope 的嵌入移到本地：
  嵌入调用量随会话轮数线性增长（每条用户表述都要向量化），
  移到本地省下的是**持续成本**，且顺带解决最敏感数据的出境问题（§3.3.1）。
  这是三档里性价比最高的一步。
- 把 `FAST` 档任务（安全/质量/摘要，约占调用次数一半）移到本地 7B 模型，
  可再省一块云端成本，**但必须先通过 §17 探测确认 JSON 稳定性** ——
  一个不稳定的本地模型会把省下的钱以"重试 + 解析失败"的形式还回去，甚至更多。
- **成本不是选本地的首要理由，隐私是**（§3.3.1）。
  若只为省钱而牺牲 `STRONG` 档质量，等于牺牲产品的核心资产（§1.1）。

### 14.2 用户反馈闭环 — MUST

【吸收 DeepSeek，并补上"回流到哪里"这个关键缺失】

DeepSeek 提到"收集用户评分用于微调或调整 RAG 权重"，但没说清回流路径。终版明确：

```
用户评分/评论 → feedback 表
    ├→ 低分问题 → 人工审阅 → 加入提问质量评估集（负例）
    ├→ 低分报告 → 检查 Evidence 校验日志 → 定位是检索问题还是生成问题
    ├→ 漏检危机 → 立即加入 redteam/crisis_cases.jsonl（最高优先级）
    └→ 高分问题 → 提炼为技术库新范式（需专家审核后入库）
```

- 反馈**不直接用于自动微调**：心理场景样本量小、主观方差大，
  自动微调极易把偶发偏好固化成系统偏差。人工审阅是必经环节。
- 反馈进评估集需经 7.6 的伦理流程（脱敏 + 授权）。
- 闭环有效性本身要度量：加入负例后，下一版在该负例上是否改善。

---

## 15. 风险清单与对策速查

| 风险 | 影响 | 对策 | 章节 |
|---|---|---|---|
| 用户处于心理危机 | 严重伤害 | 规则+模型双层，硬编码处置矩阵，不依赖 LLM 可用性 | 3.4 / 9.3 |
| 危机词中文变体绕过 | 漏检 | 谐音/拆字/拼音覆盖 + 红队集只增不减 | 3.4 / 12 |
| 承诺人工却无人响应 | 信任崩塌 | 转人工默认关闭，feature flag 控制 | 3.4 / 13.2 |
| 模型编造洞察 | 误导决策 | Evidence 原话子串程序化校验 + Critic 独立复核 | 6.4 |
| 编造心理学文献引用 | 伪权威 | 禁止文献式结论；只引知识库真实 id | 6.4 |
| 长会话上下文超长 | 截断丢信息 | 硬预算 + 分层压缩 + 金句保留 + 优先级裁剪 | 6.3 |
| 中文 token 估算偏差 | 意外截断 | 用目标模型 tokenizer 实测，禁用字符数估算 | 2.2 / 6.3 |
| 提问变成给建议 | 违背方法论 | Agent 职责隔离 + OutputGuard + 评估集打分 | 4.4 / 3.7 |
| 问题暗含预设答案 | 伪苏格拉底 | 评估集"非引导性"维度 + 专家审阅 | 7.5 / 3.8 |
| Reranker 分数不可区分 | 检索不稳 | 相对差距 + 校准 + 低分弃检索 + 受限 LLM 终审 | 7.4 |
| Embedding 换版混用 | 检索静默失效 | `embed_model` 记录 + 强制全量重建 | 7.2 |
| 中文 BM25 失效 | 关键词召回为零 | 必须接中文分词器 | 2.2 |
| 记忆冲突处理错误 | 丢失演化信息 / 记错伤信任 | 不覆盖，三类冲突分治，引用前确认 | 8.2 |
| 跨用户数据串味 | 隐私事故 | scope 隔离 + 应用层强制 owner_id 过滤（双保险） | 7.2 |
| 并发导致状态错乱 | 数据损坏 | 会话级分布式锁 + 乐观锁 + 幂等键 | 9.1 |
| 同步调用阻塞事件循环 | 吞吐崩塌 | CPU 密集任务隔离到 executor | 9.1 |
| LLM 不稳定 | 服务中断 | 超时分级 + 选择性重试 + 熔断 + 多级降级 | 9.2 |
| 静默降级 | 用户误信劣质输出 | 全链路 degraded 标记 + 响应可见 + 持久化 | 9.5 |
| 提示词注入 | 越权/越界 | 数据隔离包裹 + 指令复述 + 剥离而非报错 | 3.7 |
| 用户数据出境给供应商 | 合规风险 | 零留存条款 + DPA + 隐私政策明示 + 自托管路径 | 3.3 |
| Sentry 泄露用户原话 | 隐私事故 | `before_send` 钩子强制脱敏 | 11.1 |
| 假删除（向量库残留） | 合规风险 | 级联删除 + 专项测试用例 | 3.6 / 12 |
| 同意条款变更 | 合规失效 | 同意版本化 + 分项授权 + 重新征求 | 3.2 |
| 未成年人使用 | 严重风险 | 强制 GENTLE + 关闭恐惧维度 + 监护人提示 | 3.2 |
| 成本失控 | 经济损失 | 计量 + 会话/日上限熔断 + 小模型分流 + 前缀缓存 | 9.2 / 14.1 |
| 无质量基线 | 改坏了不知道 | 三层评估 + 回归门禁 | 7.5 |
| 评估集用真实数据 | 合规风险 | 默认合成/专家案例；授权+脱敏方可纳入 | 7.6 |
| 会话跨天恢复情绪错位 | 体验差 | PAUSED 重确认；超 14 天建议新会话 | 4.5 |
| 过早微服务化 | 交付变慢 | 模块化单体 + 接口预留 + 明确拆分优先序 | 4.1 |
| **弱模型顶替强档任务** | **输出静默劣化（最危险）** | 能力门控 + `TOY` 硬阻断 + 拒绝路由计指标 | 2.4 / 2.5 |
| **2048 维无法建 pgvector 索引** | 检索退化为全表扫描 | 统一 1024 维 + Chroma 起步 + 启动维度自检 | 2.6 |
| **换嵌入模型/维度后混用向量** | 检索静默失效 | 三字段记录 + collection 带版本名 + 启动校验 | 7.2 |
| **输出触顶截断致 JSON 解析稳定失败** | 报告无法生成 | 报告分段生成 + 输出预算按绝对值算 + 局部重试 | 6.5 |
| **function calling 不被支持** | 换模型即整链失效 | JSON 模式为主路径 + 按 caps 选择 + 参数裁剪 | 6.2 |
| **单一对话供应商** | 单厂故障即全站不可用 | 补备用供应商；未补齐前 `STRONG` 降级末端为显式失败 | 9.2 |
| **用户原话流向嵌入服务被忽视** | 隐私暴露量被低估 | 视嵌入为出境通道；`hybrid` 档 `user` scope 走本地 | 3.3 / 3.3.1 |
| **用户原话流向搜索 API** | 不可逆隐私泄露 | 搜索仅离线用；导入方向约束 + 架构测试 | 7.7.1 |
| **未审核网络内容进技术库** | 互联网决定怎么问用户 | `reviewed_by` 非空 + 专家审核 + `sources` 可核验 | 7.7.3 |
| **tiktoken 对 DeepSeek 计数不准** | 预算失真致意外截断 | 官方 tokenizer 优先；估算模式加安全系数并标记 | 6.3 |
| **OpenAlex 配额耗尽** | 知识库构建中断 | 本地缓存 + 断点续跑 + 大页请求 + 优雅退出 | 7.7.2 |
| **危机资源文件缺失仍启动** | 危机时无资源可给 | 启动自检失败即拒绝启动 | 13.2.2 |
| **对话中泄露的 key 未轮换** | 凭据被滥用 | 部署前全部轮换 + `.env` 入 gitignore | 2.7 |
| **本地 reranker 阻塞事件循环** | 吞吐崩塌 | executor/独立进程 + 启动预热 + 超时降级 | 7.4 |

---

## 16. 给下游编码工具的总纲

### 16.1 生成顺序（每步产出可运行、可测试的增量）

0. **先跑 §17 探测清单**，把实测能力写进 `config/providers.yaml` —— 后续所有步骤依赖它
1. `app/schemas/` + `app/domain/` + 单测 —— **零 LLM 依赖，先立契约**
2. `app/safety/lexicon.py` + `policy.py` + 红队用例 —— **安全先行**
3. `app/llm/providers/fake.py`（FakeLLM）+ `app/graph/` 状态机 + 集成测试 —— **流程先跑通**
4. `app/llm/capability.py`（档位与门控）+ `gateway.py` + `providers/openai_compat.py`
   （超时/重试/熔断/降级/计量/参数裁剪）+ 故障注入测试 —— **门控与 Gateway 同批交付**
5. `app/agents/` + `prompts/`（外置）+ **JSON 模式结构化输出** + `verify_evidence()`
   + **报告分段生成**（§6.5）
6. `app/context/`（预算与压缩，含输出预留）+ `app/safety/output_guard.py` + `injection.py`
7. `app/storage/`（迁移 + repositories）+ 同意与删除流程（含级联删除）
8. `app/api/`（路由 + SSE + 错误码 + 锁/幂等/限流）
9. `app/observability/` 全量接入 + Sentry 脱敏钩子
10. `app/embeddings/`（按 scope 分流 + 降维）+ `app/rag/`（含 `vector_store.py`）+ `app/memory/`
11. `app/llm/providers/local_minimind.py` + `local` 档 CI 冒烟 —— **本地模型接入放在链路稳定之后**
12. `app/kb/`（离线构建管线，§7.7）+ 专家审核流程
13. `app/tasks/`（报告异步 + 索引重建）+ `app/mcp_server/`
14. `eval/` 评估集与回归门禁 + **Reranker 消融决定是否保留它**

### 16.2 贯穿全程的硬约束（下游工具不得违反）

- 所有 IO 为 `async`；纯逻辑为同步纯函数；CPU 密集任务隔离到 executor。
- **业务代码禁止直接调用 LLM SDK**，一律经 `app/llm/gateway.py`。
- **模型名/供应商/endpoint 不出现在业务代码**，只在配置与 provider adapter 中。
- 跨 Agent / 跨节点数据一律 Pydantic 模型，**禁止裸 dict 与自由文本**。
- 任何 LLM 输出进入下一步前必须经 schema 校验。
- **提示词不硬编码在 Python 中**，全部走 `prompts/` + `PromptRegistry`。
- **危机资源信息不由模型生成**，只从 `config/crisis_resources.yaml` 读。
- **token 计数必须用目标模型 tokenizer**，禁止字符数估算。
- 用户原话不进普通日志与 Sentry。
- 任何降级路径都必须有对应测试用例。
- 类型注解完整，通过 `mypy strict`；格式与 lint 用 `ruff`。
- 删除操作必须级联到向量库与缓存，且有测试覆盖。

**v3.0 新增硬约束（与多 provider / 本地模型直接相关）：**

- **任务路由必须过能力门控**（§2.4.3）；`TOY` 档承担 `STRONG` 任务是**无条件断言失败**，
  不读配置、不可覆盖。
- **请求参数按 `ProviderCaps` 裁剪**：不支持的 `temperature` / `response_format` 一律不传。
- **结构化输出以 JSON 模式为主路径**，function calling 仅在探测确认后启用（§6.2）。
- **报告与规划分段生成**，禁止一次性生成整份 `InsightReport`（§6.5）。
- **`app/kb/` 禁止 import 会话相关模块**（`graph`/`agents`/session schema），
  用架构测试强制 —— 这是"搜索 API 绝不接触用户原话"的代码级保障（§7.7.1）。
- **嵌入模型 / 维度 / 投影版本任一变化 → 全量重建索引**，禁止混用（§7.2）。
- **启动自检不通过就不启动**（§13.2.2），尤其危机资源文件缺失。
- **切换任何 provider 都不得修改业务代码**：只改 `config/` 与新增 adapter。
- **不在文档、代码、注释中写入真实 API key**。

### 16.3 依赖清单

**项目现有（与 IDE 配置一致，保留）：**
```
langchain, langchain-core, langchain-openai, langchain-text-splitters,
langgraph, langgraph-checkpoint-sqlite, openai, chromadb, langchain-chroma,
tavily-python, fastapi, uvicorn, pydantic, pydantic-settings, python-dotenv,
requests, httpx, loguru, pytest, pytest-asyncio, mcp
```

**必须补充（P0-P3）：**
```
tiktoken                    # token 计数（对 DeepSeek 为估算，见 §6.3）
tenacity                    # 重试策略
redis[hiredis]              # 锁 / 幂等 / 限流 / 热态
arq                         # 轻量异步任务队列（报告生成）
sqlalchemy[asyncio] + alembic             # ORM 与迁移
aiosqlite                   # 开发期存储
asyncpg                     # 生产 Postgres（P5，或提前接入）
rank-bm25                   # 关键词检索（进程内）
jieba                       # 中文分词（BM25 前置，必需，见 §2.2）
prometheus-client           # 指标暴露
sentry-sdk                  # 错误追踪（须配 before_send 脱敏）
opentelemetry-sdk + opentelemetry-instrumentation-fastapi
respx                       # httpx mock（测试）
ruff, mypy                  # 质量门禁
pyyaml                      # 配置（crisis_resources / task_models / providers）
```

**本地模型相关（按需，`hybrid`/`local` 档才需要）：**
```
FlagEmbedding               # BGE-M3 嵌入 + bge-reranker 本地推理
sentence-transformers       # FlagEmbedding 的替代/依赖路径
torch                       # 上述二者的运行时（体积大，仅本地档安装）
numpy                       # 降维投影（§2.6 方案 1）
```

> **依赖分组建议**：用 `pyproject.toml` 的 optional-dependencies 分组
> （`[cloud]` / `[local]` / `[dev]`），
> 让 `cloud` 档部署不必装 `torch` —— 否则镜像体积会膨胀数 GB
> 而其中大部分代码永远不会执行。

**演进期可选（不在 P0-P3 引入）：**
```
pgvector                    # Python 绑定，迁 pgvector 时（注意 §2.6 维度上限）
qdrant-client               # 规模化替换（>100 万块）
langsmith 或 langfuse       # LLM 专项调试
locust                      # 负载测试
```

**已有但需重新定位：**
- `chromadb` / `langchain-chroma`：**升级为 P0-P3 主向量库**（决策 #18），不再是过渡方案。
- `openai`：作为 **OpenAI 兼容协议客户端**使用（DeepSeek 与本地模型共用），
  不代表使用 OpenAI 服务。
- `tavily-python`：**仅**用于 `app/kb/` 离线管线（§7.7），不进请求路径。
- `langchain-openai`：可选；若 `LLMGateway` 直接用 `openai` SDK 则**不需要**，
  少一层封装更利于精细控制超时与参数裁剪。

**明确不引入：** `kafka` / `rabbitmq`（决策 #3）· `kubernetes` 客户端（P5 前）·
Serverless 相关（决策 #12）· 任何把 provider 细节泄露到业务层的高阶封装。

### 16.4 阅读顺序建议

生成代码前，下游工具应按此顺序理解文档：
**§0 + §0.1 决策表 → §3 安全 → §2.4-2.7 模型与供应商约束 → §5 数据模型 → §4 架构
→ §16 总纲 → 其余按需**。

§0 / §0.1 / §16.2 是**约束**，其余是**说明**。
若生成过程中发现约束与常见实践冲突，**以本文档约束为准**——
每条约束背后都有本产品特定的理由（多数与安全或方法论完整性相关），
不是疏忽或过时的写法。

---

## 17. 供应商能力探测清单（v3.0 新增）— MUST 在 P1 前执行

本文档多处标注"待实测"。**这不是含糊，而是刻意**：
模型标识与能力会变，把假设写死在文档里比留下探测步骤更危险。
本节给出一次性探测脚本要回答的问题，**结果写入 `config/providers.yaml` 并纳入版本控制**。

### 17.1 LLM 探测项

| # | 探测项 | 方法 | 影响 |
|---|---|---|---|
| 1 | 可用模型标识 | `GET /models` 或最小 chat 请求 | 强/快档映射（§2.4） |
| 2 | 实际上下文窗口 | 递增长度请求至报错 | `ContextBudgeter` 预算（§6.3） |
| 3 | 实际输出上限 | 请求超长输出观察截断点 | **报告分段粒度（§6.5）** |
| 4 | 是否支持 `response_format=json_object` | 带该参数请求 | **结构化输出主路径（§6.2）** |
| 5 | 是否支持 function calling | 带 tools 请求 | 是否启用路径 2 |
| 6 | 是否接受 `temperature` | 带该参数请求，看是否报错或被忽略 | **参数裁剪（§9.2 第 8 条）** |
| 7 | 是否为推理型（输出前有思维链） | 观察响应结构与首 token 延迟 | 超时分级、输出预算 |
| 8 | 流式是否可用 | `stream=True` | SSE 可行性（§10.2） |
| 9 | 中文提问质量 | 跑 §7.5-C 提问质量评分集（10 例足够） | **档位判定（`STRONG` 与否）** |
| 10 | JSON 稳定性 | 同一嵌套 schema 重复 20 次，统计解析成功率 | 若 < 95% 则不可承担 `STRONG` |
| 11 | 实际计费与 token 计数 | 对比响应 `usage` 与本地估算 | 成本模型校准（§14.1） |
| 12 | 限流阈值（RPM/TPM） | 递增并发观察 429 | 令牌桶配置（§9.2 第 5 条） |
| 13 | 长文本延迟 P95 | 典型分析请求跑 20 次 | 超时内层值（§9.2 第 1 条） |

**第 9、10 项是档位判定的决定性依据**，不能靠模型名猜。
一个模型即使上下文很长，若 JSON 稳定性不足 95%，也不该承担 `STRONG` 任务 ——
因为 §6.5 的分段生成会把它的不稳定性放大 3-6 倍。

### 17.2 Embedding 探测项

| # | 探测项 | 方法 | 影响 |
|---|---|---|---|
| 1 | 原生维度确认 | 单条请求看返回长度 | 确认 2048（用户已实测） |
| 2 | **是否支持 `dimensions` 参数降维** | 带 `dimensions=1024` 请求 | **决定 §2.6 走方案 1 还是方案 2** |
| 3 | 批量上限与 `EMBED_BATCH_SIZE` 匹配 | 递增批量至报错 | 索引构建吞吐 |
| 4 | 长文本截断行为 | 超长输入观察是否报错或静默截断 | **静默截断需在分块时规避** |
| 5 | 归一化状态 | 计算返回向量的模长 | 相似度计算方式（余弦/内积） |
| 6 | 中文检索质量 | 跑 §7.5-A 检索评估集 | 与本地 BGE-M3 对比，决定 `hybrid` 分流 |
| 7 | 限流与延迟 | 并发探测 | 在线查询可行性 |

**第 2 项直接决定架构走向**：若支持 `dimensions=1024`，
则 §2.6 方案 1 成立、pgvector 路线保留、云本地完全同维；
若不支持，则要么自建降维投影（需验证 Recall 降幅 < 3%），要么锁定 Chroma + 2048 维。
**这一项应最先探测。**

**第 4 项容易被忽略但影响 Evidence 校验**：
若嵌入服务静默截断超长输入，那么长回答的后半部分实际未被索引，
而系统以为它已被索引 —— 跨会话检索会莫名其妙地漏掉内容。

### 17.3 本地模型探测项

对 minimind 与候选的 Ollama 模型各跑一遍：

| # | 探测项 | 判定标准 |
|---|---|---|
| 1 | 是否提供 OpenAI 兼容端点 | 决定复用 adapter 还是写包装（§2.5） |
| 2 | 上下文长度 | minimind 预期 512-1024，直接判定为 `TOY` |
| 3 | 嵌套 JSON 生成成功率（20 次） | < 95% → 不得承担 `FAST`；< 50% → 仅 `TOY` |
| 4 | 中文指令遵循（"只提问不给建议"） | 违规率 > 10% → 不得用于任何提问相关任务 |
| 5 | 推理延迟 P95 | 超过快档超时值则不可用于在线路径 |
| 6 | 并发能力 | 单实例并发数，决定是否需队列 |

**探测结论必须落盘**（`config/providers.yaml` 的 `probed_at` 字段），
并在 CI 中校验：**配置声明的 `tier` 与探测结论不符时构建失败** ——
防止有人手改配置把 minimind 提到 `STRONG` 档。

### 17.4 探测脚本要求

【生成指令】实现 `scripts/probe_providers.py`：

- 独立 CLI，读 `.env`，输出 `config/providers.yaml` + 人类可读的 markdown 报告。
- **只用少量 token**（每项探测尽量单次小请求；第 9/10 项各 10-20 次，
  这是必要开销，因为它决定档位）。
- **可重复运行**：换模型或供应商后重跑，diff 配置即知能力变化。
- 探测失败的项标 `unknown` 并**按最保守值处理**
  （如 `supports_json_mode: unknown` → 视为 `false`），
  绝不乐观假设 —— 乐观假设会在生产环境以解析失败的形式暴露。
