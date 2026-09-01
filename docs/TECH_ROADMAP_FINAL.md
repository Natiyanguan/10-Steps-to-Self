# 10-Steps-to-Self 技术路线文档（终版 v2.0）

> 一个基于「苏格拉底式反向提问」的自我探索 Agent。
> 本文档合并了两份前置技术路线（`TECH_ROADMAP.md` 与 `技术路线FromDeepSeek.txt`），
> 做了显式取舍并保留理由，**用于直接驱动下游 AI 编码工具生成工程代码**。

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
| 主模型（提问/分析） | 强指令遵循的长上下文模型 | 多供应商可切换 | 见 2.1 抽象层，**不在代码中硬编码模型名** |
| 辅助模型（安全/摘要/质量评估） | 小而快的同族模型 | 本地 Llama 3.1 8B | 成本与延迟优化，见 8.4 |
| 结构化输出 | **function calling / structured output** | — | 不用"请输出 JSON"自然语言约定 |
| 关系库 | **PostgreSQL 16**（开发可 SQLite） | — | 会话、消息、记忆、审计 |
| 向量库 | **pgvector 起步** | **Qdrant**（>100 万块或需高级过滤时） | 【决策 #4】少一个组件、事务内一致写；抽象层保证可切换 |
| 关键词检索 | **PostgreSQL 全文检索 / rank-bm25** | Elasticsearch | 中文需自定义分词，见 2.2 |
| Embedding | **BGE-M3**（中文强、支持多粒度） | text-embedding-3-large | 中文场景 BGE-M3 明显更优 |
| Reranker | **bge-reranker-v2-m3** | Cohere Rerank | 本地部署可控延迟，见 7.4 |
| 缓存 / 锁 | **Redis 7** | — | 分布式锁、幂等键、限流令牌桶、会话热态 |
| 任务队列 | **Redis + arq**（轻量） | Celery / RabbitMQ | 【决策 #3】仅报告生成等长任务用，不做全链路事件驱动 |
| 配置 | **pydantic-settings + .env** | — | 启动即校验，缺失快速失败 |
| 提示词管理 | **jinja2 外置文件 + 版本化** | LangSmith Hub | 改配置即切版本，无需部署 |
| 日志 | **loguru → JSON** | — | 结构化字段固定，隐私红线见 10.1 |
| Trace | **OpenTelemetry** | LangSmith / Langfuse | LangSmith 便于 LLM 专项调试，SHOULD |
| 指标 | **Prometheus + Grafana** | — | 吸收 DeepSeek |
| 错误追踪 | **Sentry** | — | 吸收 DeepSeek；必须配 PII 脱敏钩子 |
| 内容审核 | **供应商 Moderation API + 自建词表** | — | 吸收 DeepSeek，但词表层不可省，见 4.2 |
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
2. **Embedding 选型**：优先中文表现强的模型（BGE-M3），不要默认用英文优化模型。
3. **Token 计数偏差**：中文 token/字符比与英文差异大（约 1 字 ≈ 0.6-1.5 token，随模型而变）。
   `ContextBudgeter` **必须用目标模型自己的 tokenizer 实测**，
   不可用"字符数 ÷ 4"这类英文经验公式，否则预算估算会系统性偏低并导致意外截断。
4. **Evidence 子串匹配**：中文需归一化处理（全角/半角、中英标点、空白、繁简），
   否则用户原话校验会因标点差异误判失败，见 6.3。
5. **危机词表**：必须覆盖中文谐音、拆字、拼音缩写、火星文变体，见 4.2。

### 2.3 MCP 定位 — SHOULD

项目依赖里有 `mcp`。明确其用途，避免误用：

- **作为 MCP Server**：把「提问技术库检索」「维度覆盖检查」等**无状态能力**暴露给外部客户端。
- **作为 MCP Client**：接入外部工具（如查询本地心理援助资源目录）。
- **明确不做**：不通过 MCP 暴露用户记忆或会话状态。用户心理数据不得跨信任边界流出。

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
- 送往模型前对可识别 PII 做脱敏（见 3.5），减少出境敏感信息。

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

-- 向量块（pgvector 起步）
CREATE TABLE chunks (
  chunk_id       UUID PRIMARY KEY,
  scope          TEXT NOT NULL,              -- 'kb' | 'user'
  owner_id       UUID,                       -- scope='user' 时必填，强制过滤
  text           TEXT NOT NULL,
  embedding      vector(1024),               -- 维度随模型，切换需重建
  embed_model    TEXT NOT NULL,              -- 记录模型版本，见 7.2
  metadata       JSONB NOT NULL,
  tsv            tsvector,                   -- 中文需自定义分词，见 2.2
  created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ON chunks USING gin (tsv);
CREATE INDEX ON chunks (scope, owner_id);
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

- 一律用 **function calling / structured output** 绑定 Pydantic schema。
- 解析失败三级降级：
  1. 直接解析 → 2. 容错修复（去 markdown 围栏、修尾逗号、截断补全）
  → 3. 带错误信息重试一次（温度降 0）→ 仍失败抛 `StructuredOutputError`。
- **三次失败必须显式失败**，不允许猜测填充默认值继续跑。
  静默降级会产出"看起来正常但内容错误"的报告，比报错危险得多。

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
- token 计数**必须用目标模型自己的 tokenizer**（中文偏差大，见 2.2）。
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
- **Embedding 版本管理（两版都缺）**：
  `chunks.embed_model` 记录生成该向量的模型版本。
  换 embedding 模型 → **必须全量重建索引**，且新旧向量**绝不可混用**
  （不同模型的向量空间不可比，混用会让检索静默失效——这是最难发现的一类故障）。
  重建流程：双写新 collection → 校验指标不降 → 切流 → 下线旧索引。

### 7.3 混合检索与多路召回 — MUST

【吸收 DeepSeek 的多路召回】

```
                    ┌→ 向量检索(BGE-M3) top50 ─┐
Query 改写(HyDE/多查询)                        ├→ RRF 融合 → Reranker → top5
                    └→ BM25(中文分词) top50  ──┘
                    (可选)第二 embedding 模型 top50
```

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
   安全分类 3s / 质量评估 5s / 生成问题 15s / 摘要 20s / 分析报告 60s。
2. **选择性重试**：指数退避 + 抖动（`base=1s, factor=2, jitter=±30%, max=3`）。
   仅对 429/5xx/超时/连接错误重试；
   **4xx 参数错误与内容策略拒绝不重试**（重试必然再失败，纯浪费配额与时间）。
3. **熔断器**：滑动窗口失败率 > 50% 触发熔断，半开试探恢复。
   避免下游故障时请求堆积拖垮自己。
4. **降级链**：`主模型 → 备用模型 → 备用供应商 → 模板/缓存兜底`。
   【关键】降级必须**用户可感知**（"当前使用简化模式"），不可静默降低质量。
5. **限流适配**：本地令牌桶按 **RPM 与 TPM 双维度**预先削峰，
   而非等供应商返回 429 再退避。
6. **并发闸门**：全局 semaphore 限制在途请求数（吸收 DeepSeek 的"每模型最大并发"），
   防突发流量打爆配额。
7. **成本计量**：每次调用记录 prompt/completion tokens 与预估成本，按 session 与 user 聚合，
   支持**单会话成本上限熔断**与**单用户日配额**（防恶意刷量，见 3.7）。

### 9.3 关键设计：安全层不依赖 LLM 可用性 — MUST

**LLM 全挂时，规则层危机检测必须仍然工作。**

若安全分类器不可用且规则层未命中，系统采取**保守策略**：
不进入深度提问（因为无法评估风险），转温和模式或告知稍后重试。
**绝不在无法评估安全的情况下继续犀利追问。**

这条是两版文档里最容易被"降级到备用模型就行"的思路掩盖的：
备用模型也可能同时不可用（同供应商故障），而安全不能有单点依赖。

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

**四条硬要求：**

1. **FakeLLM 是整套测试的地基，必须最先实现。** 它让状态机测试确定、快速、免费。
   FakeLLM 需支持：预设返回序列、模拟超时、返回非法 JSON、模拟拒答。
2. 危机红队用例集 `tests/redteam/crisis_cases.jsonl` **只增不减**（回归防护）。
   每发现一个绕过案例就永久加入。
3. **级联删除必须有测试**：向量库残留是最常见的"假删除"合规缺陷。
4. CI 门禁：单元 + 契约 + 集成 + ruff + mypy strict 全绿；
   评估测试按需/定时触发（因需真实 LLM 调用）。

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
  llm/                   gateway.py / providers/ / retry.py / breaker.py / budget.py
  rag/                   retriever / fusion / reranker / calibration / indexer
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
  `cross_session_recall`、`multi_route_recall`、`llm_final_judge`。
  高风险能力必须可**一键关闭**而不需发版——出问题时这是唯一的快速止血手段。

### 13.3 演进路径（每阶段可独立交付）

| 阶段 | 目标 | 交付内容 |
|---|---|---|
| **P0 骨架** | 端到端跑通 10 问 | schemas + domain + **规则层安全** + Graph 状态机 + FakeLLM + CLI |
| **P1 可用** | 真实 LLM | LLMGateway（超时/重试/降级）+ Provider 抽象 + Interviewer/Analyst + 报告 + SQLite |
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

---

## 16. 给下游编码工具的总纲

### 16.1 生成顺序（每步产出可运行、可测试的增量）

1. `app/schemas/` + `app/domain/` + 单测 —— **零 LLM 依赖，先立契约**
2. `app/safety/lexicon.py` + `policy.py` + 红队用例 —— **安全先行**
3. `app/llm/FakeLLM` + `app/graph/` 状态机 + 集成测试 —— **流程先跑通**
4. `app/llm/gateway.py` + `providers/`（超时/重试/熔断/降级/计量）+ 故障注入测试
5. `app/agents/` + `prompts/`（外置）+ 结构化输出 + `verify_evidence()`
6. `app/context/`（预算与压缩）+ `app/safety/output_guard.py` + `injection.py`
7. `app/storage/`（迁移 + repositories）+ 同意与删除流程
8. `app/api/`（路由 + SSE + 错误码 + 锁/幂等/限流）
9. `app/observability/` 全量接入 + Sentry 脱敏钩子
10. `app/rag/` + `app/memory/`
11. `app/tasks/`（报告异步 + 索引重建）+ `app/mcp_server/`
12. `eval/` 评估集与回归门禁

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

### 16.3 依赖清单

**项目现有（与 IDE 配置一致，保留）：**
```
langchain, langchain-core, langchain-openai, langchain-text-splitters,
langgraph, langgraph-checkpoint-sqlite, openai, chromadb, langchain-chroma,
tavily-python, fastapi, uvicorn, pydantic, pydantic-settings, python-dotenv,
requests, httpx, loguru, pytest, pytest-asyncio, mcp
```

**必须补充：**
```
tiktoken                    # token 精确计数（中文偏差，见 2.2）
tenacity                    # 重试策略
redis[hiredis]              # 锁 / 幂等 / 限流 / 热态
arq                         # 轻量异步任务队列（报告生成）
asyncpg + sqlalchemy[asyncio] + alembic   # Postgres 异步与迁移
pgvector                    # 向量扩展 Python 绑定
rank-bm25                   # 关键词检索
jieba                       # 中文分词（BM25 前置，必需）
FlagEmbedding               # BGE-M3 embedding / bge-reranker 本地推理
opentelemetry-sdk + opentelemetry-instrumentation-fastapi
prometheus-client           # 指标暴露
sentry-sdk                  # 错误追踪（须配 before_send 脱敏）
respx                       # httpx mock（测试）
ruff, mypy                  # 质量门禁
```

**演进期可选（不在 P0-P3 引入）：**
```
qdrant-client               # 向量库规模化替换（>100 万块）
langsmith 或 langfuse       # LLM 专项调试
locust                      # 负载测试
```

**明确不引入：** `kafka` / `rabbitmq`（决策 #3）· `kubernetes` 客户端（P5 前）·
Serverless 相关（决策 #12）· `chromadb` 在 P3 后可移除（改用 pgvector，决策 #4）。

### 16.4 阅读顺序建议

生成代码前，下游工具应按此顺序理解文档：
**§0 决策表 → §3 安全 → §5 数据模型 → §4 架构 → §16 总纲 → 其余按需**。

§0 与 §16.2 是**约束**，其余是**说明**。
若生成过程中发现约束与常见实践冲突，**以本文档约束为准**——
每条约束背后都有本产品特定的理由（多数与安全或方法论完整性相关），
不是疏忽或过时的写法。
