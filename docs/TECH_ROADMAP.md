# 10-Steps-to-Self 技术路线文档

> 一个基于「苏格拉底式反向提问」的自我探索 Agent。
> 本文档是**工程实施蓝图**，用于驱动下游 AI 编码工具生成代码。
> 每一节都可独立作为一个生成任务的输入；节内的 `【生成指令】` 是给编码工具的显式约束。

- 版本：v1.0
- 目标运行环境：Python 3.12 / Windows 11 开发 + Linux 容器部署
- 核心栈：LangGraph · LangChain · FastAPI · Pydantic v2 · Chroma · SQLite→Postgres · MCP
- 文档约定：`MUST` 必须实现；`SHOULD` 建议实现；`LATER` 明确留到后期阶段

---

## 0. 一句话定义

系统扮演一位「精通苏格拉底提问术的生涯规划 / 心理咨询取向的引导者」，
**不给答案，而是通过 10 个递进式问题把用户潜意识里的真实诉求"接生"出来**，
最后输出一份「潜意识诉求分析 + 可执行建议」的结构化复盘报告。

产品的核心资产不是"回答能力"，而是**提问质量**与**长期记忆的连贯性**。
所有技术选型都服务于这两点。

---

## 1. 理论到工程的映射（Theory → Component）

这一节是全系统的设计源头。**每个理论都必须落成可测量的工程构件**，
否则理论只是包装。下游生成代码时以本表作为模块清单的依据。

| 理论依据 | 工程构件 | 可观测指标 |
|---|---|---|
| 苏格拉底产婆术 | `QuestionPlanner`：递进式提问策略引擎，禁止给建议的输出约束 | 提问深度递进率、"直接给答案"违规率 |
| 乔哈里视窗 | `BlindSpotProbe`：维度覆盖检查器，强制问题打到用户未提及的维度 | 四象限覆盖度、盲区维度命中数 |
| 焦点解决短期治疗（奇迹提问） | `TechniqueLibrary`：提问技术库（奇迹问句 / 例外问句 / 量尺问句 / 循环问句） | 技术使用分布、单一技术过载告警 |
| 边界对象与认知卸载 | `SessionState` + 单问单答节奏 + 进度可视化 | 单轮认知负荷（问题长度/概念数）、中途放弃率 |

### 1.1 提问技术库（TechniqueLibrary）— MUST

代码化为枚举 + 模板 + 适用条件，而非散落在提示词里的自然语言：

```python
class Technique(StrEnum):
    MIRACLE      = "miracle"        # 奇迹问句：假设问题已解决，你会注意到什么不同？
    EXCEPTION    = "exception"      # 例外问句：什么时候这个困扰没有出现？
    SCALING      = "scaling"        # 量尺问句：0-10 分你现在在几分？1 分之差是什么？
    CIRCULAR     = "circular"       # 循环问句：如果你最亲近的人来回答，他会怎么说？
    COST_OF_LOSS = "cost_of_loss"   # 代价问句：做这个决定你最怕失去什么？
    REFRAME      = "reframe"        # 重构问句：换个角度这件事在保护你什么？
    VALUE_PROBE  = "value_probe"    # 价值观追问：这件事之所以重要，是因为你在意什么？
    TIME_SHIFT   = "time_shift"     # 时间线：5 年后的你会如何评价现在的犹豫？
```

### 1.2 提问维度矩阵（覆盖乔哈里盲区）— MUST

10 个问题不是随机生成，而是在**维度矩阵**上做覆盖规划：

`事实层` → `情绪层` → `价值层` → `恐惧层` → `资源层` → `代价层` → `关系层` → `身份层`

`QuestionPlanner` 必须保证：前 3 问偏事实与情绪（建立安全感），
中间 4 问打价值 / 恐惧 / 代价（核心挖掘），后 3 问收敛到资源与行动。
**任何一个维度连续两问重复 = 规划失败，触发重规划。**

【生成指令】把 1.1 和 1.2 实现为 `app/domain/technique.py` 与 `app/domain/dimension.py`，
纯数据 + 纯函数，不依赖 LLM，必须 100% 单测覆盖。

---

## 2. 安全边界（最高优先级，先于功能实现）

这是一个触及心理状态的产品。**安全模块必须在第一个可运行版本里就存在**，
不能作为后期补丁。以下规则在架构上强制，而不只写在提示词里
（提示词可被绕过，代码分支不能）。

### 2.1 产品定位声明 — MUST

- 系统是**自我探索与思维梳理工具**，不是医疗器械，不做诊断、不做治疗、不开处方。
- 首次进入与每次会话开始，UI 层必须展示定位与局限声明，并记录用户确认。
- 任何输出中禁止出现"诊断""你患有""疗程""处方"等表述，由 `OutputGuard` 正则 + 语义双重拦截。

### 2.2 危机识别与转介（Crisis Detection & Referral）— MUST

**双层检测，串行短路：**

1. **第一层：规则前置过滤器**（`app/safety/lexicon.py`）
   - 自伤 / 自杀意念、伤害他人、严重物质滥用、未成年人受虐等关键词与变体（含拼音、谐音、拆字、空格插入）。
   - 零 LLM 依赖，延迟 < 1ms，**LLM 不可用时仍然生效**。这是最后一道防线，必须能独立工作。
2. **第二层：轻量分类器**（小模型或专用 prompt，低温度、结构化输出）
   - 输出 `risk_level ∈ {none, low, medium, high, imminent}` + 依据片段。
   - 仅在第一层未命中但语义可疑时调用，避免全量成本。

**处置矩阵（硬编码为状态机分支，不由 LLM 自由决定）：**

| risk_level | 系统行为 |
|---|---|
| none / low | 正常提问流程 |
| medium | 中断提问节奏，切换到关怀性回应，降低追问锐度，提示专业资源 |
| high / imminent | **立即终止提问流程**，输出危机资源信息（当地紧急电话、心理援助热线），不再生成任何深挖性问题 |

- 危机分支**不允许**继续"犀利提问"。产品调性在此让位于安全。
- 危机事件必须落审计日志（脱敏后），可用于后续人工复核与阈值调优。
- 热线信息**不硬编码在提示词**里，放配置文件按地区可维护（`config/crisis_resources.yaml`），避免模型幻觉出错误号码。

【生成指令】`app/safety/` 目录下实现 `lexicon.py`（规则层）、`classifier.py`（模型层）、
`policy.py`（处置矩阵）。`policy.py` 必须是纯函数 `decide(risk_level, session) -> SafetyAction`，
可完全离线单测。危机路径的单测覆盖率要求 100%。

### 2.3 提问锐度控制（Sharpness Governor）— MUST

用户要求"犀利问题"，但犀利与伤害只隔一层。引入**锐度分级**：

```python
class Sharpness(IntEnum):
    GENTLE = 1   # 温和探索
    MODERATE = 2 # 适度挑战（默认）
    SHARP = 3    # 犀利直击（需用户显式开启）
```

- 默认 `MODERATE`；`SHARP` 需用户在会话开始显式选择，并可随时下调。
- 检测到 `risk_level >= medium`、或用户情绪强度指标上升、或用户回答长度骤降（回避信号）时，
  **自动降级**锐度并记录降级原因。
- 锐度作为参数注入提示词模板，同时作为 `OutputGuard` 的校验基线。

### 2.4 越权与滥用防护 — MUST

- **禁止代入真实身份**：不扮演具名的真实心理医生、不声称持有执业资质。
- **提示词注入防护**：用户输入永远作为数据传入，用明确分隔符包裹；
  系统指令置于用户内容之前并在末尾复述关键约束（sandwich 结构）；
  对"忽略以上指令""你现在是另一个 Agent"类内容做检测并忽略。
- **PII 最小化**：不主动索取真实姓名、身份证、住址、单位全称；
  若用户主动提供，存储前脱敏（`app/safety/pii.py`），日志中一律以键名代替值。
- **输出侧兜底**（`OutputGuard`）：拦截医疗断言、绝对化承诺（"一定会好"）、
  越界建议（用药、停药、法律与投资决策）。命中则重生成，重生成仍失败则降级到安全模板回复。

---

## 3. 系统架构

### 3.1 分层视图

```
┌──────────────────────────────────────────────────────────┐
│ 接入层  FastAPI (REST + SSE 流式) · 限流 · 鉴权 · 幂等   │
├──────────────────────────────────────────────────────────┤
│ 编排层  LangGraph StateGraph（含检查点持久化 / 断点续跑）│
│   Intake → Plan → Ask → Listen → Probe → Analyze → Report│
├──────────────────────────────────────────────────────────┤
│ Agent 层                                                 │
│   Planner · Interviewer · Safety · Analyst · Critic       │
├──────────────────────────────────────────────────────────┤
│ 能力层  LLM Gateway · RAG(检索+重排) · Memory · Tools(MCP)│
├──────────────────────────────────────────────────────────┤
│ 存储层  Postgres(会话/审计) · Chroma(向量) · Redis(缓存/锁)│
├──────────────────────────────────────────────────────────┤
│ 可观测  结构化日志 · Trace · 指标 · 评估集回归            │
└──────────────────────────────────────────────────────────┘
```

### 3.2 为什么用 LangGraph 而不是纯 Chain — 设计理由

提问流程本质是**有状态、可中断、需回溯**的过程：
用户可能中途离开、答得敷衍需要追问、触发安全分支需跳转。
Chain 是线性的，表达不了这些；`StateGraph` 的显式节点 + 条件边 + Checkpointer 正好匹配。

- **可恢复**：`langgraph-checkpoint-sqlite`（开发）→ Postgres checkpointer（生产），
  用户三天后回来能接着第 6 问继续。
- **可审计**：每次状态迁移落库，出问题能重放。
- **可插拔安全边**：安全检查作为条件边而非节点内 if，任何节点后都能短路到危机分支。

### 3.3 状态机定义 — MUST

```python
class Phase(StrEnum):
    INTAKE   = "intake"    # 收集困惑描述、确认知情同意、选择锐度
    PLAN     = "plan"      # 生成 10 问的维度规划（不是问题文本本身）
    ASK      = "ask"       # 生成并发出当前问题
    LISTEN   = "listen"    # 接收回答，质量评估
    PROBE    = "probe"     # 回答敷衍/回避时的追问（不占用 10 问配额）
    ANALYZE  = "analyze"   # 潜意识诉求分析
    CRITIC   = "critic"    # 分析结果自检（幻觉与依据校验）
    REPORT   = "report"    # 输出结构化报告
    CRISIS   = "crisis"    # 安全终止分支
    DONE     = "done"
```

**关键条件边：**

- `LISTEN → PROBE`：回答质量分低于阈值且追问次数未超上限（默认每问最多 2 次追问）。
- `LISTEN → ASK`：质量合格，推进到下一问。
- `任意节点 → CRISIS`：安全检查命中 `high/imminent`。**这条边优先级最高。**
- `ANALYZE → CRITIC → ANALYZE`：自检不通过时最多重做 2 次，超限则输出带不确定性标注的报告
  （宁可承认"依据不足"，也不编造洞察）。

### 3.4 Agent 职责与协作契约 — MUST

多 Agent 的失败模式几乎都来自**职责重叠**和**隐式契约**。因此：
每个 Agent 只有一个职责，Agent 之间**只通过 Pydantic 模型通信，禁止传自由文本**。

| Agent | 唯一职责 | 输入 | 输出 | 禁止 |
|---|---|---|---|---|
| Planner | 规划 10 问的维度与技术序列 | 困惑描述 + 画像 + 检索片段 | `QuestionPlan` | 生成问题文本 |
| Interviewer | 把规划槽位渲染成一个自然的问题 | `PlanSlot` + 对话历史摘要 | `Question` | 给建议、下判断 |
| Safety | 风险判级 | 用户最新输入 | `RiskAssessment` | 修改流程（只判级，处置交 policy） |
| Analyst | 潜意识诉求分析 + 建议 | 全量问答对 | `InsightReport` | 引用未出现的内容 |
| Critic | 校验 Analyst 输出的依据 | `InsightReport` + 问答对 | `CritiqueResult` | 自己改写报告 |

**协作原则：**

- **单写者**：任一状态字段只有一个 Agent 可写（Planner 写 plan，Interviewer 写 question…），
  从根上消除并发写冲突。
- **Critic 不改写，只否决**：改写权保留给 Analyst，避免两个 Agent 互相覆盖导致的振荡。
- **无自由文本传递**：所有跨 Agent 数据结构化，避免"上一个 Agent 的措辞影响下一个 Agent 的判断"这类隐性耦合。

【生成指令】编排代码放 `app/graph/`，一个节点一个文件；
Agent 放 `app/agents/`；所有跨节点数据模型放 `app/schemas/`。
节点函数签名统一为 `async def node(state: SessionState) -> dict`，只返回增量更新字段。

---

## 4. 数据模型（Pydantic v2）— MUST

模型即契约。下游工具应先生成本节全部模型并通过类型检查，再实现逻辑。

```python
class PlanSlot(BaseModel):
    index: int = Field(ge=1, le=10)
    dimension: Dimension
    technique: Technique
    intent: str                    # 这一问要挖什么（给 Interviewer 的内部说明）
    depends_on: list[int] = []     # 依赖前面哪些问的回答

class QuestionPlan(BaseModel):
    slots: list[PlanSlot]
    rationale: str
    @model_validator(mode="after")
    def check_coverage(self):      # 维度覆盖 + 无连续重复，校验失败即重规划
        ...

class Question(BaseModel):
    index: int
    text: str = Field(max_length=300)   # 硬约束：单问不超长，防认知过载
    technique: Technique
    sharpness: Sharpness

class Answer(BaseModel):
    question_index: int
    text: str
    quality: AnswerQuality | None = None

class AnswerQuality(BaseModel):
    score: float = Field(ge=0, le=1)
    is_evasive: bool               # 回避信号
    is_too_brief: bool
    missing_aspects: list[str]     # 供 PROBE 节点生成追问

class RiskAssessment(BaseModel):
    level: RiskLevel
    matched_rules: list[str] = []
    evidence: str | None = None
    source: Literal["lexicon", "classifier"]

class Evidence(BaseModel):
    """报告中每条洞察必须挂依据，这是反幻觉的结构性约束。"""
    question_index: int
    quote: str                     # 必须是用户原话的子串
    @field_validator("quote")
    def must_be_verbatim(cls, v, info):
        ...                        # 运行期再用 verify_evidence() 做全量校验

class Insight(BaseModel):
    statement: str
    evidence: list[Evidence] = Field(min_length=1)   # 无依据不许成条
    confidence: Literal["low", "medium", "high"]

class Suggestion(BaseModel):
    action: str
    first_step: str                # 必须可在 24h 内启动，防空泛建议
    linked_insights: list[int]

class InsightReport(BaseModel):
    surface_concern: str           # 用户自述的表层困惑
    underlying_needs: list[Insight]  # 潜意识诉求
    inner_conflicts: list[Insight]   # 内在冲突
    suggestions: list[Suggestion]
    uncertainty_notes: list[str] = []  # 明确写出"依据不足之处"

class SessionState(BaseModel):
    session_id: str
    user_id: str
    phase: Phase
    sharpness: Sharpness
    concern: str
    plan: QuestionPlan | None
    qa_pairs: list[tuple[Question, Answer]]
    probe_counts: dict[int, int]
    risk_history: list[RiskAssessment]
    report: InsightReport | None
    consent_at: datetime | None
    revision: int                  # 乐观锁版本号
```

**设计要点：**

- `Evidence.quote` 必须是用户原话子串 —— 把"不许编造"从提示词约束升级为**可程序化验证的约束**。
  这是全系统最重要的一条反幻觉设计。
- `Suggestion.first_step` 强制 24h 可启动，逼迫模型输出具体动作而非"多沟通""调整心态"。
- `uncertainty_notes` 是**允许模型说不知道的出口**。没有这个字段，模型只会编。
- `revision` 用于并发控制（见第 8 节）。

---

## 5. 提示词工程与上下文预算

### 5.1 提示词版本化与外置 — MUST

- 提示词**不写在 Python 字符串里**，放 `prompts/{agent}/{version}.jinja2`，
  以文件为单位版本化（`v1`, `v2`…），运行时由 `PromptRegistry` 按配置加载。
- 每个提示词文件头部注明：适用 Agent、期望输出 schema、变更原因。
- 切换提示词版本 = 改配置，**不需要改代码、不需要重新部署**，便于 A/B 与快速回滚。
- 所有提示词渲染后记录 hash 到 trace，便于问题复现时定位是哪一版。

### 5.2 结构化输出（对抗格式漂移）— MUST

- 一律使用 **function calling / structured output** 绑定 Pydantic schema，
  不用"请输出 JSON"这种自然语言约定。
- 解析失败的三级降级：
  1. 直接解析 → 2. 容错修复（去 markdown 围栏、修尾逗号）→ 3. 带错误信息重试一次（温度降到 0）
  → 仍失败则抛 `StructuredOutputError`，由上层决定降级策略。
- 三次失败必须**显式失败**，不允许猜测填充默认值后继续跑
  —— 静默降级会产出看起来正常但内容错误的报告，比报错危险得多。

### 5.3 上下文预算与截断策略 — MUST

这是长会话（10 问 + 追问，可能 30+ 轮）的核心工程问题。
**绝不能靠"祈祷不超长"，必须有硬预算。**

```
总预算 = min(模型上下文, 配置上限)
分配：系统指令 15% | 会话摘要 20% | 近期原文 35% | 检索片段 20% | 输出预留 10%
```

**分层记忆压缩（Hierarchical Summarization）：**

1. **近期窗口**：最近 3 轮问答保留**完整原文**（不压缩，因为追问依赖细节）。
2. **中期摘要**：第 4 轮之前的问答，按"维度 + 关键事实 + 情绪信号 + 用户原话金句"结构化摘要。
   金句必须原样保留 —— 后面 `Evidence` 校验要用到原文。
3. **长期画像**：跨会话的稳定特征（价值观倾向、反复出现的主题、已尝试过的方案）。

**实施要点：**

- 进入每个 LLM 调用前，`ContextBudgeter.fit()` 用 tiktoken 精确计数并按优先级裁剪，
  **超预算时先砍检索片段，再砍中期摘要，永不砍系统指令和安全约束**。
- 摘要过程本身要防信息丢失：摘要器输出必须包含 `preserved_quotes` 字段，
  且原文中的用户金句在压缩后仍可被 `Evidence` 校验命中。
- 结构化摘要（而非自由文本摘要）便于选择性加载：
  Analyst 需要全部维度，Interviewer 只需要相邻维度。

【生成指令】实现 `app/context/budgeter.py` 与 `app/context/summarizer.py`。
`budgeter.fit()` 必须是纯函数且可单测（输入 token 预算与片段列表，输出裁剪后列表），
不允许在裁剪逻辑里调用 LLM。

### 5.4 幻觉治理（分层防御）— MUST

单一手段防不住幻觉，需要**四层叠加**：

| 层级 | 手段 | 拦截对象 |
|---|---|---|
| 生成前 | 检索增强 + 明确"仅基于用户所说"指令 + 低温度（分析阶段 ≤0.3） | 无依据推断 |
| 生成时 | 结构化输出强制每条 Insight 挂 Evidence | 空口洞察 |
| 生成后 | `verify_evidence()` 程序化校验 quote 是否为用户原话子串（归一化后模糊匹配，阈值 0.9） | 伪造引用 |
| 复核 | Critic Agent 独立复核依据与结论的逻辑关联 | 引用真实但推论跳跃 |

- **第三层是最硬的**：不依赖任何模型判断，纯字符串校验。
  校验不通过的 Insight **直接丢弃**，而不是保留待人工判断。
- Critic 使用**独立提示词与更低温度**，输入只给"报告 + 问答对"，
  不给 Analyst 的推理过程，避免被其思路带偏（防止 anchoring）。
- 分析阶段禁止模型引入外部心理学结论去"套"用户，
  凡引用理论必须标注为"参考框架"而非"你的情况就是这样"。

---

## 6. RAG 设计

### 6.1 检索什么 —— 先明确边界

本系统 RAG **不用于回答用户问题**，只用于两个目的：

1. **提问技术库检索**：给 Planner 提供适用于当前情境的提问范式与案例。
2. **用户历史检索**：跨会话召回该用户过去的相关表述（"你三个月前说过…"）。

**明确不做**：检索心理学文献去给用户下结论。这既是安全边界，也避免"看似权威实则套用"的输出。
知识库内容以**提问范式**为主，而非诊断标准。

### 6.2 索引设计 — MUST

- **两个独立 collection**，绝不混在一起（避免用户隐私数据与公共知识库串味）：
  - `kb_techniques`：公共，提问技术与案例，按技术类型 + 适用情境打元数据。
  - `mem_{user_id}`：私有，每用户物理隔离（Chroma 按 collection 隔离 + 应用层强制 user_id 过滤，**双重保险**）。
- 分块策略：技术库按"一个技术 + 一个示例"为一块（语义完整），
  不用固定长度切分 —— 提问范式被切断就失去了参考价值。
- 元数据必须含 `source_type`、`created_at`、`session_id`、`dimension`，支撑过滤与时间衰减。

### 6.3 混合检索 + 重排 — MUST

```
Query 改写(HyDE/多查询) → 向量检索 top50 ─┐
                                          ├→ RRF 融合 → Reranker → top5
                        BM25 检索 top50 ─┘
```

- 向量单打独斗在"用户口语化表述"上召回差，必须配 BM25 补齐关键词匹配。
- 融合用 **RRF（倒数排名融合）**，只依赖排名不依赖分数，天然规避两路分数不可比的问题。

### 6.4 Reranker 分数过近 / 不可靠 —— 专项方案

这是你特别提到的问题，也是 RAG 落地最常见的坑。**分数绝对值不可信，需要用"分布特征"而不是"阈值"来决策：**

1. **分数归一化后看差距，而非看绝对值**
   计算 top1 与 top2 的相对差距 `gap = (s1 - s2) / (s1 + ε)`。
   `gap` 小于阈值（如 0.05）说明模型无法区分，此时**不要强行选 top1**。

2. **低区分度时的三种处置**（按场景选，配置化）
   - **全都要**：把分数接近的前 N 条一起送入上下文（提问场景容忍多个参考范式，这是首选）。
   - **换维度决策**：用元数据做二次排序（时间新近度、技术使用频率均衡性、维度匹配度）。
   - **弃用检索**：若 top1 分数低于绝对下限，说明知识库无相关内容，
     **直接走无检索路径**，而不是塞入不相关片段污染上下文。这一条最重要 —— 宁可不给参考，不给错参考。

3. **校准（Calibration）**
   - 用标注集在离线阶段拟合分数到"相关概率"的映射（isotonic regression 或 Platt scaling），
     线上用校准后的概率做阈值判断，让阈值有可解释含义。
   - 定期重新校准，模型或知识库更新后必须重跑。

4. **不确定性显式传递**
   检索结果携带 `confidence` 与 `is_ambiguous` 标记进入提示词，
   低置信时提示词明确写"以下参考可能不适用，请优先基于用户原话"。
   **让下游知道上游不确定，而不是假装确定。**

5. **Reranker 失败降级链**
   `Cross-encoder Reranker → RRF 排名 → 纯向量 top-k → 无检索`
   任一环节超时（> 800ms）或异常，自动降到下一级并记录降级事件。
   **降级是常态而非异常**，指标里要能看到降级率。

【生成指令】实现 `app/rag/retriever.py`、`app/rag/fusion.py`、`app/rag/reranker.py`、
`app/rag/calibration.py`。Reranker 必须封装在带超时和降级链的 wrapper 里，
且 `decide_from_scores()` 为纯函数以便单测各种分数分布（含全部接近、全部偏低、单峰突出三类边界）。

### 6.5 RAG 效果评估方法 — MUST

**不建评估集就等于没有 RAG 质量概念。** 分三层评估：

**A. 检索层（无需 LLM，成本低，可高频跑）**
- 构建 50-100 条标注查询集（query → 相关文档 id）。
- 指标：`Recall@k`、`MRR`、`nDCG@k`、`Hit Rate`。
- 消融对比：纯向量 / 纯 BM25 / RRF / RRF+Rerank，量化每一层的增益。
  **如果 Reranker 增益 < 2%，就应该砍掉它**（省延迟省钱），用数据做决策。

**B. 生成层（RAGAS 式，需 LLM 评判）**
- `Faithfulness`：回答是否忠于检索内容（本系统更关键的是"是否忠于用户原话"）。
- `Answer Relevancy`、`Context Precision`、`Context Recall`。
- LLM-as-judge 必须：固定评判模型与版本、固定提示词、**用同一批人工标注做校准**，
  否则评分漂移会让指标失去可比性。

**C. 端到端产品层（本系统特有，最重要）**
通用 RAG 指标衡量不了"提问质量"，需要自定义评估集：
- **提问质量评分集**：20-30 个典型困惑场景，人工评分维度：
  深度（是否触及价值/恐惧层）、非引导性（是否暗含预设答案）、
  自然度、锐度适配、**是否违规给建议**。
- **维度覆盖率**：10 问是否覆盖预设维度矩阵（可程序化计算，零成本）。
- **依据校验通过率**：Evidence 程序化校验的通过比例（直接反映幻觉水平）。
- **回归门禁**：提示词或模型变更后必须跑评估集，
  关键指标下降超过阈值则阻断上线。

【生成指令】评估代码放 `eval/`，与 `app/` 同级。
必须提供 `eval/run.py` 一键跑全部评估并输出对比报告（markdown + json）。
评估集数据放 `eval/datasets/`，用 jsonl 格式，纳入版本控制。

---

## 7. 记忆系统与记忆冲突

### 7.1 四层记忆 — MUST

| 层 | 内容 | 存储 | 生命周期 |
|---|---|---|---|
| 工作记忆 | 当前会话近 3 轮原文 | LangGraph State | 会话内 |
| 会话记忆 | 本次会话结构化摘要 | Postgres | 永久（可删） |
| 长期画像 | 稳定特征、价值观倾向、反复主题 | Postgres（结构化字段） | 永久，可版本化 |
| 语义记忆 | 可检索的历史表述片段 | Chroma `mem_{user_id}` | 永久，带时间衰减 |

**长期画像用结构化字段而非自由文本**，因为自由文本画像会越写越长、无法校验、且冲突时无法定位。

### 7.2 记忆冲突 —— 专项方案

用户的想法**本来就会变**（三个月前想离职，现在想留下）。
这不是数据错误，而是**真实的时间演化**。因此核心原则：

> **不覆盖、不删除、不"纠正"用户 —— 记忆是带时间戳的观察序列，不是唯一真值。**

**冲突检测：**
- 写入新记忆条目时，检索同 `topic` 的既有条目，用 NLI（蕴含/矛盾判断）或结构化字段比对识别矛盾。
- 冲突类型分三种，处置不同：
  1. **时间演化型**（想法确实变了）→ 两条都保留，新条目标 `supersedes`，旧条目标 `superseded_by`，
     **检索时默认取最新，但矛盾本身是宝贵的提问素材**：
     "你三个月前提到 X，现在倾向 Y，这个变化里发生了什么？" —— 这恰好是绝佳的苏格拉底式问题。
  2. **矛盾并存型**（用户同时持有冲突信念，如"想要自由"+"想要稳定"）→ 两条都保留并标记为
     `inner_conflict`，这是 Analyst 分析内在冲突的核心输入，**绝不可消解**。
  3. **提取错误型**（系统抽取错了）→ 标记 `invalidated`，软删除保留审计痕迹。
     判定为提取错误需要证据（用户显式否认），不能由系统自行判断。

**实施要点：**
- 记忆条目 schema 必须含：`asserted_at`、`confidence`、`source_session_id`、`status`、`supersedes`。
- 检索时按 `recency × confidence × relevance` 三因子加权，**不做硬性覆盖**。
- 提供用户可见的记忆管理入口：**可查看、可更正、可删除**（GDPR 式权利，也是信任基础）。
- 跨会话引用历史前**必须先确认**："我记得你之前提到过 X，还是这样吗？"
  —— 直接断言记忆内容是体验灾难，记错了会摧毁信任。

【生成指令】实现 `app/memory/store.py`（CRUD + 状态机）、`app/memory/conflict.py`（检测与分类）、
`app/memory/retrieval.py`（三因子加权）。`conflict.classify()` 必须可单测，
测试用例需覆盖上述三种冲突类型。

---

## 8. 并发、可靠性与延迟

### 8.1 并发模型 — MUST

- **全链路 async**：FastAPI + httpx.AsyncClient + async LLM client。
  LLM 调用是 IO 密集，同步阻塞会让并发能力直接崩掉。
- **CPU 密集任务隔离**：tokenizer 计数、embedding 本地推理、BM25 建索引
  放 `run_in_executor` 或独立进程池，避免阻塞事件循环。
- **每用户会话串行化**：同一 `session_id` 的请求必须串行处理。
  用 Redis 分布式锁（`SET NX PX` + 唯一 token 释放，防误删他人锁）。
  并发提交同一会话的两个回答会导致状态错乱，这是必须堵住的。
- **乐观锁兜底**：`SessionState.revision` 每次写入 +1，
  `UPDATE ... WHERE revision = :expected`，影响行数为 0 则冲突，返回 409 让客户端重取。
  锁 + 版本号双保险，锁失效（如 Redis 抖动）时版本号仍能防脏写。
- **幂等**：写接口接受 `Idempotency-Key` 头，Redis 记录键 → 结果，TTL 24h。
  网络重试不能造成重复推进问题序号。

### 8.2 LLM 调用可靠性 — MUST

集中封装 `app/llm/gateway.py`，**业务代码禁止直接调 SDK**。Gateway 负责：

1. **超时分级**：安全分类 3s / 生成问题 15s / 分析报告 60s。
   不同任务用同一超时值是常见错误。
2. **重试**：指数退避 + 抖动（`base=1s, factor=2, jitter=±30%, max_attempts=3`）。
   **仅对可重试错误重试**（429、5xx、超时、连接错误）；
   4xx 参数错误、内容策略拒绝**不重试**（重试必然再失败，纯浪费）。
3. **熔断器**：滑动窗口内失败率 > 50% 触发熔断，半开状态试探恢复。
   避免下游故障时请求堆积拖垮自己。
4. **模型降级链**：`主模型 → 备用模型 → 备用供应商 → 缓存/模板回复`。
   降级必须**用户可感知**（"当前使用简化模式"），不可静默降低质量。
5. **限流适配**：本地令牌桶预先削峰（按 RPM 与 TPM 双维度），
   而不是等供应商返回 429 再退避。
6. **并发闸门**：全局 semaphore 限制在途 LLM 请求数，防止突发流量打爆配额。
7. **成本与 token 计量**：每次调用记录 prompt/completion tokens 与预估成本，
   按 session 聚合，支持单会话成本上限熔断（防恶意刷量）。

### 8.3 关键设计：安全层不依赖 LLM 可用性

**LLM 全挂时，规则层危机检测必须仍然工作。**
若安全分类器不可用且规则层未命中，系统采取**保守策略**：
不进入深度提问（因为无法评估风险），转为温和模式或告知稍后重试。
**绝不在无法评估安全的情况下继续犀利追问。**

### 8.4 延迟优化 — SHOULD

| 手段 | 说明 |
|---|---|
| SSE 流式输出 | 首 token 时间是感知延迟的关键；问题文本边生成边推送 |
| 并行化 | 安全检查与检索并行（`asyncio.gather`），二者无依赖 |
| Prompt 缓存 | 系统指令与技术库前缀放提示词头部保持稳定，命中供应商前缀缓存 |
| 预生成 | 用户回答时后台预取下一问所需检索片段（投机执行，答完即用） |
| 小模型分流 | 安全分类、回答质量评估、摘要用小模型；只有提问与分析用强模型 |
| 语义缓存 | 谨慎使用：仅缓存技术库检索结果，**绝不缓存针对用户的问题与分析** |
| 报告异步生成 | 最终报告 60s 级任务转后台任务 + 完成通知，避免请求挂死 |

**分析报告场景明确接受更高延迟**：宁可 30 秒给出有依据的分析，
不要 3 秒给出泛泛而谈。产品上用进度反馈掩盖延迟，而非牺牲质量。

### 8.5 失败可见性 — MUST

- 每个降级、重试、熔断、截断事件都要打结构化日志并计入指标。
- **静默失败是最危险的**：报告少了一半依据但接口返回 200，
  用户会当成真实结论使用。任何影响输出质量的降级都必须在响应里带 `degraded: true` 与原因。

---

## 9. 可观测性

### 9.1 结构化日志 — MUST

- loguru + JSON sink，字段固定：`ts`、`level`、`trace_id`、`session_id`、`user_id_hash`、
  `node`、`agent`、`event`、`duration_ms`、`tokens`、`degraded`、`prompt_version`。
- **隐私红线**：用户原话**不进普通日志**。
  仅在显式开启的调试模式且用户授权下记录，且必须过 PII 脱敏。
  `user_id` 一律哈希后记录。
- 危机事件走**独立审计日志**（append-only，单独权限），保留用于合规与人工复核。

### 9.2 Trace — MUST

- OpenTelemetry span 覆盖：每个 graph 节点、每次 LLM 调用、每次检索、每次重排。
- span 属性含模型名、提示词版本 hash、token 数、是否降级、检索命中数与分数分布。
- 一次会话的完整 trace 必须可回放，这是排查"为什么问了个奇怪的问题"的唯一途径。

### 9.3 核心指标 — MUST

**技术指标**：P50/P95/P99 延迟（分节点）、LLM 错误率与重试率、熔断次数、
降级率（按类型）、token 消耗与成本（按会话/用户）、上下文截断触发率。

**质量指标**：维度覆盖率、Evidence 校验通过率、Critic 否决率、
追问触发率、结构化输出解析失败率、Reranker 低区分度触发率。

**产品指标**：会话完成率（走完 10 问的比例）、中途放弃点分布（第几问流失最多）、
报告有用性评分、跨会话回访率。

**安全指标**：各风险等级触发次数、锐度自动降级次数、OutputGuard 拦截次数与类型。

> 「第几问流失最多」是最有价值的产品指标 —— 它直接指出哪一问设计得太重或太突兀。

---

## 10. 测试策略

| 层级 | 范围 | 要点 |
|---|---|---|
| 单元测试 | 纯函数与领域逻辑 | technique/dimension/budgeter/conflict/policy/scores 决策，**全部零 LLM 依赖，必须 100% 覆盖** |
| 契约测试 | Pydantic 模型 | 边界值、校验器、序列化往返 |
| 集成测试 | Graph 流转 | 用 **FakeLLM**（可编程返回值）跑完整状态机，含危机分支、追问分支、Critic 否决重做 |
| 评估测试 | 提示词质量 | 见 6.5，跑评估集，作为上线门禁 |
| 混沌测试 | 故障注入 | LLM 超时/429/返回非法 JSON、Chroma 不可用、Redis 宕机、Reranker 超时，验证降级链 |
| 安全红队 | 对抗测试 | 提示词注入、危机词变体（谐音/拆字/空格）、诱导越界建议、诱导医疗诊断 |

**关键要求：**
- **FakeLLM 是整套测试的地基**，必须先实现。它让状态机测试确定、快速、免费。
- 危机检测的红队用例集单独维护（`tests/redteam/crisis_cases.jsonl`），
  每次发现绕过案例就加进去，**只增不减**（回归防护）。
- CI 中单测与集成测试必须全绿；评估测试因需调用 LLM，按需/定时触发。

---

## 11. 部署与工程规范

### 11.1 目录结构 — MUST

```
app/
  main.py              FastAPI 入口
  config.py            pydantic-settings 配置
  api/                 路由 / 依赖 / 中间件（限流、鉴权、幂等、trace）
  graph/               LangGraph 节点与图装配
  agents/              Planner / Interviewer / Safety / Analyst / Critic
  domain/              technique.py / dimension.py（纯逻辑）
  schemas/             Pydantic 模型
  llm/                 gateway.py / retry.py / breaker.py / fallback.py
  rag/                 retriever / fusion / reranker / calibration
  memory/              store / conflict / retrieval
  context/             budgeter / summarizer
  safety/              lexicon / classifier / policy / pii / output_guard
  storage/             repositories / migrations
  observability/       logging / tracing / metrics
prompts/               外置提示词（jinja2，按版本）
config/                crisis_resources.yaml 等
eval/                  评估集与评估脚本
tests/                 unit / integration / redteam
```

### 11.2 配置管理 — MUST

- `pydantic-settings` + `.env`，**所有密钥只从环境变量读，不入库不入仓**。
- 提供 `.env.example` 列出全部必需变量（含注释说明）。
- 启动时校验必需配置，缺失则**快速失败**（fail fast），不允许带着空 API key 启动后在运行时报错。
- 环境分离：`dev` / `staging` / `prod`，模型选择、日志级别、锐度上限按环境配置。

### 11.3 演进路径（分阶段，每阶段可独立交付）

| 阶段 | 目标 | 包含 |
|---|---|---|
| **P0 骨架** | 端到端能跑通 10 问 | Graph 状态机 + FakeLLM + 数据模型 + 规则层安全 + CLI 交互 |
| **P1 可用** | 真实 LLM + 结构化输出 | LLM Gateway（重试/超时/降级）+ Interviewer/Analyst + 报告生成 + SQLite 持久化 |
| **P2 可靠** | 抗故障与并发 | 熔断 + 分布式锁 + 幂等 + 上下文预算 + Critic + OutputGuard + 完整可观测 |
| **P3 智能** | RAG 与记忆 | 技术库 RAG + 混合检索 + Reranker（含低区分度处置）+ 四层记忆 + 冲突处理 |
| **P4 质量** | 评估驱动优化 | 评估集 + 回归门禁 + 提示词 A/B + 校准 + 红队集 |
| **P5 规模** | 生产化 | Postgres + Redis + 容器化 + 水平扩容 + 成本治理 |

**强制顺序约束：**
- **安全规则层在 P0 就必须有**，不可延后。
- **P2 之前不建议对外开放**（缺可靠性与输出防护）。
- **RAG 放在 P3** —— 先把提问流程本身做好。过早引入 RAG 会掩盖提问设计的问题，
  且没有评估集时无法判断 RAG 是否真的有帮助。

---

## 12. 风险清单与对策速查

| 风险 | 影响 | 对策 | 章节 |
|---|---|---|---|
| 用户处于心理危机 | 严重伤害 | 规则+模型双层检测，硬编码处置矩阵，不依赖 LLM 可用性 | 2.2 / 8.3 |
| 模型编造洞察 | 误导决策 | Evidence 原话子串程序化校验 + Critic 独立复核 | 4 / 5.4 |
| 长会话上下文超长 | 截断丢信息 | 硬预算 + 分层压缩 + 金句保留 + 优先级裁剪 | 5.3 |
| 提问变成给建议 | 违背方法论核心 | Agent 职责隔离 + OutputGuard 拦截 + 评估集打分 | 3.4 / 2.4 |
| Reranker 分数不可区分 | 检索质量不稳 | 相对差距判据 + 校准 + 低分弃用检索 + 降级链 | 6.4 |
| 记忆冲突 | 记错伤信任 | 不覆盖，时间序列 + 冲突分类 + 引用前确认 | 7.2 |
| 并发导致状态错乱 | 数据损坏 | 会话级分布式锁 + 乐观锁 + 幂等键 | 8.1 |
| LLM 不稳定 | 服务中断 | 超时分级 + 选择性重试 + 熔断 + 多级降级 | 8.2 |
| 静默降级 | 用户误信劣质输出 | 全链路 degraded 标记 + 响应可见 + 指标告警 | 8.5 |
| 提示词注入 | 越权/越界 | 数据隔离包裹 + 指令复述 + 注入检测 | 2.4 |
| 隐私泄露 | 合规风险 | 原话不入普通日志 + PII 脱敏 + 用户 id 哈希 + 记忆可删 | 9.1 / 7.2 |
| 成本失控 | 经济损失 | token 计量 + 会话成本上限 + 小模型分流 + 前缀缓存 | 8.2 / 8.4 |
| 无质量基线 | 改坏了不知道 | 三层评估 + 回归门禁 | 6.5 |

---

## 13. 给下游编码工具的总纲

按此顺序生成，每步产出可运行、可测试的增量：

1. `app/schemas/` + `app/domain/` + 对应单测（**零 LLM 依赖，先立契约**）
2. `app/safety/lexicon.py` + `policy.py` + 红队用例（**安全先行**）
3. `app/graph/` 状态机 + `FakeLLM` + 集成测试（**流程先跑通**）
4. `app/llm/gateway.py`（重试/超时/熔断/降级）+ 故障注入测试
5. `app/agents/` + `prompts/`（外置提示词）+ 结构化输出
6. `app/context/`（预算与压缩）+ `app/safety/output_guard.py`
7. `app/api/` + 并发控制（锁/幂等/限流）+ SSE 流式
8. `app/observability/` 全量接入
9. `app/rag/` + `app/memory/`
10. `eval/` 评估集与门禁

**贯穿全程的硬约束：**

- 所有 IO 为 `async`；纯逻辑为同步纯函数。
- **业务代码禁止直接调用 LLM SDK**，一律经 `app/llm/gateway.py`。
- 跨 Agent / 跨节点数据一律 Pydantic 模型，**禁止裸 dict 与自由文本**。
- 任何 LLM 输出在进入下一步前必须经过 schema 校验。
- 任何降级路径都必须有对应测试用例。
- 提示词不硬编码在 Python 中。
- 危机资源信息不由模型生成，只从配置读取。
- 类型注解完整，通过 mypy strict；代码格式用 ruff。

**依赖清单**（与项目现有 IDE 配置一致）：
```
langchain, langchain-core, langchain-openai, langchain-text-splitters,
langgraph, langgraph-checkpoint-sqlite, openai, chromadb, langchain-chroma,
tavily-python, fastapi, uvicorn, pydantic, pydantic-settings, python-dotenv,
requests, httpx, loguru, pytest, pytest-asyncio, mcp
补充建议：tiktoken(计数), tenacity(重试), redis(锁/缓存), rank-bm25(关键词检索),
opentelemetry-sdk(trace), ruff+mypy(质量), respx(httpx mock)
```
