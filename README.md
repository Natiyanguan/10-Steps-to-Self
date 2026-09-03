# 10-Steps-to-Self

一个基于**苏格拉底式反向提问**的自我探索 Agent。

它不直接给你答案，而是扮演一位精通苏格拉底提问术的引导者，通过 10 个递进式问题，
把你自己心里已有的真实想法"接生"出来，最后生成一份**潜意识诉求分析 + 可执行建议**的复盘报告。

> ⚠️ 这是一个**自我探索与思维梳理工具**，不是医疗服务，不做诊断、不做治疗、不开处方。
> 如你正处于心理危机，请优先联系专业帮助。

---

## 方法论

系统依赖四个心理学/认知科学理论（详见 `docs/TECH_ROADMAP_FINAL.md` §1）：

| 理论 | 在系统里的落点 |
|---|---|
| 苏格拉底产婆术 | 递进式提问策略引擎，**禁止给建议** |
| 乔哈里视窗 | 维度覆盖检查器，强制打到用户未提及的盲区 |
| 焦点解决短期治疗 | 奇迹/例外/量尺/循环等提问技术库 |
| 边界对象与认知卸载 | 单问单答节奏 + 进度可视化 |

核心资产不是"回答能力"，而是**提问质量**与**长期记忆的连贯性**。

## 关键工程特性

- **能力门控**：模型按档位（STRONG/FAST/TOY）分级，档位不足的任务显式拒绝路由，
  绝不静默劣化出"格式完整但内容失效"的心理分析。
- **反幻觉四层防御**：提示词约束 → 结构化输出强制挂依据 → **`Evidence.quote` 逐字校验**
  （程序化，不依赖模型判断）→ Critic 独立复核。
- **安全优先**：危机词表零 LLM 依赖（LLM 全挂时仍工作）、注入剥离、输出守卫、
  危机资源全部来自配置而非模型生成。
- **上下文压缩**：长会话按水位触发摘要，逐字原话保留，压缩失败自动降级不阻断。
- **会话可恢复 + 乐观锁**：SessionState 快照持久化，`revision` 乐观锁防并发脏写。

## 快速开始

### 1. 安装

```bash
# 推荐用 uv（Python 3.12+）
uv pip install -e ".[dev]"      # 含测试与质量门禁
uv pip install -e ".[cloud]"    # 含 FastAPI / SQLAlchemy / 指标
# 本地嵌入（hybrid 档才需要，体积数 GB）：
uv pip install -e ".[local]"
```

### 2. 配置

```bash
cp .env.example .env
# 编辑 .env，填入 DeepSeek key 与模型标识
```

### 3. 验证

```bash
tss doctor          # 启动自检 + 能力矩阵
tss chat --fake     # 离线跑通完整 10 问流程（不花 token、不联网）
tss-probe           # 供应商能力探测，生成 config/providers.yaml
```

### 4. 真实运行

```bash
tss chat                        # CLI 交互式会话
uvicorn app.main:app --reload   # HTTP API（/docs 看接口）
```

## 目录结构

```
app/
  domain/       纯函数领域逻辑（维度/技术/模式），零 LLM 依赖
  schemas/      Pydantic 全流程状态契约
  safety/       危机词表 / 注入剥离 / Evidence 校验 / 输出守卫（零 LLM、零网络）
  llm/          能力门控 / Gateway（重试/熔断/降级）/ provider 抽象
  agents/       7 个 Agent（planner/interviewer/analyst/critic/…）
  graph/        LangGraph 状态机（按 phase 条件入口，可恢复）
  context/      上下文预算 + 压缩决策
  storage/      SQLAlchemy 持久化（快照 + 乐观锁 + 审计）
  api/          FastAPI 路由 + 错误码映射
  observability Prometheus 指标 + loguru JSON 日志
prompts/        外置提示词（jinja2，版本化）
config/         危机资源等配置
tests/          单元 / 集成测试（FakeLLM 为地基）
docs/           技术路线 / 本地模型 / 交接说明
```

## 测试与质量门禁

```bash
pytest                          # 289 项测试
ruff check app tests scripts    # lint
mypy app scripts                # 类型检查（strict）
```

测试全部离线、免费：用 `FakeProvider`（可编程）和 `DemoProvider`（从提示词解析约束）
驱动完整状态机，不依赖真实 LLM。

## 文档

| 文档 | 内容 |
|---|---|
| `docs/TECH_ROADMAP_FINAL.md` | 权威技术路线（架构决策 + 生成顺序） |
| `docs/LOCAL_MODELS.md` | 本地模型（minimind/Ollama/BGE-M3）怎么接 |
| `docs/HANDOFF.md` | 当前进展与下一步（接手必读） |

## 技术栈

Python 3.12 · LangGraph · FastAPI · Pydantic v2 · SQLAlchemy(async) · SQLite
· DeepSeek（OpenAI 兼容协议，可切本地模型）· loguru · Prometheus
