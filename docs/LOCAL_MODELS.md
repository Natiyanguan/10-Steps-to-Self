# 本地模型：怎么生成、怎么接、能干什么

> 对应路线文档 §2.4（能力分级与门控）· §2.5（minimind 定位）· §3.3.1（嵌入分流）· §13.2.1（三档运行档位）
>
> 本文回答三个问题：**本地模型在这个项目里能干什么、不能干什么、以及怎么接进来。**

---

## 0. 先说结论（省你时间）

| 你想用本地模型做什么 | 可行性 | 用什么 | 说明 |
|---|---|---|---|
| **用户历史的向量化** | ✅ **强烈推荐** | BGE-M3 | 隐私收益最大、质量无损失。**本地化的首选项** |
| CI / 开发期跑通流程，不烧 token | ✅ 推荐 | minimind 或 Ollama 任意小模型 | 这是 minimind 的正确用途 |
| 安全判级、回答质量评估、摘要 | ⚠️ 可行但需实测 | Qwen2.5-7B-Instruct（Ollama/vLLM） | 必须先过 `tss-probe` 验证 JSON 稳定性 |
| 提问技术库检索的重排 | ✅ 可行 | bge-reranker-v2-m3 | 无云端 rerank API，本地是唯一选项 |
| **规划 10 问 / 生成问题 / 分析报告** | ❌ **不推荐**（消费级硬件） | 留在 DeepSeek | 见 §1，这是产品的核心价值所在 |
| 让 minimind 承担上面任何一项生产任务 | ❌ **代码硬阻断** | — | 见 §2 |

**一句话**：本地化的最大收益在**嵌入**（隐私），不在对话模型（质量）。

---

## 1. 为什么核心链路不建议用本地模型

这不是"本地模型不行"的偏见，而是这个产品的具体要求决定的。

四个 `STRONG` 任务对模型的真实要求（路线文档 §2.4.1）：

| 任务 | 硬要求 | 消费级本地模型的现实 |
|---|---|---|
| **Planner** 规划 10 问 | 同时满足 5 条约束（段内维度合法、相邻不重复、技术与维度配对、覆盖 ≥5 维、技术不连续重复）+ 输出嵌套 JSON | 7B 级模型经常违反其中 1-2 条，触发重规划 |
| **Interviewer** 生成问题 | 中文表达质量**极高** —— 问题的措辞就是产品本体 | 措辞生硬、爱说教、容易变成"你有没有想过应该…" |
| **Analyst** 分析报告 | 长上下文（全量问答）+ **逐字引用用户原话** | 上下文吃紧；引用容易改写，导致依据校验失败被丢弃 |
| **Critic** 复核依据 | 判断"引用真实但推论跳跃" | 这是最难的判断任务，小模型基本无法胜任 |

**最关键的一点**：这些失败**不会报错**。

模型会返回格式完整的 JSON，流程会正常走完，用户会拿到一份看起来很像样的心理分析报告 ——
只是里面的洞察是空话，建议是套话。**接口返回 200，没人知道出了问题。**

这就是路线文档 §9.5 反复强调"静默降级最危险"的具体含义，
也是 §2.4.3 能力门控存在的全部理由。

### 如果你就是要全本地（数据不出境的合规要求）

那么正确做法不是硬上，而是：

1. 用 Qwen2.5-14B-Instruct 或更大的模型（需要 ~16GB+ 显存跑 4bit 量化）
2. 跑 `tss-probe` 实测，把结果写进 `config/providers.yaml`
3. **在评估集上量化质量下降幅度**（路线文档 §7.5-C），而不是假装没有下降
4. 前端明确告知用户"当前为本地模式，分析深度受限"

第 3 步不可省。没有量化就不知道退化了多少，那才是真正的风险。

---

## 2. minimind：它在这个项目里是测试替身

### 2.1 事实基础

[minimind](https://github.com/jingyaogong/minimind) 是一个**从零训练小型中文 LLM 的教学项目**。
典型配置的参数量在 0.02B–0.1B 量级，上下文长度通常 512–8192（取决于你的训练配置）。

它的价值在于**让你完整走一遍 LLM 的训练流程**（预训练 → SFT → DPO / LoRA），
这个价值很高，但**不是"得到一个能用于生产的对话模型"**。

一个 0.02B 参数的模型和 DeepSeek 之间不是"差一点"，而是差几个数量级的能力。
它不具备本项目所需的复杂指令遵循与稳定嵌套 JSON 生成能力。

### 2.2 因此代码里做了硬阻断

`.env` 里的 `LOCAL_LLM_TIER` **不允许填 `strong`**：

```bash
LOCAL_LLM_TIER=toy    # minimind 等从零训练的小模型
# LOCAL_LLM_TIER=fast # 指令微调过的 7B 级模型（需先跑 tss-probe 验证）
# LOCAL_LLM_TIER=strong  ← 启动自检会直接拒绝启动
```

试一下就知道：

```bash
LOCAL_LLM_TIER=strong tss doctor
# ✗ 配置加载失败：LOCAL_LLM_TIER 不允许为 strong（§2.4.3）：
#     本地小模型不具备 Planner/Interviewer/Analyst/Critic 所需的能力，
#     声明为 strong 会让用户拿到格式完整但内容失效的心理分析。
```

而且**即使你绕过配置**，`app/llm/capability.py` 里还有一条无条件断言：
`TOY` 档承担 `STRONG` 任务时直接抛 `InsufficientCapabilityError`，不读任何配置。

这条有测试守着（`tests/integration/test_local_profile.py::test_toy_tier_hard_blocked_even_if_misconfigured`）。

### 2.3 那 minimind 用在哪

**CI 冒烟**。这个用途的价值被低估了：

```bash
# 完全离线、免费、每次提交都跑
pytest tests/integration/test_local_profile.py
```

它验证的是**真实的 HTTP 链路**：provider adapter 通不通、流式能不能收、
超时会不会触发、重试链会不会转、schema 校验会不会拦。

这些是 FakeLLM 验证不了的（FakeLLM 不走网络），
但用 DeepSeek 验证又要花钱、要联网、还不稳定。minimind 正好填这个位置。

**断言只针对流程与 schema，绝不针对内容质量** —— 这是刻意的设计。

### 2.4 重要：TOY 档不经过 Gateway

一个容易踩的坑。本项目**所有** `TaskKind` 的最低档位要求都是 `FAST`：

```
plan/ask/probe/analyze/critique  → STRONG
safety/quality/summarize/query_rewrite/rerank_judge → FAST
```

`TOY < FAST`，所以 **minimind 无法通过 Gateway 承担任何任务** —— 不是"能力受限"，
而是路由层面一个都不给。把它配进 `.env` 然后跑 `tss chat`，
你会看到"无可用 provider"，那不是代码坏了，是门控在正常工作。

因此 minimind 的正确用法是**在 provider 层直连**，绕开 Gateway：

```python
# 测试里直接构造，不走 gateway.complete()
from app.llm.providers.local_minimind import MinimindProvider

p = MinimindProvider(base_url="http://127.0.0.1:8000")
resp = await p.complete(req)   # 验证 adapter 本身
```

`app/llm/providers/local_minimind.py` 的存在意义就在这里：
验证"形状转换 + 错误映射 + 超时"这一层是对的。
至于它能不能被业务调用 —— 按设计，不能。

---

## 3. 怎么训练 minimind（租 GPU 服务器）

你提到可以租服务器跑训练。下面是务实路径。

### 3.1 硬件与成本预估

| 阶段 | 显存需求 | 参考时长 | 说明 |
|---|---|---|---|
| 预训练（0.02B / 小语料） | 单卡 8–24GB | 数小时 | 一张 RTX 3090 / 4090 / A10 够用 |
| SFT 指令微调 | 同上 | 1–3 小时 | 语料量小得多 |
| DPO / LoRA | 同上 | 1 小时内 | 可选 |

按国内 GPU 租用市场的常见价格，单卡时租通常在几元人民币量级，
**跑完一轮完整流程的成本大概是一顿饭钱**。这也是 minimind 作为教学项目的意义所在。

> 具体价格与显存需求请以你选的平台和 minimind 当前版本的 README 为准 ——
> 这个项目在持续更新，我不保证上面的数字和你看到的完全一致。

### 3.2 训练流程（照 minimind 官方 README 走）

```bash
# 1. 拉代码（在租的服务器上）
git clone https://github.com/jingyaogong/minimind.git
cd minimind
pip install -r requirements.txt

# 2. 准备数据集（项目提供了下载脚本 / HuggingFace 数据集链接）
#    预训练语料 + SFT 指令数据

# 3. 预训练
python train_pretrain.py

# 4. 指令微调（这一步决定它能不能听懂"输出 JSON"这类指令）
python train_full_sft.py

# 5. 可选：DPO 对齐
python train_dpo.py

# 6. 把 checkpoint 打包下载回本地
```

**对本项目而言，第 4 步的 SFT 最关键。** 没有 SFT 的模型连"只输出 JSON"都做不到，
连 CI 冒烟都跑不通（它会返回散文）。

### 3.3 如果你想让它在冒烟测试里更有用

在 SFT 阶段**掺入一批本项目的结构化输出样本**，能显著提高它产出合法 JSON 的比例：

```jsonl
{"instruction": "只输出 JSON：{\"text\": \"<问题>\"}", "input": "困惑：要不要离职", "output": "{\"text\": \"你最怕失去的是什么？\"}"}
{"instruction": "只输出 JSON：{\"level\": \"none|low|medium|high|imminent\", \"reason\": \"\"}", "input": "最近有点累", "output": "{\"level\": \"low\", \"reason\": \"一般性疲惫\"}"}
```

这些样本可以直接从 `prompts/` 下的模板 + `tests/conftest.py` 里的合法输出生成。

**注意边界**：这么做是为了让 CI 冒烟更顺畅，**不是**为了让它承担生产任务。
即使 JSON 格式对了，内容质量也不会因此达到 `STRONG` 要求。

### 3.4 训练数据的隐私红线

**绝不要用真实用户的会话数据训练模型。**

路线文档 §7.6 对评估集的要求同样适用于训练集：
默认只用合成/专家编写的案例；用户数据需显式授权 `eval_use` 且经充分脱敏改写。

模型会记住训练数据。用真实心理咨询对话训练出的模型，可能在推理时吐出别人的原话。
这是不可逆的泄露。

---

## 4. 怎么把本地模型接进来

### 4.1 关键设计：不需要写新代码

`app/llm/providers/openai_compat.py` 这一个 adapter **同时服务 DeepSeek 和所有本地模型** ——
只要对方暴露 OpenAI 兼容的 `/chat/completions` 接口。

这是路线文档决策 #15 的核心收益：**切换 provider = 改配置，业务代码零改动。**

### 4.2 路径 A：Ollama（最省事，推荐）

```bash
# 安装 Ollama 后
ollama pull qwen2.5:7b-instruct
ollama serve          # 默认监听 11434
```

```bash
# .env
RUNTIME_PROFILE=hybrid
LOCAL_LLM_BASE_URL=http://127.0.0.1:11434/v1
LOCAL_LLM_MODEL=qwen2.5:7b-instruct
LOCAL_LLM_TIER=fast          # 先跑 tss-probe 确认后再填 fast
```

### 4.3 路径 B：vLLM（吞吐更好，适合有独立 GPU 机器）

```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --port 8000 \
  --max-model-len 32768
```

```bash
LOCAL_LLM_BASE_URL=http://127.0.0.1:8000/v1
LOCAL_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
LOCAL_LLM_TIER=fast
```

### 4.4 路径 C：minimind

**先读 §2.4** —— minimind 是 `TOY` 档，填完配置也**不会被 Gateway 路由**。
下面说的是怎么让 adapter 本身可用（供 CI 冒烟直连），不是让它接管业务。

若 minimind 自带 OpenAI 兼容的 serve 接口，用现成的 `OpenAICompatProvider` 即可，
不需要写新代码：

```python
from app.llm.capability import ModelTier
from app.llm.providers.openai_compat import OpenAICompatProvider
from app.llm.providers.local_minimind import minimind_caps

p = OpenAICompatProvider(
    name="local_minimind",
    base_url="http://127.0.0.1:8000/v1",
    api_key="",
    model="minimind",
    caps=minimind_caps(),      # tier 恒为 TOY
)
```

若只有原生推理接口（非 OpenAI 形状），用 `app/llm/providers/local_minimind.py`：

```python
from app.llm.providers.local_minimind import MinimindProvider

p = MinimindProvider(base_url="http://127.0.0.1:8000", endpoint="/generate")
resp = await p.complete(req)
```

接不同的 serve 脚本**只需改 `_to_native` / `_from_native` 两个方法**，
超时映射、错误分类（4xx 不可重试 / 5xx 可重试）、usage 估算都已就位。

两点要注意：

- **chat template 必须与 SFT 训练时一致**。`_to_native` 默认用最朴素的
  `role: content` 拼接；若你训练时用了 `<|im_start|>` 之类的模板，这里不对齐会让
  输出明显变差 —— **而且不会报错**，只表现为答非所问。
- **不要在业务层为它开特例分支**（§2.5）。一旦出现 `if provider == "minimind"`，
  provider 抽象就失效了，之后每接一个模型都要改业务代码。差异全部收在这一层。

### 4.5 接完必须跑探测

```bash
tss-probe                    # 产出 config/providers.yaml
tss doctor --probe           # 看各任务能路由到哪些 provider
```

探测会实测 10 项能力，其中**两项决定档位**：

- **JSON 稳定性**（20 次采样）：低于 95% 不得承担对应档位。
  为什么这么严：报告分段生成会把不稳定性放大 3–6 倍（§6.5）。
- **中文提问质量**（10 次采样）：直接用 `OutputGuard` 检查是否给建议、是否引导性提问。
  这一项不合格的模型不能做 `ASK` 任务 —— 措辞就是产品本体。

**探测结果不佳时，`tss-probe` 会主动建议降档**，而不是让你把它填成 `fast` 然后在生产踩坑。

---

## 5. 本地嵌入：这才是本地化的最大收益

### 5.1 为什么

路线文档 §3.3 指出了一个容易被忽略的隐私风险：

> **嵌入服务也是数据出境通道。**
> 为了做跨会话检索，用户的每一段历史表述都要送去向量化，
> **累积暴露量甚至超过单次对话。**

只盯着 LLM 而忽略 embedding 是常见疏漏。而这件事本地化的代价很小：
BGE-M3 原生 1024 维，中文表现优秀，CPU 也能跑（慢一些但可接受）。

### 5.2 按 scope 分流（推荐的生产配置）

```yaml
# config/task_models.yaml
embeddings:
  kb:    cloud_zhipu      # 公共知识库：无隐私问题，用云端省本地算力
  user:  local_bge_m3     # 用户历史：留在本机，隐私最优
```

```bash
# .env
RUNTIME_PROFILE=hybrid       # 这一档就是为此设计的
LOCAL_EMBED_MODEL=BAAI/bge-m3
LOCAL_EMBED_DEVICE=cpu       # 有 GPU 就填 cuda
EMBED_TARGET_DIM=1024        # 与 BGE-M3 原生维度一致
```

### 5.3 硬约束：两侧必须同维

`EMBED_TARGET_DIM=1024` 不是随便挑的数字（路线文档 §2.6 决策 #19）：

- BGE-M3 原生就是 1024 维
- pgvector 的 HNSW 索引上限是 2000 维，而智谱 `embedding-3` 是 **2048 维**
- 统一到 1024，云/本地切换就不必改表结构，将来迁 pgvector 也能建索引

**换嵌入模型或维度 → 必须全量重建索引**，新旧向量绝不可混用 ——
不同模型的向量空间不可比，混用会让检索**静默失效**，这是最难发现的一类故障。

启动自检会校验 collection 维度与 `EMBED_TARGET_DIM` 是否一致，不一致直接拒绝启动。

### 5.4 安装

```bash
uv pip install -e ".[local]"     # 装 FlagEmbedding + torch（体积数 GB）
```

注意：`cloud` 档**不需要**装这一组。路线文档 §16.3 特意做了依赖分组，
就是为了让云端部署的镜像不必背着几 GB 的 torch。

---

## 6. 三档运行档位速查

```bash
RUNTIME_PROFILE=cloud     # LLM 云端 + 嵌入云端（起步默认）
RUNTIME_PROFILE=hybrid    # LLM 云端 + 用户历史嵌入本地（★ 推荐生产）
RUNTIME_PROFILE=local     # 全本地（开发 / CI / 数据不出境）
```

| | cloud | hybrid | local |
|---|---|---|---|
| 提问质量 | 最好 | 最好 | 明显下降 |
| 用户历史隐私 | 出境 | **不出境** | 不出境 |
| 持续成本 | 最高 | 中 | 最低 |
| 硬件要求 | 无 | 能跑 BGE-M3 | 能跑 7B+ |
| `STRONG` 能力 | ✅ | ✅ | ❌ |

**切换档位只改配置，不改代码。** 但切换嵌入档位会触发索引重建校验。

---

## 7. 常见问题

**Q：我把 minimind 标成 `fast`，能跑安全判级吗？**

技术上启动自检会放行（`fast` 是合法值），但 `tss-probe` 几乎肯定会给出降档建议。
而且路线文档 §2.5 明确禁止：**安全不能建立在能力不足的模型上**。

安全判级的第一层是 `app/safety/lexicon.py` 的规则词表 —— **零 LLM 依赖**。
LLM 全挂时它仍然工作，这是设计要求（§9.3）。
第二层分类器不可用时，系统走保守分支（`HOLD_UNVERIFIABLE`），
**不进入深度提问**，而不是当作"无风险"继续。

所以：与其让弱模型做不可靠的判级，不如让它明确不可用，走保守路径。

**Q：能不能让本地模型做 Planner，云端只做 Interviewer？**

能配置，但收益很低。Planner 是约束最密集的任务（5 条硬约束同时满足），
本地模型在这里失败率最高，而失败意味着重规划 —— 每次重规划都是真实的 token 开销。

真要省钱，正确的切分是把 `FAST` 档任务（安全/质量/摘要，约占调用次数一半）移到本地，
`STRONG` 档留在云端。这也是 §14.1 成本模型的建议。

**Q：为什么 `--fake` 用的不是 minimind？**

`tss chat --fake` 用的是 `DemoProvider`，它从提示词里**解析约束**后产出结构合法的响应。
这样做同时验证了两件事：提示词真的把约束讲清楚了（否则解析不到），
以及领域约束是可满足的（存在至少一个合法解）。

这个设计已经抓到过一个真实缺陷：提示词最初把"允许的维度"和"可用的技术"
列成两份独立列表，没说清哪个技术配哪个维度，导致模型组合出非法配对、反复重规划。
如果 `--fake` 只是返回硬编码答案，这个缺陷会一直留到接真实模型时才暴露 —— 而且要花钱才能发现。

---

## 8. 落地顺序建议

```
1. 先用 cloud 档把流程跑通            tss doctor && tss chat
2. 跑探测，把实测能力落盘              tss-probe
3. 装本地嵌入，切 hybrid              uv pip install -e ".[local]"
4. （可选）接 Ollama 承担 FAST 任务    先 tss-probe 验证再改 tier
5. （可选）训练 minimind 做 CI 冒烟    租 GPU 跑 SFT
```

**第 3 步的性价比最高** —— 隐私收益大、质量无损失、成本下降。
第 4、5 步是可选优化，不做也不影响产品可用。
