"""供应商能力探测（路线文档 §17）。

**接真实模型前必须先跑这个。** 产出 `config/providers.yaml`，
后续所有能力门控（§2.4.3）都以它为依据。

设计原则（§17.4）：
  - 只花少量 token：多数探测项是单次小请求；
    JSON 稳定性与中文质量各跑 10-20 次 —— 这是必要开销，因为它们决定档位。
  - **可重复运行**：换模型或供应商后重跑，diff 配置即知能力变化。
  - **失败按最保守值处理**，绝不乐观假设：
    `supports_json_mode` 探测失败 → 记为 false。
    乐观假设会在生产以"解析失败"的形式暴露，那是持续成本。

用法：
    tss-probe                    # 探测全部（LLM + 嵌入 + 本地）
    tss-probe --llm-only         # 只探 LLM
    tss-probe --quick            # 跳过多次采样项（省 token，但不足以定档位）
    tss-probe --dry-run          # 只打印将要做什么，不发请求
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import yaml

from app.config import Settings, get_settings
from app.llm.base import ChatRequest, Message, Role, TaskKind
from app.llm.capability import ModelTier, ProviderCaps, ZhQuality
from app.llm.providers.openai_compat import OpenAICompatProvider
from app.llm.structured import parse_json_lenient
from app.safety.output_guard import check_question

if sys.platform == "win32":
    for _s in (sys.stdout, sys.stderr):
        if hasattr(_s, "reconfigure"):
            _s.reconfigure(encoding="utf-8", errors="replace")

# 采样次数：§17.1 第 10 项 / §17.3 第 3 项（决定档位，不能省）
JSON_SAMPLES = 20
QUALITY_SAMPLES = 10
LATENCY_SAMPLES = 5

# JSON 稳定性阈值（§17.3）：低于此值不得承担对应档位
JSON_OK_FOR_STRONG = 0.95
JSON_OK_FOR_FAST = 0.95
JSON_MIN_FOR_TOY = 0.50

# 上下文探测的候选长度（token 近似，用中文字符数放大）
CONTEXT_PROBES = (4_000, 8_000, 16_000, 32_000, 64_000, 128_000)


def _hr(c: str = "─", n: int = 68) -> str:
    return c * n


# --------------------------------------------------------------------------
@dataclass
class LLMProbeResult:
    """§17.1 的探测结果。字段与 ProviderCaps 对齐，便于直接落盘。"""

    name: str
    model: str
    reachable: bool = False
    error: str = ""

    context_window: int = 0
    max_output_tokens: int = 0
    supports_json_mode: bool = False
    supports_function_calling: bool = False
    supports_temperature: bool = True
    supports_streaming: bool = False
    is_reasoning: bool = False

    json_success_rate: float | None = None
    zh_quality: str = "unknown"
    zh_violations: list[str] = field(default_factory=list)

    latency_p95_ms: float | None = None
    token_ratio_actual_vs_estimate: float | None = None

    suggested_tier: str = "unknown"
    notes: list[str] = field(default_factory=list)


NESTED_SCHEMA_PROMPT = """只输出一个 JSON 对象，不要 markdown 围栏、不要解释文字。结构：
{"items": [{"name": "<字符串>", "tags": ["<字符串>"], "score": <0-1 小数>}], "总结": "<字符串>"}
要求 items 恰好 2 条，每条 tags 恰好 2 个。"""

ZH_QUALITY_CASES = (
    "我在纠结是否要离职创业",
    "和伴侣总在同一件事上吵架",
    "工作三年感觉没有成长",
    "要不要接受外地的工作机会",
    "总觉得自己不如同龄人",
    "父母希望我考公务员",
    "想换行业但怕来不及",
    "对现在的生活提不起劲",
    "被提拔了却很焦虑",
    "不知道该不该结束一段关系",
)

ASK_PROMPT = """你是苏格拉底式提问者。针对下面的困惑，提出**一个**开放式问句。

硬要求：
1. 只输出 JSON：{"text": "<你的问题>"}
2. 必须是问句，以问号结尾
3. **不能给建议**（不要出现"你应该""我建议""不如"）
4. **不能暗含预设答案**（如"难道你不觉得该辞职吗？"）
5. 不超过 60 字

困惑：{concern}"""


def _make_provider(
    *, name: str, base_url: str, api_key: str, model: str, timeout: float
) -> OpenAICompatProvider:
    """探测阶段用最宽松的 caps —— 目的就是测出真实能力，不能被门控挡住。"""
    permissive = ProviderCaps(
        tier=ModelTier.STRONG,
        context_window=1_000_000,
        max_output_tokens=32_000,
        supports_json_mode=True,
        supports_function_calling=False,
        supports_temperature=True,
        supports_streaming=True,
        zh_quality=ZhQuality.HIGH,
        probed_at=None,
    )
    return OpenAICompatProvider(
        name=name,
        base_url=base_url,
        api_key=api_key,
        model=model,
        caps=permissive,
        default_timeout_s=timeout,
    )


async def _ask(
    provider: OpenAICompatProvider,
    text: str,
    *,
    json_mode: bool = False,
    max_out: int = 200,
    temperature: float | None = None,
    timeout: float = 60.0,
) -> tuple[str, Any, str]:
    """发一次请求。返回 (文本, usage, 错误)。"""
    req = ChatRequest(
        task=TaskKind.QUERY_REWRITE,
        messages=(Message(Role.USER, text),),
        json_mode=json_mode,
        temperature=temperature,
        max_output_tokens=max_out,
        timeout_s=timeout,
    )
    try:
        resp = await provider.complete(req)
    except Exception as exc:
        return "", None, f"{type(exc).__name__}: {exc}"
    return resp.text, resp, ""


async def probe_llm(
    *, name: str, base_url: str, api_key: str, model: str, quick: bool, timeout: float
) -> LLMProbeResult:
    r = LLMProbeResult(name=name, model=model)
    p = _make_provider(
        name=name, base_url=base_url, api_key=api_key, model=model, timeout=timeout
    )

    try:
        # ---- 1. 连通性 ----
        text, resp, err = await _ask(p, "ping", max_out=8)
        if err:
            r.error = err
            r.notes.append("连通性失败：后续探测跳过，所有能力按最保守值记录")
            return r
        r.reachable = True
        if resp is not None:
            r.model = resp.model or model

        # ---- 2. 是否推理型（§17.1 第 7 项）----
        # 推理型模型在给出正文前会先产出思维链，completion_tokens 会明显高于正文长度
        if resp is not None and resp.usage.completion_tokens > max(20, len(text) * 2):
            r.is_reasoning = True
            r.notes.append(
                f"疑似推理型：completion_tokens={resp.usage.completion_tokens} "
                f"远超正文长度 {len(text)} —— 输出预算需为思维链留额度（§6.5）"
            )

        # ---- 3. temperature 支持（§17.1 第 6 项）----
        _, _, err_t = await _ask(p, "ping", max_out=8, temperature=0.5)
        r.supports_temperature = not err_t
        if err_t:
            r.notes.append(f"不接受 temperature（{err_t[:80]}）—— Gateway 必须裁剪该参数")

        # ---- 4. JSON 模式（§17.1 第 4 项）----
        txt_j, _, err_j = await _ask(
            p, NESTED_SCHEMA_PROMPT, json_mode=True, max_out=400
        )
        if err_j:
            r.supports_json_mode = False
            r.notes.append(f"response_format 被拒绝：{err_j[:80]}")
        else:
            r.supports_json_mode = parse_json_lenient(txt_j).ok
            if not r.supports_json_mode:
                r.notes.append("接受 response_format 但输出仍非合法 JSON")

        # ---- 5. 流式（§17.1 第 8 项）----
        try:
            got = False
            req = ChatRequest(
                task=TaskKind.QUERY_REWRITE,
                messages=(Message(Role.USER, "数到三"),),
                max_output_tokens=32,
                timeout_s=timeout,
            )
            async for chunk in p.stream(req):
                if chunk:
                    got = True
                    break
            r.supports_streaming = got
        except Exception as exc:
            r.notes.append(f"流式不可用：{type(exc).__name__}")

        # ---- 6. 输出上限（§17.1 第 3 项）----
        txt_long, resp_long, err_long = await _ask(
            p, "请连续输出「测试」这两个字，重复 2000 次，不要任何其他内容。", max_out=4096
        )
        if not err_long and resp_long is not None:
            r.max_output_tokens = max(
                resp_long.usage.completion_tokens, len(txt_long) // 2
            )
            if resp_long.truncated:
                r.notes.append(
                    f"输出在 {resp_long.usage.completion_tokens} tokens 处被截断 —— "
                    "报告必须分段生成（§6.5）"
                )

        # ---- 7. 上下文窗口（§17.1 第 2 项）----
        r.context_window = await _probe_context(p, timeout=timeout, quick=quick)

        # ---- 8. token 计数偏差（§17.1 第 11 项）----
        if resp is not None:
            from app.context.budget import count_text

            probe_text = "这是一段用于校准 token 计数的中文文本。" * 10
            _, resp_c, err_c = await _ask(p, probe_text, max_out=8)
            if not err_c and resp_c is not None and resp_c.usage.prompt_tokens:
                est = count_text(probe_text, tokenizer_hint="cl100k_base").tokens
                if est:
                    r.token_ratio_actual_vs_estimate = round(
                        resp_c.usage.prompt_tokens / est, 3
                    )
                    if not 0.8 <= r.token_ratio_actual_vs_estimate <= 1.25:
                        r.notes.append(
                            f"token 估算偏差较大（实际/估算={r.token_ratio_actual_vs_estimate}）"
                            " —— 需调整 ESTIMATE_SAFETY_FACTOR（§6.3）"
                        )

        if quick:
            r.notes.append("--quick：跳过 JSON 稳定性与中文质量采样，**不足以判定档位**")
            r.suggested_tier = "unknown"
            return r

        # ---- 9. JSON 稳定性（§17.1 第 10 项，决定档位）----
        ok = 0
        for _ in range(JSON_SAMPLES):
            t, _, e = await _ask(
                p,
                NESTED_SCHEMA_PROMPT,
                json_mode=r.supports_json_mode,
                max_out=400,
                temperature=0.3 if r.supports_temperature else None,
            )
            if not e and parse_json_lenient(t).ok:
                ok += 1
        r.json_success_rate = round(ok / JSON_SAMPLES, 3)

        # ---- 10. 中文提问质量（§17.1 第 9 项，决定档位）----
        r.zh_quality, r.zh_violations = await _probe_zh_quality(p, r)

        # ---- 11. 延迟 P95（§17.1 第 13 项）----
        lat: list[float] = []
        for case in ZH_QUALITY_CASES[:LATENCY_SAMPLES]:
            t0 = time.perf_counter()
            _, _, e = await _ask(p, ASK_PROMPT.format(concern=case), max_out=200)
            if not e:
                lat.append((time.perf_counter() - t0) * 1000)
        if lat:
            r.latency_p95_ms = round(
                statistics.quantiles(lat, n=20)[-1] if len(lat) >= 4 else max(lat), 1
            )

        r.suggested_tier = _suggest_tier(r)
    finally:
        await p.aclose()
    return r


async def _probe_context(
    p: OpenAICompatProvider, *, timeout: float, quick: bool
) -> int:
    """递增长度直到报错。quick 模式只试前两档，省 token。"""
    probes = CONTEXT_PROBES[:2] if quick else CONTEXT_PROBES
    best = 0
    for target in probes:
        # 中文约 1 字 ≈ 1 token，用重复文本逼近目标长度
        filler = "上下文长度探测。" * (target // 8)
        _, _, err = await _ask(
            p, filler + "\n请只回复「ok」。", max_out=8, timeout=timeout
        )
        if err:
            break
        best = target
    return best


async def _probe_zh_quality(
    p: OpenAICompatProvider, r: LLMProbeResult
) -> tuple[str, list[str]]:
    """用 OutputGuard 客观检查提问质量（§17.1 第 9 项）。

    这里不做主观打分 —— 用与生产同一套规则检查"是否违规给建议/是否引导性"，
    结果才和线上行为一致。
    """
    violations: list[str] = []
    valid = 0
    for case in ZH_QUALITY_CASES[:QUALITY_SAMPLES]:
        t, _, e = await _ask(
            p,
            ASK_PROMPT.format(concern=case),
            json_mode=r.supports_json_mode,
            max_out=300,
        )
        if e:
            continue
        parsed = parse_json_lenient(t)
        question = ""
        if parsed.ok and isinstance(parsed.data, dict):
            question = str(parsed.data.get("text", ""))
        else:
            question = t.strip()
        if not question:
            continue
        guard = check_question(question)
        if guard.ok:
            valid += 1
        else:
            violations.append(f"{question[:40]} → {guard.summary[:60]}")

    rate = valid / QUALITY_SAMPLES
    if rate >= 0.9:
        return "high", violations
    if rate >= 0.7:
        return "medium", violations
    return "low", violations


def _suggest_tier(r: LLMProbeResult) -> str:
    """按实测结果建议档位（§2.4.2）。**只降不升，宁可保守。**"""
    if not r.reachable:
        return "unknown"
    jsr = r.json_success_rate if r.json_success_rate is not None else 0.0

    # §17.3：JSON 稳定性不足 95% 不得承担 STRONG（分段生成会放大 3-6 倍）
    if (
        jsr >= JSON_OK_FOR_STRONG
        and r.supports_json_mode
        and r.zh_quality == "high"
        and r.context_window >= 16_000
        and r.max_output_tokens >= 1_500
    ):
        return "strong"
    if jsr >= JSON_OK_FOR_FAST and r.supports_json_mode and r.zh_quality != "low":
        return "fast"
    if jsr >= JSON_MIN_FOR_TOY:
        return "toy"
    return "toy"


# --------------------------------------------------------------------------
@dataclass
class EmbedProbeResult:
    """§17.2 的探测结果。第 2 项决定架构走向（§2.6）。"""

    model: str
    reachable: bool = False
    error: str = ""
    native_dim: int = 0
    supports_dimensions_param: bool = False
    dim_when_requested: int | None = None
    is_normalized: bool | None = None
    batch_limit_ok: bool | None = None
    long_text_behavior: str = "unknown"
    notes: list[str] = field(default_factory=list)


async def probe_embeddings(settings: Settings, *, timeout: float = 60.0) -> EmbedProbeResult:
    r = EmbedProbeResult(model=settings.embedding_model)
    if not settings.embed_api_key:
        r.error = "EMBED_API_KEY 未配置"
        return r

    url = settings.embed_base_url.rstrip("/") + "/embeddings"
    headers = {
        "Authorization": f"Bearer {settings.embed_api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=timeout) as client:

        async def call(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
            try:
                resp = await client.post(url, headers=headers, json=payload)
            except httpx.HTTPError as exc:
                return None, f"{type(exc).__name__}: {exc}"
            if resp.status_code >= 400:
                return None, f"HTTP {resp.status_code}: {resp.text[:200]}"
            return resp.json(), ""

        # ---- 1. 原生维度 ----
        data, err = await call({"model": r.model, "input": "维度探测"})
        if err or not data:
            r.error = err or "空响应"
            return r
        r.reachable = True
        vec = (data.get("data") or [{}])[0].get("embedding") or []
        r.native_dim = len(vec)
        if r.native_dim != settings.embed_dim_native:
            r.notes.append(
                f"实测维度 {r.native_dim} 与 EMBED_DIM_NATIVE={settings.embed_dim_native} 不一致"
            )

        # ---- 5. 是否已归一化（决定用余弦还是内积）----
        if vec:
            norm = sum(x * x for x in vec) ** 0.5
            r.is_normalized = abs(norm - 1.0) < 0.05
            r.notes.append(f"向量模长 {norm:.4f} → {'已归一化' if r.is_normalized else '未归一化'}")

        # ---- 2. 是否支持 dimensions 降维（**最先该看的一项**，§2.6）----
        target = settings.embed_target_dim
        data2, err2 = await call(
            {"model": r.model, "input": "降维探测", "dimensions": target}
        )
        if err2 or not data2:
            r.supports_dimensions_param = False
            r.notes.append(
                f"不支持 dimensions 参数（{(err2 or '')[:80]}）→ "
                f"须走 §2.6 方案 2（Chroma + {r.native_dim} 维）"
                "，或自建投影并验证 Recall@10 降幅 < 3%"
            )
        else:
            v2 = (data2.get("data") or [{}])[0].get("embedding") or []
            r.dim_when_requested = len(v2)
            r.supports_dimensions_param = len(v2) == target
            if r.supports_dimensions_param:
                r.notes.append(
                    f"支持 dimensions={target} → §2.6 方案 1 成立，"
                    "云/本地同维，pgvector 路线保留"
                )
            else:
                r.notes.append(f"请求 {target} 维但返回 {len(v2)} 维 —— 视为不支持")

        # ---- 3. 批量上限 ----
        batch = ["批量探测"] * settings.embed_batch_size
        data3, err3 = await call({"model": r.model, "input": batch})
        r.batch_limit_ok = not err3 and len((data3 or {}).get("data") or []) == len(batch)
        if not r.batch_limit_ok:
            r.notes.append(f"EMBED_BATCH_SIZE={settings.embed_batch_size} 不被接受：{err3[:80]}")

        # ---- 4. 长文本截断行为（静默截断会让检索莫名漏内容）----
        long_text = "长文本截断行为探测。" * 4000
        data4, err4 = await call({"model": r.model, "input": long_text})
        if err4:
            r.long_text_behavior = "explicit_error"
            r.notes.append("超长输入会显式报错 —— 分块时需自行控制长度（可预期，较安全）")
        elif data4:
            r.long_text_behavior = "silent_truncate"
            r.notes.append(
                "超长输入被**静默接受** —— 可能已截断且不报错，"
                "长回答的后半部分会实际未被索引（§17.2 第 4 项）"
            )

    return r


# --------------------------------------------------------------------------
def _caps_dict(r: LLMProbeResult) -> dict[str, Any]:
    """转成 providers.yaml 的条目。失败项按最保守值写入（§17.4）。"""
    return {
        "tier": r.suggested_tier if r.suggested_tier != "unknown" else "toy",
        "model": r.model,
        "context_window": r.context_window or 8_000,
        "max_output_tokens": r.max_output_tokens or 2_048,
        "supports_json_mode": r.supports_json_mode,
        "supports_function_calling": r.supports_function_calling,
        "supports_temperature": r.supports_temperature,
        "supports_streaming": r.supports_streaming,
        "zh_quality": r.zh_quality if r.zh_quality != "unknown" else "low",
        "tokenizer_hint": "cl100k_base",
        "probed_at": datetime.now(UTC).isoformat(),
    }


def write_config(results: list[LLMProbeResult], path: Path) -> None:
    payload = {
        "# 说明": "由 tss-probe 自动生成（§17）。手改无效 —— 重跑探测以更新。",
        "generated_at": datetime.now(UTC).isoformat(),
        "providers": {r.name: _caps_dict(r) for r in results if r.reachable},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


def render_report(
    llm: list[LLMProbeResult], embed: EmbedProbeResult | None
) -> str:
    lines = [
        "# 供应商能力探测报告",
        "",
        f"生成时间：{datetime.now(UTC).isoformat()}",
        "",
        "> 本报告由 `tss-probe` 生成（路线文档 §17）。",
        "> 探测失败的项按**最保守值**记录 —— 绝不乐观假设（§17.4）。",
        "",
        "## LLM",
        "",
        "| provider | 可达 | 建议档位 | 上下文 | 输出上限 | JSON模式 | JSON稳定性 |"
        " 中文 | 温度 | 流式 | 推理型 | P95 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in llm:
        jsr = f"{r.json_success_rate:.0%}" if r.json_success_rate is not None else "—"
        p95 = f"{r.latency_p95_ms:.0f}ms" if r.latency_p95_ms else "—"
        lines.append(
            f"| {r.name}<br>`{r.model}` | {'✓' if r.reachable else '✗'} "
            f"| **{r.suggested_tier}** | {r.context_window or '—'} "
            f"| {r.max_output_tokens or '—'} | {'✓' if r.supports_json_mode else '✗'} "
            f"| {jsr} | {r.zh_quality} | {'✓' if r.supports_temperature else '✗'} "
            f"| {'✓' if r.supports_streaming else '✗'} "
            f"| {'是' if r.is_reasoning else '否'} | {p95} |"
        )

    for r in llm:
        if not (r.notes or r.error or r.zh_violations):
            continue
        lines += ["", f"### {r.name} 详情", ""]
        if r.error:
            lines.append(f"- **错误**：{r.error}")
        for n in r.notes:
            lines.append(f"- {n}")
        if r.zh_violations:
            lines += ["", "  提问质量违规样例（由 OutputGuard 判定）："]
            lines += [f"  - {v}" for v in r.zh_violations[:5]]

    if embed is not None:
        lines += [
            "",
            "## 嵌入",
            "",
            f"- 模型：`{embed.model}`",
            f"- 可达：{'✓' if embed.reachable else '✗ ' + embed.error}",
            f"- 原生维度：{embed.native_dim or '—'}",
            f"- **支持 dimensions 降维：{'✓' if embed.supports_dimensions_param else '✗'}**"
            "（此项决定 §2.6 走方案 1 还是方案 2）",
            f"- 长文本行为：{embed.long_text_behavior}",
        ]
        for n in embed.notes:
            lines.append(f"- {n}")

    lines += [
        "",
        "## 下一步",
        "",
        "1. 核对上表的建议档位，写入 `config/task_models.yaml`。",
        "2. 若某 provider 的 JSON 稳定性 < 95%，**不要**让它承担 STRONG 任务（§2.4.3）。",
        "3. 若嵌入不支持 dimensions 降维，按 §2.6 方案 2 锁定 Chroma。",
        "4. 换模型或供应商后重跑本脚本，diff `config/providers.yaml` 即知能力变化。",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
async def run(args: argparse.Namespace) -> int:
    settings = get_settings()

    print(_hr("═"))
    print("供应商能力探测（§17）")
    print(_hr("═"))
    if args.quick:
        print("⚠ --quick：跳过多次采样，结果**不足以判定档位**\n")
    else:
        est = JSON_SAMPLES + QUALITY_SAMPLES + LATENCY_SAMPLES + 12
        print(f"约需 {est} 次请求/provider。多次采样项决定档位，不建议省（§17.1 第 9/10 项）\n")

    targets: list[tuple[str, str, str, str]] = []
    if settings.llm_api_key and settings.llm_model_strong:
        targets.append(
            ("cloud_strong", settings.llm_base_url, settings.llm_api_key, settings.llm_model_strong)
        )
    if settings.llm_api_key and settings.llm_model_fast:
        targets.append(
            ("cloud_fast", settings.llm_base_url, settings.llm_api_key, settings.llm_model_fast)
        )
    if settings.has_fallback_provider:
        targets.append(
            (
                "cloud_backup",
                settings.llm_fallback_base_url,
                settings.llm_fallback_api_key,
                settings.llm_fallback_model,
            )
        )
    if settings.local_llm_model and not args.no_local:
        targets.append(
            ("local", settings.local_llm_base_url, "", settings.local_llm_model)
        )

    if not targets:
        print("✗ 没有可探测的 provider。请先配置 .env（参考 .env.example）")
        return 1

    if args.dry_run:
        print("将要探测：")
        for name, base, _, model in targets:
            print(f"  · {name:<14} {model:<28} @ {base}")
        if not args.llm_only:
            print(f"  · embeddings    {settings.embedding_model} @ {settings.embed_base_url}")
        return 0

    results: list[LLMProbeResult] = []
    for name, base, key, model in targets:
        print(f"→ 探测 {name}（{model}）…", flush=True)
        r = await probe_llm(
            name=name,
            base_url=base,
            api_key=key,
            model=model,
            quick=args.quick,
            timeout=settings.llm_timeout_seconds,
        )
        results.append(r)
        mark = "✓" if r.reachable else "✗"
        print(
            f"  {mark} 建议档位={r.suggested_tier} "
            f"json={r.json_success_rate} 中文={r.zh_quality}"
        )
        if r.error:
            print(f"    错误：{r.error[:160]}")

    embed: EmbedProbeResult | None = None
    if not args.llm_only:
        print("→ 探测 embeddings…", flush=True)
        embed = await probe_embeddings(settings)
        print(
            f"  {'✓' if embed.reachable else '✗'} 维度={embed.native_dim} "
            f"dimensions参数={'支持' if embed.supports_dimensions_param else '不支持'}"
        )

    cfg = Path(args.out or settings.providers_file)
    write_config(results, cfg)
    print(f"\n✓ 已写入 {cfg}")

    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_report(results, embed), encoding="utf-8")
    print(f"✓ 报告已写入 {report}")

    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "llm": [asdict(r) for r in results],
                    "embeddings": asdict(embed) if embed else None,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"✓ 原始数据已写入 {args.json}")

    strong = [r for r in results if r.suggested_tier == "strong"]
    if not strong:
        print("\n⚠ 没有任何 provider 达到 STRONG 档 ——")
        print("  Planner/Interviewer/Analyst/Critic 将被能力门控拒绝（§2.4.3）。")
        print("  这是设计而非 bug：宁可显式失败，也不让用户拿到劣质心理分析。")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tss-probe", description="供应商能力探测（路线文档 §17）"
    )
    p.add_argument("--quick", action="store_true", help="跳过多次采样（省 token，不足以定档位）")
    p.add_argument("--llm-only", action="store_true", help="只探 LLM，跳过嵌入")
    p.add_argument("--no-local", action="store_true", help="跳过本地模型")
    p.add_argument("--dry-run", action="store_true", help="只打印计划，不发请求")
    p.add_argument("--out", help="providers.yaml 输出路径（默认取配置）")
    p.add_argument("--report", default="docs/PROBE_REPORT.md", help="markdown 报告路径")
    p.add_argument("--json", help="额外导出原始探测数据")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        code: int = asyncio.run(run(args))
        return code
    except KeyboardInterrupt:
        print("\n已中断。")
        return 130


if __name__ == "__main__":
    sys.exit(main())
