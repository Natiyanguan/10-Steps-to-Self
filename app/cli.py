"""命令行入口（路线文档 §13.3 P0：CLI 交互）。

三个子命令：
  `doctor` —— 启动自检 + 能力矩阵，**接真实模型前先跑这个**
  `chat`   —— 交互式会话（`--fake` 走 FakeLLM，完全离线免费）
  `prompts`—— 列出已注册的提示词版本

为什么 P0 先做 CLI 而不是 HTTP API：
CLI 能在没有数据库、没有 Redis、没有前端的情况下验证完整状态机，
把"流程对不对"与"服务能不能跑"两类问题分开排查。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import ConfigError, RuntimeProfile, Settings, get_settings, run_startup_checks
from app.domain.dimension import TOTAL_QUESTIONS
from app.domain.mode import AskMode
from app.graph.build import build_graph
from app.graph.deps import Deps
from app.graph.state import GraphState
from app.llm import capability as cap
from app.llm.base import TaskKind
from app.llm.gateway import LLMGateway, Route
from app.safety.resources import load_crisis_resources
from app.schemas.enums import ConsentScope, Phase, Sharpness
from app.schemas.session import Consent

# Windows 终端默认 GBK，中文输出会乱码
if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _hr(char: str = "─", width: int = 64) -> str:
    return char * width


def _build_gateway(settings: Settings, *, fake: bool) -> LLMGateway:
    if fake:
        # DemoProvider 而非 FakeProvider：后者默认返回 `{}`，会 schema 校验失败。
        # Demo 从提示词里解析约束后产出**结构合法**的响应，
        # 因此 --fake 验证的是真实的图流转与校验链路，只是内容无意义（§2.5）。
        from app.llm.providers.demo import DemoProvider

        return LLMGateway(routes=[Route(provider=DemoProvider(), label="primary")])
    from app.llm.providers.registry import build_gateway

    return build_gateway(settings)


# --------------------------------------------------------------------------
# doctor
# --------------------------------------------------------------------------
async def cmd_doctor(args: argparse.Namespace) -> int:
    print(_hr("═"))
    print("10-Steps-to-Self 启动自检")
    print(_hr("═"))

    try:
        settings = get_settings()
    except Exception as exc:
        print(f"✗ 配置加载失败：{exc}")
        return 1

    print(f"运行档位     : {settings.runtime_profile.value}")
    print(f"LLM base_url : {settings.llm_base_url or '(未配置)'}")
    print(f"强档模型     : {settings.llm_model_strong or '(未配置)'}")
    print(f"快档模型     : {settings.llm_model_fast or '(未配置)'}")
    print(f"本地模型     : {settings.local_llm_model or '(未配置)'} [{settings.local_llm_tier}]")
    print(f"嵌入模型     : {settings.embedding_model} "
          f"({settings.embed_dim_native}→{settings.embed_target_dim} 维)")
    print(f"向量后端     : {settings.vector_backend.value}")
    print(f"用户嵌入本地 : {settings.user_embeddings_local}")
    print()

    try:
        warnings = run_startup_checks(settings)
    except ConfigError as exc:
        print(f"✗ 自检失败（拒绝启动）：\n{exc}")
        return 1

    print("✓ 启动自检通过")
    if warnings:
        print(f"\n{len(warnings)} 条警告：")
        for w in warnings:
            print(f"  ⚠ {w}")

    crisis = load_crisis_resources(settings.crisis_resources_file)
    print(f"\n✓ 危机资源 v{crisis.version}，覆盖地区：{sorted(crisis.regions)}")
    for w in crisis.warnings:
        print(f"  ⚠ {w}")

    print(f"\n{_hr()}\n能力要求矩阵（§2.4.1）\n{_hr()}")
    print(cap.describe_requirements())

    if args.probe:
        print(f"\n{_hr()}\nprovider 连通性探测\n{_hr()}")
        gw = _build_gateway(settings, fake=False)
        for task in (TaskKind.PLAN, TaskKind.ASK, TaskKind.SAFETY, TaskKind.ANALYZE):
            routes = gw.eligible_routes(task)
            names = [r.provider.name for r in routes] or ["(无)"]
            mark = "✓" if routes else "✗"
            print(f"  {mark} {task.value:<14} → {', '.join(names)}")
        if gw.capability_rejections:
            print("\n  被门控拒绝的路由（§2.4.3，这是设计而非 bug）：")
            seen: set[tuple[str, str]] = set()
            for t, name, why in gw.capability_rejections:
                key = (t.value, name)
                if key in seen:
                    continue
                seen.add(key)
                print(f"    · {t.value} / {name}：{why[:90]}")
        await gw.aclose()

    return 0


# --------------------------------------------------------------------------
# chat
# --------------------------------------------------------------------------
def _print_emissions(state: GraphState, shown: int) -> int:
    """打印本轮新增事件，返回新的已打印计数。"""
    for em in state.emissions[shown:]:
        data = em.data
        if em.event == "question":
            tag = "追问" if data.get("is_probe") else f"第 {data['index']}/{TOTAL_QUESTIONS} 问"
            print(f"\n【{tag}】{data['text']}")
        elif em.event == "consent_required":
            print(f"\n{_hr()}\n{data.get('disclaimer', '')}\n{_hr()}")
            print(f"条款版本：{data.get('policy_version')}")
            if data.get("reason"):
                print(f"需要重新确认：{data['reason']}")
        elif em.event == "crisis_stopped":
            print(f"\n{_hr('━')}")
            print(data["text"])
            print(_hr("━"))
        elif em.event == "crisis_resources":
            print(f"\n[关怀提示]\n{data['text']}")
        elif em.event == "degraded":
            print(f"\n[降级] {data.get('reason')}：{data.get('impact', '')}")
        elif em.event == "error":
            print(f"\n[错误 {data.get('code')}] {data.get('message')}")
            if data.get("detail"):
                print(f"  详情：{str(data['detail'])[:300]}")
        elif em.event == "report":
            _print_report(data)
    return len(state.emissions)


def _print_report(data: dict[str, Any]) -> None:
    r = data["payload"]
    print(f"\n{_hr('═')}\n复盘报告\n{_hr('═')}")
    if data.get("degraded"):
        print("⚠ 本报告在生成过程中发生降级，部分内容可能不完整\n")

    print(f"【你描述的困惑】\n  {r['surface_concern']}\n")

    if r["underlying_needs"]:
        print("【可能的深层诉求】")
        for i, ins in enumerate(r["underlying_needs"]):
            print(f"  {i}. {ins['statement']}（置信度 {ins['confidence']}）")
            for ev in ins["evidence"]:
                print(f"     依据 · 第{ev['question_index']}问：「{ev['quote']}」")
        print()

    if r["inner_conflicts"]:
        print("【内在冲突】")
        offset = len(r["underlying_needs"])
        for i, ins in enumerate(r["inner_conflicts"]):
            print(f"  {offset + i}. {ins['statement']}")
            for ev in ins["evidence"]:
                print(f"     依据 · 第{ev['question_index']}问：「{ev['quote']}」")
        print()

    if r["suggestions"]:
        print("【可以试试的下一步】")
        for s in r["suggestions"]:
            print(f"  · {s['action']}")
            print(f"    24h 内第一步：{s['first_step']}")
        print()

    if r["uncertainty_notes"]:
        print("【需要说明的地方】")
        for n in r["uncertainty_notes"]:
            print(f"  · {n}")
        print()

    print(_hr())
    print(data.get("disclaimer", ""))
    print(_hr())


async def cmd_chat(args: argparse.Namespace) -> int:
    try:
        settings = get_settings()
        if not args.fake:
            for w in run_startup_checks(settings):
                print(f"⚠ {w}")
    except ConfigError as exc:
        print(f"✗ 启动自检失败：{exc}")
        print("  提示：加 --fake 可在无凭据的情况下用假模型跑通流程")
        return 1

    if args.fake:
        settings = settings.model_copy(update={"runtime_profile": RuntimeProfile.LOCAL})

    gateway = _build_gateway(settings, fake=args.fake)
    deps = Deps.build(settings=settings, gateway=gateway)
    graph = build_graph(deps)

    print(_hr("═"))
    print("自我探索会话" + ("（假模型模式：仅验证流程，输出无意义）" if args.fake else ""))
    print(_hr("═"))
    print(deps.crisis.disclaimer)
    print(_hr())

    # ASYNC250：单用户 CLI 就该阻塞等输入 —— 此时没有其他协程需要推进，
    # 换成 to_thread 只会让 Ctrl-C 的处理更难写，不带来任何并发收益。
    concern = args.concern or input("\n你想梳理的困惑是什么？\n> ").strip()  # noqa: ASYNC250
    if not concern:
        print("没有输入，退出。")
        return 0

    state = GraphState(
        session_id=f"cli-{datetime.now(UTC):%Y%m%d%H%M%S}",
        user_id="cli-user",
        mode=AskMode(args.mode),
        sharpness=Sharpness[args.sharpness.upper()],
        is_minor=args.minor,
        consent=Consent(
            policy_version=settings.policy_version,
            granted_at=datetime.now(UTC),
            scopes={ConsentScope.STORE_MEMORY},
        ),
        inbox=concern,
    )

    shown = 0
    try:
        for _ in range(TOTAL_QUESTIONS * 4 + 10):
            raw = await graph.ainvoke(state)
            state = GraphState.model_validate(raw)
            shown = _print_emissions(state, shown)

            if state.phase is Phase.DONE:
                break
            if state.phase is Phase.CRISIS:
                print("\n会话已安全终止。请照顾好自己。")
                break
            if state.phase is Phase.PAUSED:
                print("\n会话已暂停并保留，可稍后继续。")
                break

            if state.open_questions:
                try:
                    reply = input("\n> ").strip()  # noqa: ASYNC250
                except (EOFError, KeyboardInterrupt):
                    print("\n已中断，会话状态保留。")
                    break
                if reply.lower() in ("/quit", "/exit"):
                    print("已退出，会话状态保留。")
                    break
                if not reply:
                    print("（空输入，请再说几句）")
                    continue
                state = state.model_copy(update={"inbox": reply, "turn_done": False})
                continue

            if not state.inbox:
                break
    finally:
        p_in, p_out = gateway.total_usage()
        print(f"\n[计量] 调用 {len(gateway.records)} 次，"
              f"输入 {p_in} tokens，输出 {p_out} tokens")
        if state.degrade_reasons:
            print(f"[降级] {[r.value for r in state.degrade_reasons]}")
        await gateway.aclose()

    if args.dump:
        # ASYNC240：会话结束时的一次性导出，此刻已无并发工作可推进。
        Path(args.dump).write_text(  # noqa: ASYNC240
            json.dumps(state.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[已导出会话状态] {args.dump}")
    return 0


async def cmd_prompts(args: argparse.Namespace) -> int:
    from app.prompts import PromptRegistry

    settings = get_settings()
    registry = PromptRegistry(settings.prompts_dir)
    available = registry.available()
    if not available:
        print(f"✗ {settings.prompts_dir} 下没有找到提示词模板")
        return 1
    print(f"提示词目录：{settings.prompts_dir}")
    for agent, versions in available.items():
        print(f"  {agent:<14} {', '.join(versions)}")
    return 0


# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tss",
        description="10-Steps-to-Self：苏格拉底式自我探索 Agent",
    )
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("doctor", help="启动自检与能力矩阵（接真实模型前先跑）")
    d.add_argument("--probe", action="store_true", help="额外做 provider 路由探测")
    d.set_defaults(func=cmd_doctor)

    c = sub.add_parser("chat", help="交互式会话")
    c.add_argument("--fake", action="store_true", help="用 FakeLLM，完全离线免费")
    c.add_argument("--concern", help="直接给出困惑，跳过首次输入")
    c.add_argument("--mode", choices=[m.value for m in AskMode], default=AskMode.BATCHED.value)
    c.add_argument("--sharpness", choices=["gentle", "moderate", "sharp"], default="moderate")
    c.add_argument("--minor", action="store_true", help="标记为未成年人（强制温和模式）")
    c.add_argument("--dump", help="结束后把会话状态导出到该文件")
    c.set_defaults(func=cmd_chat)

    pr = sub.add_parser("prompts", help="列出提示词版本")
    pr.set_defaults(func=cmd_prompts)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        exit_code: int = asyncio.run(args.func(args))
        return exit_code
    except KeyboardInterrupt:
        print("\n已中断。")
        return 130


if __name__ == "__main__":
    sys.exit(main())
