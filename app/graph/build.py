"""状态机装配（路线文档 §4.2 / §4.3 / §4.5）。

两个关键设计：

1. **按 `phase` 做条件入口**（`set_conditional_entry_point`）。
   会话可能在任意阶段被中断后恢复（§4.5 暂停/恢复），
   固定入口会让恢复时从头重跑 —— 那会重复提问甚至重复计费。

2. **每轮跑到 `turn_done` 就停**。
   一轮 = 一次用户输入到下一次等待输入之间，
   而不是"跑完整个 10 问流程"。这是可恢复的前提。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from langgraph.graph import END, StateGraph

from app.graph.deps import Deps
from app.graph.nodes import analyze as analyze_nodes
from app.graph.nodes.ask import ask
from app.graph.nodes.consent import consent_gate
from app.graph.nodes.intake import intake
from app.graph.nodes.listen import listen
from app.graph.nodes.plan import plan
from app.graph.nodes.probe import probe
from app.graph.state import GraphState
from app.schemas.enums import Phase

NodeFn = Callable[[GraphState, Deps], Awaitable[dict[str, Any]]]


class BoundNode(Protocol):
    """已绑定 deps 的节点，签名与 langgraph 的节点协议对齐。

    不能用 `Callable[[GraphState], Awaitable[...]]` 代替：
    那是**仅位置参数**，而 langgraph 把 `state` 声明为可按名传入，
    因此 Callable 形式不满足 `add_node` 的任何重载。
    """

    def __call__(self, state: GraphState) -> Awaitable[dict[str, Any]]: ...

#: phase → 节点名。恢复会话时据此决定从哪里继续。
_PHASE_ENTRY: dict[Phase, str] = {
    Phase.INTAKE: "intake",
    Phase.CONSENT: "consent",
    Phase.PLAN: "plan",
    Phase.ASK: "ask",
    Phase.LISTEN: "listen",
    Phase.PROBE: "probe",
    Phase.ANALYZE: "analyze",
    Phase.CRITIC: "critic",
    Phase.REPORT: "report",
    Phase.CRISIS: "crisis",
}


def _bind(fn: NodeFn, deps: Deps) -> BoundNode:
    async def wrapped(state: GraphState) -> dict[str, Any]:
        return await fn(state, deps)

    return wrapped


def _entry(state: GraphState) -> str:
    """条件入口：按当前 phase 决定起点。

    PAUSED 的语义是"上一轮因故停下"，恢复时从 `plan` 重新尝试 ——
    因为导致暂停的原因（能力不足、LLM 不可用）可能已经消失。
    """
    if state.phase in (Phase.DONE,):
        return END
    if state.phase is Phase.PAUSED:
        return "plan" if state.plan is not None else "intake"
    return _PHASE_ENTRY.get(state.phase, "intake")


def _after(node: str) -> Callable[[GraphState], str]:
    """节点后的路由：本轮已终止则结束，否则按新 phase 继续。"""

    def route(state: GraphState) -> str:
        if state.turn_done or state.phase in (Phase.DONE, Phase.PAUSED):
            return END
        target = _PHASE_ENTRY.get(state.phase, END)
        # 防自环：同一节点连续路由到自己会死循环
        return END if target == node else target

    return route


def build_graph(deps: Deps, *, checkpointer: Any | None = None) -> Any:
    """装配并编译图。`checkpointer` 用于会话持久化（§4.2）。"""
    # StateGraph 的 4 个类型参数：StateT / ContextT / InputT / OutputT。
    # 我们只约束状态类型，其余交给 langgraph 推断。
    graph: StateGraph[GraphState, Any, Any, Any] = StateGraph(GraphState)

    nodes: dict[str, NodeFn] = {
        "intake": intake,
        "consent": consent_gate,
        "plan": plan,
        "ask": ask,
        "listen": listen,
        "probe": probe,
        "analyze": analyze_nodes.analyze,
        "critic": analyze_nodes.critic,
        "report": analyze_nodes.report,
        "crisis": analyze_nodes.crisis,
    }

    for name, fn in nodes.items():
        graph.add_node(name, _bind(fn, deps))

    graph.set_conditional_entry_point(
        _entry,
        {**{name: name for name in nodes}, END: END},
    )

    for name in nodes:
        graph.add_conditional_edges(
            name,
            _after(name),
            {**{n: n for n in nodes if n != name}, END: END},
        )

    return graph.compile(checkpointer=checkpointer)
