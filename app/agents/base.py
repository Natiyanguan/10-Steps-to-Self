"""Agent 共享装配（路线文档 §4.4）。

每个 Agent 只有一个职责，Agent 间**只通过 Pydantic 模型通信，禁止传自由文本**。
这里只提供 gateway + prompt registry 的接线，不含任何业务判断。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel

from app.llm.base import Message, Role, TaskKind
from app.llm.gateway import CallOutcome, LLMGateway
from app.prompts import PromptRegistry, RenderedPrompt

T = TypeVar("T", bound=BaseModel)


@dataclass(slots=True)
class AgentContext:
    """所有 Agent 共享的运行期依赖。"""

    gateway: LLMGateway
    prompts: PromptRegistry


class Agent:
    """Agent 基类：渲染提示词 → 结构化调用 → 返回模型实例 + 降级痕迹。"""

    #: 子类覆盖：提示词目录名
    prompt_agent: str = ""
    #: 子类覆盖：任务类型（决定超时/档位/温度）
    task: TaskKind

    def __init__(self, ctx: AgentContext, *, prompt_version: str = "v1") -> None:
        self._ctx = ctx
        self._version = prompt_version

    @property
    def gateway(self) -> LLMGateway:
        return self._ctx.gateway

    def render(self, context: dict[str, Any], *, version: str | None = None) -> RenderedPrompt:
        return self._ctx.prompts.render(
            self.prompt_agent, context, version=version or self._version
        )

    async def call(
        self,
        prompt: RenderedPrompt,
        model: type[T],
        *,
        user_payload: str = "请开始。",
        max_output_tokens: int | None = None,
    ) -> tuple[T, CallOutcome]:
        """system = 渲染后的提示词，user = 触发语。

        提示词整体放 system 位：内容稳定 → 命中供应商前缀缓存（§9.4），
        且用户数据已在模板内用分隔符包裹（§3.7 sandwich 结构）。
        """
        messages = [
            Message(Role.SYSTEM, prompt.text),
            Message(Role.USER, user_payload),
        ]
        return await self.gateway.complete_structured(
            self.task,
            messages,
            model,
            max_output_tokens=max_output_tokens,
            prompt_version=prompt.ref,
        )
