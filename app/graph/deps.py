"""图节点的依赖容器（路线文档 §4.5 / §13.2）。

为什么用显式容器而不是全局单例：
节点需要在测试里被注入 FakeLLM（§12 硬要求 1）。
全局单例会让"同一进程里跑两套配置"变得不可能，
而集成测试正需要这个（cloud 档与 local 档各跑一遍）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.agents.analyst import Analyst
from app.agents.base import AgentContext
from app.agents.critic import Critic
from app.agents.interviewer import Interviewer
from app.agents.planner import Planner
from app.agents.quality import QualityAgent
from app.agents.safety_agent import SafetyAgent
from app.agents.summarizer import Summarizer
from app.config import Settings
from app.llm.gateway import LLMGateway
from app.prompts import PromptRegistry
from app.safety.resources import CrisisResources, load_crisis_resources


@dataclass(slots=True)
class Deps:
    """一次会话运行所需的全部依赖。"""

    settings: Settings
    gateway: LLMGateway
    prompts: PromptRegistry
    crisis: CrisisResources

    planner: Planner
    interviewer: Interviewer
    quality: QualityAgent
    safety: SafetyAgent
    analyst: Analyst
    critic: Critic
    summarizer: Summarizer

    @classmethod
    def build(
        cls,
        *,
        settings: Settings,
        gateway: LLMGateway,
        prompts_root: Path | None = None,
        crisis: CrisisResources | None = None,
    ) -> Deps:
        registry = PromptRegistry(prompts_root or settings.prompts_dir)
        ctx = AgentContext(gateway=gateway, prompts=registry)
        return cls(
            settings=settings,
            gateway=gateway,
            prompts=registry,
            crisis=crisis or load_crisis_resources(settings.crisis_resources_file),
            planner=Planner(ctx),
            interviewer=Interviewer(ctx),
            quality=QualityAgent(ctx),
            safety=SafetyAgent(ctx),
            analyst=Analyst(ctx),
            critic=Critic(ctx),
            summarizer=Summarizer(ctx),
        )
