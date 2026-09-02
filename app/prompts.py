"""提示词注册与渲染（路线文档 §6.1）。

硬约束（§16.2）：**提示词不硬编码在 Python 中**，全部走 `prompts/` + 本注册器。
切换版本 = 改配置，无需改代码、无需重新部署。

渲染后记录 hash 到 trace —— 问题复现时能定位到确切版本。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateNotFound


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    text: str
    agent: str
    version: str
    sha: str  # 前 12 位，进 trace 与日志（§6.1）

    @property
    def ref(self) -> str:
        return f"{self.agent}/{self.version}@{self.sha}"


class PromptError(RuntimeError):
    pass


class PromptRegistry:
    """按 `prompts/{agent}/{version}.jinja2` 加载模板。

    用 `StrictUndefined`：模板里引用了未传的变量应**立即报错**，
    而不是渲染成空字符串。静默渲染出缺字段的提示词，
    会让模型收到残缺指令却照样返回看似正常的结果 —— 属于最难查的一类问题。
    """

    def __init__(self, root: Path, *, default_version: str = "v1") -> None:
        self._root = root
        self._default_version = default_version
        self._env = Environment(
            loader=FileSystemLoader(str(root)),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=False,
            autoescape=False,  # 输出是给模型的纯文本，不是 HTML
        )

    def render(
        self, agent: str, context: dict[str, Any], *, version: str | None = None
    ) -> RenderedPrompt:
        ver = version or self._default_version
        name = f"{agent}/{ver}.jinja2"
        try:
            template = self._env.get_template(name)
        except TemplateNotFound as exc:
            raise PromptError(
                f"提示词模板不存在：{self._root / name}。"
                " 提示词必须外置（§6.1），不允许回退到代码内硬编码字符串。"
            ) from exc

        text = template.render(**context).strip()
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        return RenderedPrompt(text=text, agent=agent, version=ver, sha=sha)

    def available(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        if not self._root.exists():
            return out
        for agent_dir in sorted(p for p in self._root.iterdir() if p.is_dir()):
            versions = sorted(p.stem for p in agent_dir.glob("*.jinja2"))
            if versions:
                out[agent_dir.name] = versions
        return out


@lru_cache(maxsize=4)
def get_registry(root: Path, default_version: str = "v1") -> PromptRegistry:
    return PromptRegistry(root, default_version=default_version)
