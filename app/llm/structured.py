"""结构化输出与容错解析（路线文档 §6.2 / 决策 #20）。

主路径是 **JSON 模式**而非 function calling（决策 #20）：
DeepSeek 系与本地模型对 function calling 的支持程度不一致，
推理型模型常不支持工具调用。JSON 模式的兼容面明显更宽。

三层容错，**顺序不可颠倒**：
  1. 直接 `json.loads`
  2. 容错修复：剥 markdown 围栏、截取首个完整 JSON、去尾逗号、补引号
  3. 带错误信息重试（最多 2 次）

三次失败后**显式失败**（`StructuredOutputError`），
绝不猜测填充默认值继续跑 —— 那会把解析问题伪装成内容问题（§6.2）。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from app.llm.base import StructuredOutputError

MAX_PARSE_ATTEMPTS = 3

_FENCE = re.compile(r"^\s*```(?:json|JSON)?\s*|\s*```\s*$", re.MULTILINE)
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")
# 推理型模型可能把思维链放在 <think> 标签里
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_wrappers(text: str) -> str:
    s = _THINK_BLOCK.sub("", text)
    s = _FENCE.sub("", s)
    return s.strip()


def _extract_first_json(text: str) -> str | None:
    """从混杂文本中截取第一个完整的 JSON 对象/数组。

    用括号配对而非正则：正则无法正确处理嵌套。
    同时跳过字符串内的括号与转义字符 —— 用户原话里可能就带 `{`。
    """
    start = None
    for i, ch in enumerate(text):
        if ch in "{[":
            start = i
            break
    if start is None:
        return None

    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escaped = False

    for i in range(start, len(text)):
        ch = text[i]
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


@dataclass(frozen=True, slots=True)
class ParseAttempt:
    ok: bool
    data: Any = None
    error: str = ""
    repaired: bool = False


def parse_json_lenient(text: str) -> ParseAttempt:
    """两层解析：直接解析 → 容错修复。纯函数，可完全离线单测。"""
    if not text or not text.strip():
        return ParseAttempt(False, error="响应为空")

    try:
        return ParseAttempt(True, data=json.loads(text))
    except json.JSONDecodeError:
        pass

    candidate = _strip_wrappers(text)
    last = "无可用的修复候选（响应中未找到 JSON 结构）"
    for transform in (
        lambda s: s,
        lambda s: _extract_first_json(s) or "",
        lambda s: _TRAILING_COMMA.sub(r"\1", _extract_first_json(s) or ""),
    ):
        try:
            fixed = transform(candidate)
        except Exception:
            continue
        if not fixed:
            continue
        try:
            return ParseAttempt(True, data=json.loads(fixed), repaired=True)
        except json.JSONDecodeError as exc:
            last = str(exc)

    return ParseAttempt(False, error=f"容错修复后仍无法解析：{last}")


def _format_validation_error(exc: ValidationError, model: type[BaseModel]) -> str:
    """把 pydantic 错误转成给模型看的、可操作的反馈。

    关键：告诉它**哪个字段**错了、**期望什么**，而不是丢一段 traceback。
    模糊的反馈换不来正确的重试。
    """
    parts: list[str] = []
    for err in exc.errors()[:8]:
        loc = ".".join(str(x) for x in err["loc"]) or "(根)"
        parts.append(f"字段 `{loc}`：{err['msg']}")
    return f"不符合 {model.__name__} 的结构要求 —— " + "；".join(parts)


def validate_into[T: BaseModel](raw: Any, model: type[T]) -> tuple[T | None, str]:
    """用 pydantic 校验。返回 (实例, 错误说明)。"""
    try:
        return model.model_validate(raw), ""
    except ValidationError as exc:
        return None, _format_validation_error(exc, model)


def build_retry_hint(error: str, model: type[BaseModel]) -> str:
    """构造重试提示（§6.2 第 3 层）—— 必须包含具体错误，否则重试是盲猜。"""
    return (
        "你上一次的输出无法使用，原因如下：\n"
        f"{error}\n\n"
        f"请重新输出**严格符合 {model.__name__} 结构**的 JSON。硬要求：\n"
        "  1. 只输出 JSON 本体，不要 markdown 围栏、不要任何解释文字\n"
        "  2. 不要添加结构中未声明的字段\n"
        "  3. 所有字符串用双引号，不要有尾逗号\n"
        "  4. 如果某项你无法确定，按结构要求给出合法的空值，不要编造内容"
    )


def parse_or_raise[T: BaseModel](
    text: str, model: type[T], *, attempt: int = MAX_PARSE_ATTEMPTS
) -> T:
    """单次解析并校验，失败即抛。供无重试通道的场景使用。"""
    parsed = parse_json_lenient(text)
    if not parsed.ok:
        raise StructuredOutputError(
            f"结构化输出解析失败（第 {attempt} 次）：{parsed.error}", raw_text=text[:1000]
        )
    obj, err = validate_into(parsed.data, model)
    if obj is None:
        raise StructuredOutputError(
            f"结构化输出校验失败（第 {attempt} 次）：{err}", raw_text=text[:1000]
        )
    return obj


def json_schema_hint(model: type[BaseModel]) -> str:
    """生成给提示词用的 schema 说明。

    用 pydantic 自带的 JSON Schema，避免手写一份再和代码脱节 ——
    手写副本一定会在某次改字段后忘记同步。
    """
    schema = model.model_json_schema()
    return json.dumps(schema, ensure_ascii=False, indent=2)
