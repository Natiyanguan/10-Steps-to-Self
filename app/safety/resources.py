"""危机资源加载（路线文档 §3.4 / §16.2）。

硬约束：危机资源信息**只从配置读取，绝不由模型生成**。
模型编造错误的求助电话，后果不可接受。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

VERIFY_STALE_DAYS = 180  # 超过此天数未人工核验则告警


@dataclass(frozen=True, slots=True)
class Hotline:
    name: str
    number: str
    hours: str = ""
    note: str = ""


@dataclass(frozen=True, slots=True)
class RegionResources:
    key: str
    label: str
    hotlines: tuple[Hotline, ...]
    emergency: tuple[Hotline, ...]
    guidance: str = ""


@dataclass(frozen=True, slots=True)
class CrisisResources:
    version: str
    disclaimer: str
    crisis_message: str
    regions: dict[str, RegionResources]
    default_region: str
    last_verified: date | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def region(self, key: str | None = None) -> RegionResources:
        chosen = key or self.default_region
        if chosen not in self.regions:
            chosen = self.default_region
        return self.regions[chosen]

    def render(self, region_key: str | None = None) -> str:
        """渲染给用户的危机资源文本。

        纯字符串拼接，**不经过任何模型** —— 这是本函数存在的全部理由。
        """
        r = self.region(region_key)
        lines: list[str] = [self.crisis_message.strip(), ""]

        if r.hotlines:
            lines.append(f"【{r.label}｜心理支持热线】")
            for h in r.hotlines:
                suffix = f"（{h.hours}）" if h.hours else ""
                lines.append(f"  · {h.name}：{h.number}{suffix}")
            lines.append("")

        if r.emergency:
            lines.append("【紧急情况】")
            for h in r.emergency:
                lines.append(f"  · {h.name}：{h.number}")
            lines.append("")

        if r.guidance:
            lines.append(r.guidance.strip())
            lines.append("")

        lines.append(self.disclaimer.strip())
        return "\n".join(lines).strip()


def _parse_hotlines(raw: Any) -> tuple[Hotline, ...]:
    if not raw:
        return ()
    out: list[Hotline] = []
    for item in raw:
        if not isinstance(item, dict) or "name" not in item or "number" not in item:
            raise ValueError(f"热线条目必须含 name 与 number：{item!r}")
        out.append(
            Hotline(
                name=str(item["name"]),
                number=str(item["number"]),
                hours=str(item.get("hours", "")),
                note=str(item.get("note", "")),
            )
        )
    return tuple(out)


def load_crisis_resources(path: Path) -> CrisisResources:
    """加载并校验危机资源。任何结构问题都抛异常 —— 启动自检会因此拒绝启动。"""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("危机资源文件根节点必须是映射")

    raw_regions = data.get("regions")
    if not isinstance(raw_regions, dict) or not raw_regions:
        raise ValueError("regions 不能为空：没有任何求助渠道时不允许启动")

    regions: dict[str, RegionResources] = {}
    for key, val in raw_regions.items():
        if not isinstance(val, dict):
            raise ValueError(f"region {key} 必须是映射")
        hotlines = _parse_hotlines(val.get("hotlines"))
        emergency = _parse_hotlines(val.get("emergency"))
        if not hotlines and not emergency:
            raise ValueError(f"region {key} 至少需要一条热线或紧急联系方式")
        regions[str(key)] = RegionResources(
            key=str(key),
            label=str(val.get("label", key)),
            hotlines=hotlines,
            emergency=emergency,
            guidance=str(val.get("guidance", "")),
        )

    default_region = str(data.get("default_region", next(iter(regions))))
    if default_region not in regions:
        raise ValueError(f"default_region={default_region} 不在 regions 中")

    disclaimer = str(data.get("disclaimer", "")).strip()
    if not disclaimer:
        raise ValueError("disclaimer 不能为空（§3.1 产品定位声明）")

    warnings: list[str] = []
    last_verified: date | None = None
    raw_verified = data.get("last_verified")
    if raw_verified:
        last_verified = (
            raw_verified
            if isinstance(raw_verified, date)
            else datetime.fromisoformat(str(raw_verified)).date()
        )
        age = (datetime.now(UTC).date() - last_verified).days
        if age > VERIFY_STALE_DAYS:
            warnings.append(
                f"危机资源已 {age} 天未人工核验（阈值 {VERIFY_STALE_DAYS} 天）："
                " 号码可能失效，需重新核对（§3.8）"
            )
    else:
        warnings.append("危机资源缺少 last_verified：无法判断号码是否仍然有效")

    return CrisisResources(
        version=str(data.get("version", "unknown")),
        disclaimer=disclaimer,
        crisis_message=str(data.get("crisis_message", "")).strip() or disclaimer,
        regions=regions,
        default_region=default_region,
        last_verified=last_verified,
        warnings=tuple(warnings),
    )


@lru_cache(maxsize=4)
def load_cached(path: Path) -> CrisisResources:
    return load_crisis_resources(path)
