"""配置与启动自检（路线文档 §13.2 / §13.2.2）。

原则：**启动即校验，缺失快速失败**。
不允许带着空 API key 或缺失危机资源启动后在运行时才炸。
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RuntimeProfile(StrEnum):
    """三档运行档位（§13.2.1）。"""

    CLOUD = "cloud"    # LLM 云端 + 嵌入云端
    HYBRID = "hybrid"  # LLM 云端 + 用户历史嵌入走本地（推荐生产，隐私最优）
    LOCAL = "local"    # 全本地（仅开发/CI/演示，无 STRONG 能力）


class VectorBackend(StrEnum):
    CHROMA = "chroma"
    PGVECTOR = "pgvector"


# pgvector 的 HNSW / IVFFlat 索引维度上限（§2.6）
PGVECTOR_INDEX_MAX_DIM = 2000


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- 运行档位 ----
    runtime_profile: RuntimeProfile = RuntimeProfile.CLOUD

    # ---- LLM（OpenAI 兼容协议）----
    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_model_strong: str = ""
    llm_model_fast: str = ""
    llm_temperature: float = 0.3
    llm_max_tokens: int = 4096
    llm_timeout_seconds: float = 180.0

    # ---- 备用供应商（决策 #24）----
    llm_fallback_base_url: str = ""
    llm_fallback_api_key: str = ""
    llm_fallback_model: str = ""

    # ---- 本地 LLM ----
    local_llm_base_url: str = "http://127.0.0.1:11434/v1"
    local_llm_model: str = ""
    local_llm_tier: str = "toy"

    # ---- Embedding（云）----
    embed_api_key: str = ""
    embed_base_url: str = "https://open.bigmodel.cn/api/paas/v4/"
    embedding_model: str = "embedding-3"
    embed_dim_native: int = 2048
    embed_target_dim: int = 1024
    embed_batch_size: int = 32

    # ---- Embedding（本地）----
    local_embed_model: str = "BAAI/bge-m3"
    local_embed_device: str = "cpu"

    # ---- Reranker ----
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_enabled: bool = False
    reranker_timeout_ms: int = 800

    # ---- 向量库 ----
    vector_backend: VectorBackend = VectorBackend.CHROMA
    chroma_persist_dir: Path = Path("./data/chroma")
    chroma_collection_kb: str = "kb_zhipu1024_v1"
    chroma_collection_user_prefix: str = "mem_"

    # ---- 存储 ----
    database_url: str = "sqlite+aiosqlite:///./data/app.db"
    redis_url: str = ""

    # ---- 知识库构建（离线，§7.7）----
    bocha_api_key: str = ""
    tavily_api_key: str = ""
    openalex_base: str = "https://api.openalex.org"
    openalex_api_key: str = ""
    openalex_daily_credits: int = 10000
    kb_cache_dir: Path = Path("./data/kb_cache")

    # ---- 安全与合规 ----
    policy_version: str = "unset"
    crisis_resources_file: Path = Path("config/crisis_resources.yaml")
    session_retention_days: int = 90
    default_region: str = "CN"

    # ---- 可观测 ----
    sentry_dsn: str = ""
    otel_exporter_otlp_endpoint: str = ""
    log_level: str = "INFO"
    debug_log_raw_text: bool = False

    # ---- 提示词 ----
    prompts_dir: Path = Path("prompts")
    task_models_file: Path = Path("config/task_models.yaml")
    providers_file: Path = Path("config/providers.yaml")

    @field_validator("local_llm_tier")
    @classmethod
    def _tier_never_strong(cls, v: str) -> str:
        """§2.4.3 第 3 条：TOY/本地档绝不允许声明为 strong。硬断言，不读配置绕过。"""
        normalized = v.strip().lower()
        if normalized == "strong":
            raise ValueError(
                "LOCAL_LLM_TIER 不允许为 strong（§2.4.3）：\n"
                "  本地小模型不具备 Planner/Interviewer/Analyst/Critic 所需的能力，\n"
                "  声明为 strong 会让用户拿到格式完整但内容失效的心理分析。\n"
                "  请填 toy（minimind 等）或 fast（指令微调 7B 级模型）。"
            )
        if normalized not in {"toy", "fast"}:
            raise ValueError(f"LOCAL_LLM_TIER 必须是 toy 或 fast，收到 {v!r}")
        return normalized

    @field_validator("llm_temperature")
    @classmethod
    def _temp_range(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError(f"LLM_TEMPERATURE 应在 0..2，收到 {v}")
        return v

    # ---- 派生属性 ----

    @property
    def uses_cloud_llm(self) -> bool:
        return self.runtime_profile is not RuntimeProfile.LOCAL

    @property
    def user_embeddings_local(self) -> bool:
        """user scope 是否走本地嵌入（§3.3.1 隐私最优组合）。"""
        return self.runtime_profile in (RuntimeProfile.HYBRID, RuntimeProfile.LOCAL)

    @property
    def has_fallback_provider(self) -> bool:
        return bool(self.llm_fallback_base_url and self.llm_fallback_model)


class ConfigError(RuntimeError):
    """启动自检失败。故意不可恢复 —— 见 §13.2.2。"""


def run_startup_checks(settings: Settings) -> list[str]:
    """启动自检（§13.2.2）。返回警告列表；致命问题直接抛 ConfigError。"""
    warnings: list[str] = []

    # 1) 云端档位必须有 LLM 凭据与模型标识
    if settings.uses_cloud_llm:
        missing = [
            name
            for name, val in (
                ("LLM_API_KEY", settings.llm_api_key),
                ("LLM_MODEL_STRONG", settings.llm_model_strong),
                ("LLM_MODEL_FAST", settings.llm_model_fast),
            )
            if not val
        ]
        if missing:
            raise ConfigError(
                f"运行档位 {settings.runtime_profile.value} 需要以下配置：{', '.join(missing)}。"
                " 参考 .env.example。"
            )
    elif not settings.local_llm_model:
        raise ConfigError("RUNTIME_PROFILE=local 需要 LOCAL_LLM_MODEL")

    # 2) 向量维度与后端的兼容性（§2.6 —— 这是 v3.0 修订的核心约束）
    if (
        settings.vector_backend is VectorBackend.PGVECTOR
        and settings.embed_target_dim > PGVECTOR_INDEX_MAX_DIM
    ):
        raise ConfigError(
            f"EMBED_TARGET_DIM={settings.embed_target_dim} 超过 pgvector 索引上限"
            f" {PGVECTOR_INDEX_MAX_DIM} 维（§2.6）。\n"
            "  HNSW/IVFFlat 无法为该维度建索引，检索会退化为全表扫描。\n"
            "  三条出路：降到 1024 维 / 改用 VECTOR_BACKEND=chroma / 用 halfvec。"
        )

    # 3) 危机资源必须存在且可解析 —— 缺失时不允许启动（比任何功能都优先，§3.4）
    if not settings.crisis_resources_file.exists():
        raise ConfigError(
            f"危机资源文件不存在：{settings.crisis_resources_file}。\n"
            "  这比任何功能都优先：没有它，危机分支无法给出求助渠道，\n"
            "  而让模型即兴生成求助电话是不可接受的（§3.4 / §16.2）。"
        )

    from app.safety.resources import load_crisis_resources

    try:
        resources = load_crisis_resources(settings.crisis_resources_file)
    except Exception as exc:
        raise ConfigError(f"危机资源文件无法解析：{exc}") from exc

    if settings.default_region not in resources.regions:
        raise ConfigError(
            f"DEFAULT_REGION={settings.default_region} 在危机资源文件中不存在。"
            f" 可用：{sorted(resources.regions)}"
        )

    # 4) 同意条款版本必须显式设置（§3.2 版本化）
    if settings.policy_version == "unset":
        warnings.append("POLICY_VERSION 未设置：同意记录将无法追溯到具体条款版本（§3.2）")

    # 5) 备用供应商缺失 —— 单点风险（决策 #24）
    if settings.uses_cloud_llm and not settings.has_fallback_provider:
        warnings.append(
            "未配置备用供应商（决策 #24）：单厂故障即全站不可用。"
            " STRONG 任务的降级链末端将是显式失败，而不会用弱档顶替。"
        )

    # 6) 隐私提示
    if settings.debug_log_raw_text:
        warnings.append(
            "DEBUG_LOG_RAW_TEXT=true：用户原话会进入日志（§11.1 隐私红线）。"
            " 仅在调试且已获用户授权时开启，生产必须关闭。"
        )
    if settings.runtime_profile is RuntimeProfile.CLOUD:
        warnings.append(
            "cloud 档位下用户历史会送往云端嵌入服务（§3.3）。"
            " 生产建议用 hybrid：user scope 走本地 BGE-M3，最敏感数据不出本机。"
        )

    return warnings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
