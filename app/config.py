"""应用配置：pydantic-settings 读取 .env。

对应 docs/technical-design.md 附录 A：字段名与默认值完全一致。
- DEEPSEEK_API_KEY / MP_API_KEY 必填（min_length=1），缺失或为空时实例化即报错，不静默回退。
- extra="ignore"：容忍 .env 中的无关键（如 Claude Code 全局配置遗留的键）。
"""
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_ENV: str = "dev"
    DATA_DIR: str = "./data"
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/chem-agent.db"
    CHROMA_DIR: str = "./data/chroma"
    DEEPSEEK_API_KEY: str = Field(min_length=1)
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com/v1"
    DEEPSEEK_MODEL: str = "deepseek-chat"
    MP_API_KEY: str = Field(min_length=1)
    EMBEDDING_MODEL: str = "paraphrase-multilingual-MiniLM-L12-v2"
    HF_ENDPOINT: str | None = None
    LLM_TIMEOUT_SECONDS: int = 120
    MP_TIMEOUT_SECONDS: int = 30
    MAX_PDF_MB: int = 50
    MAX_STRUCTURE_MB: int = 10
    SESSION_TTL_HOURS: int = 168
    CORS_ORIGINS: str = "*"
    LOG_LEVEL: str = "INFO"


settings = Settings()
