# backend/deps.py

from functools import lru_cache
from pydantic import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str | None = None
    # openai_model: str = "gpt-4o-mini"
    s3_base: str = "s3://kumo-public-datasets/hm_with_images/"
    # demo_mode: bool = True # for dev/demo
    backend_url: str = "http://localhost:8000"

    class Config:
        env_file = ".env"
        env_prefix = ""
        case_sensitive = False

@lru_cache()
def get_settings() -> Settings:
    return Settings()
