from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    TG_BOT_TOKEN: str

    RABBITMQ_TOPIC: str
    RABBITMQ_URL: str

    S3_ACCESS_KEY_ID: str
    S3_SECRET_ACCESS_KEY: str
    S3_ENDPOINT_URL: str
    S3_BUCKET: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

settings = Settings()
