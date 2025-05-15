from pydantic_settings import BaseSettings
from pydantic import computed_field


class ToolConfig:
    env_file_encoding = "utf8"
    extra = "ignore"


class NWSettings(BaseSettings):
    OPENAI_KEY: str = ""
    PROXY: str = ""
    MODEL_NAME: str = "gpt-4o"

    class Config(ToolConfig):
        env_prefix = "nw_"


class RedisSettings(BaseSettings):
    HOST: str = "redis"
    PORT: int = 6379
    CELERY_DB: int = 1

    @computed_field(return_type=str)
    @property
    def URI(self):
        return f"redis://{self.HOST}:{self.PORT}/{self.CELERY_DB}"

    class Config(ToolConfig):
        env_prefix = "redis_"


class RabbitMQSettings(BaseSettings):
    HOST: str = "rabbitmq"
    AMQP_PORT: int = 5672
    LOGIN: str = "admin"
    PASSWORD: str = "admin"

    @computed_field(return_type=str)
    @property
    def URI(self):
        return f"amqp://{self.LOGIN}:{self.PASSWORD}@{self.HOST}:{self.AMQP_PORT}/"

    class Config(ToolConfig):
        env_prefix = "rabbitmq_"


class MinIOSettings(BaseSettings):
    HOST: str = "minio"
    BUCKET: str = "user_docs"
    API_PORT: int = 9000
    ROOT_USER: str = "admin"
    ROOT_PASSWORD: str = "admin_password"

    @computed_field(return_type=str)
    @property
    def URI(self):
        return f"http://{self.HOST}:{self.API_PORT}"

    class Config(ToolConfig):
        env_prefix = "minio_"


class PostgresSettings(BaseSettings):
    HOST: str = "localhost"
    PORT: int = 5432
    USER: str = "postgres"
    PASSWORD: str = "passwd"
    DB: str = "ds"
    MAX_OVERFLOW: int = 15
    POOL_SIZE: int = 15

    @computed_field(return_type=str)
    @property
    def URI(self):
        return f"postgresql://{self.USER}:{self.PASSWORD}@{self.HOST}:{self.PORT}/{self.DB}"

    class Config(ToolConfig):
        env_prefix = "postgres_"


postgres_settings = PostgresSettings()
app_settings = NWSettings()
redis_settings = RedisSettings()
minio_settings = MinIOSettings()
rabbitmq_settings = RabbitMQSettings()
