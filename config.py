from dotenv import load_dotenv
from pydantic_settings import BaseSettings


load_dotenv()


class Config(BaseSettings):
    model: str
    embedding_length: int
    minimum_similarity: float
    index_name: str
    token_target: int
    max_tokens: int
    opensearch_host: str
    opensearch_port: int
    opensearch_username: str
    opensearch_password: str


config = Config()
