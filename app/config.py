from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_name: str = "lotr-rag"
    debug: bool = False

    # Step 4/10 - embedding model; set to "text-embedding-3-small" to use OpenAI
    embedding_model: str = "all-MiniLM-L6-v2"  # override in .env: EMBEDDING_MODEL=text-embedding-3-small

    # Step 4 - Chroma storage path
    chroma_path: str = "./chroma_db"

    # Vector store backend: "chroma" (local) or "qdrant" (production)
    vector_store: str = "chroma"
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    # Must match the embedding model's output dimension
    embedding_dim: int = 384  # all-MiniLM-L6-v2=384, text-embedding-3-small=1536

    # Step 6 - generation
    anthropic_api_key: str = ""

    # Step 10 - OpenAI API key (only needed when embedding_model = text-embedding-*)
    openai_api_key: str = ""


settings = Settings()
