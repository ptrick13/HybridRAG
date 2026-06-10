"""Central configuration for HybridRAG.

Parameters are resolved from environment variables (via the .env file).
Defaults are provided for local Docker Compose development; override every
credential before deploying to a shared environment.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings resolved from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────────────────────────
    openai_api_key: str = Field(..., description="OpenAI or Azure OpenAI API key.")
    openai_model: str = Field(default="gpt-4o", description="Chat completion model name.")
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="API base URL. Override for Azure OpenAI.",
    )

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_model: str = Field(
        default="text-embedding-3-large",
        description="OpenAI embedding model used for dense retrieval.",
    )
    embedding_dimensions: int = Field(
        default=3072,
        description="Output dimensions for text-embedding-3-large.",
    )

    # ── Neo4j ─────────────────────────────────────────────────────────────────
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="hybridrag_neo4j")

    # ── PostgreSQL ────────────────────────────────────────────────────────────
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    postgres_db: str = Field(default="devanalytics")
    postgres_user: str = Field(default="hybridrag")
    postgres_password: str = Field(default="hybridrag_postgres")

    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_host: str = Field(default="localhost")
    qdrant_port: int = Field(default=6333)
    qdrant_collection_tickets: str = Field(default="dev_tickets")
    qdrant_collection_arch_docs: str = Field(default="dev_arch_docs")
    qdrant_collection_postmortems: str = Field(default="dev_postmortems")

    @property
    def qdrant_collection(self) -> str:
        """Backward-compatible alias — returns the tickets collection name."""
        return self.qdrant_collection_tickets

    # ── Retrieval Parameters ──────────────────────────────────────────────────
    top_k_results: int = Field(
        default=5,
        description="Number of results returned per retrieval call.",
        ge=1,
        le=50,
    )
    max_retrieval_iterations: int = Field(
        default=3,
        description="Maximum closed-loop iterations in V2 before forced acceptance.",
        ge=1,
        le=10,
    )

    # ── Integration Endpoints ─────────────────────────────────────────────────
    mcp_port: int = Field(default=8001)
    a2a_port: int = Field(default=8002)
    a2a_base_url: str = Field(
        default="http://localhost:8002",
        description="Public base URL of the A2A server. Override for external deployments.",
    )

    @property
    def postgres_dsn(self) -> str:
        """Build a psycopg2-compatible DSN string from individual parameters."""
        return (
            f"host={self.postgres_host} "
            f"port={self.postgres_port} "
            f"dbname={self.postgres_db} "
            f"user={self.postgres_user} "
            f"password={self.postgres_password}"
        )


settings = Settings()
