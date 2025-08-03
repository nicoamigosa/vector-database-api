from pydantic_settings import BaseSettings
from typing import Optional, List


class Settings(BaseSettings):
    """Application configuration settings"""

    # API Configuration
    app_name: str = "Vector Database API"
    app_version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000

    # Cohere API Configuration
    cohere_api_key: Optional[str] = None
    cohere_model: str = "embed-v4.0"  # Default embedding model

    # Vector Database Configuration
    default_embedding_dimension: int = 1024  # Cohere embed-english-v3.0 dimension
    max_chunks_per_document: int = 1000
    max_documents_per_library: int = 100

    # Vector Index Configuration
    available_index_types: List[str] = ["lsh", "ivf"]
    default_index_type: str = "lsh"

    # Index-specific settings
    lsh_num_tables: int = 8
    lsh_hash_length: int = 12
    ivf_nlist: int = 100
    ivf_nprobe: int = 5

    # Search Configuration
    default_k_neighbors: int = 5
    max_k_neighbors: int = 100
    similarity_threshold: float = 0.0  # Minimum similarity score

    # Performance Configuration
    max_concurrent_requests: int = 10
    request_timeout_seconds: int = 30

    # Storage Configuration (for future disk persistence)
    data_directory: str = "./data"
    checkpoint_interval_seconds: int = 300  # 5 minutes

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()


# Validation functions
def validate_embedding_dimension(embedding: list) -> bool:
    """Validate that embedding has correct dimensions"""
    return len(embedding) == settings.default_embedding_dimension


def validate_k_neighbors(k: int) -> int:
    """Validate and clamp k neighbors parameter"""
    if k <= 0:
        return settings.default_k_neighbors
    if k > settings.max_k_neighbors:
        return settings.max_k_neighbors
    return k

def validate_index_type(index_type: str) -> bool:
    """Validate that index type is supported"""
    return index_type in settings.available_index_types


def get_default_index_type() -> str:
    """Get default index type"""
    return settings.default_index_type


def get_available_index_types() -> List[str]:
    """Get list of available index types"""
    return settings.available_index_types.copy()