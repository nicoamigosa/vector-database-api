import asyncio
import aiohttp
from typing import List, Optional
from app.config import settings
import logging
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating embeddings using Cohere API"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.cohere_api_key
        self.base_url = "https://api.cohere.ai/v2"
        self.model = settings.cohere_model

        if not self.api_key:
            raise ValueError(
                "Cohere API key is required. Set COHERE_API_KEY environment variable or pass api_key parameter.")

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        embeddings = await self.generate_embeddings([text])
        return embeddings[0] if embeddings else []

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }

                payload = {
                    "texts": texts,
                    "model": self.model,
                    "input_type": "search_document"
                }

                async with session.post(
                        f"{self.base_url}/embed",
                        json=payload,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=settings.request_timeout_seconds)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        embeddings_data = data.get("embeddings", {})
                        return embeddings_data.get("float", [])
                    else:
                        error_text = await response.text()
                        logger.error(f"Cohere API error {response.status}: {error_text}")
                        raise Exception(f"Failed to generate embeddings: {error_text}")

        except asyncio.TimeoutError:
            logger.error("Timeout while generating embeddings")
            raise Exception("Timeout while generating embeddings")
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise


class VectorOperations:
    """Utility class for vector operations"""

    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same dimension")

        v1 = np.array(vec1)
        v2 = np.array(vec2)

        # Handle zero vectors
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(v1, v2) / (norm1 * norm2))

    @staticmethod
    def euclidean_distance(vec1: List[float], vec2: List[float]) -> float:
        """Calculate Euclidean distance between two vectors"""
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same dimension")

        v1 = np.array(vec1)
        v2 = np.array(vec2)

        return float(np.linalg.norm(v1 - v2))

    @staticmethod
    def dot_product(vec1: List[float], vec2: List[float]) -> float:
        """Calculate dot product between two vectors"""
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same dimension")

        return float(np.dot(vec1, vec2))

    @staticmethod
    def normalize_vector(vector: List[float]) -> List[float]:
        """Normalize vector to unit length"""
        v = np.array(vector)
        norm = np.linalg.norm(v)
        if norm == 0:
            return vector  # Return original if zero vector
        return (v / norm).tolist()

    @staticmethod
    def batch_cosine_similarity(query_vec: List[float], vectors: List[List[float]]) -> List[float]:
        """Calculate cosine similarity between query vector and batch of vectors (optimized)"""
        if not vectors:
            return []

        query = np.array(query_vec)
        matrix = np.array(vectors)

        # Normalize query vector
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return [0.0] * len(vectors)
        query_normalized = query / query_norm

        # Normalize all vectors in the matrix
        norms = np.linalg.norm(matrix, axis=1)
        # Handle zero vectors
        mask = norms != 0
        similarities = np.zeros(len(vectors))

        if np.any(mask):
            matrix_normalized = matrix[mask] / norms[mask][:, np.newaxis]
            similarities[mask] = np.dot(matrix_normalized, query_normalized)

        return similarities.tolist()


# Global embedding service instance
embedding_service = EmbeddingService()