from typing import List, Optional, Dict, Any
from app.models import SearchQuery, SearchResult, SearchResponse, Chunk
from app.database.storage import storage
from app.utils.embedding import embedding_service
from app.services.library_service import library_service
from app.config import validate_k_neighbors
import logging
import time

logger = logging.getLogger(__name__)


class SearchService:
    """
    Service layer for vector search operations.
    Handles query processing, embedding generation, and result formatting.
    """

    def __init__(self):
        self.storage = storage
        self.embedding_service = embedding_service
        logger.info("SearchService initialized")

    async def search_library(
            self,
            library_id: str,
            query: SearchQuery
    ) -> SearchResponse:
        """Perform vector search within a library"""
        start_time = time.time()

        try:
            # Validate query
            query.validate_query()

            # Validate library exists
            library = await library_service.get_library(library_id)
            if not library:
                raise ValueError(f"Library not found: {library_id}")

            # Validate k parameter
            k = validate_k_neighbors(query.k)

            # Get query embedding
            query_embedding = await self._get_query_embedding(query)
            if not query_embedding:
                raise ValueError("Could not generate query embedding")

            # Get index manager for the library
            index_manager = library_service.get_index_manager(library_id)
            if not index_manager:
                # Fallback to flat search if no index
                results = await self._fallback_search(library_id, query_embedding, k, query.metadata_filters)
            else:
                # Use indexed search
                results = await self._indexed_search(library_id, index_manager, query_embedding, k,
                                                     query.metadata_filters)

            execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            response = SearchResponse(
                query=query,
                results=results,
                total_results=len(results),
                execution_time_ms=execution_time
            )

            logger.info(
                f"Search completed for library {library_id}: "
                f"{len(results)} results in {execution_time:.2f}ms"
            )

            return response

        except Exception as e:
            logger.error(f"Error searching library {library_id}: {str(e)}")
            raise

    async def _get_query_embedding(self, query: SearchQuery) -> Optional[List[float]]:
        """Get or generate query embedding"""
        if query.query_embedding:
            return query.query_embedding
        elif query.query_text:
            return await self.embedding_service.generate_embedding(query.query_text)
        else:
            return None

    async def _indexed_search(
            self,
            library_id: str,
            index_manager,
            query_embedding: List[float],
            k: int,
            metadata_filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Perform search using the library's index"""

        # Get all chunks in library for metadata filtering and result construction
        all_chunks = self.storage.get_chunks_by_library(library_id)
        chunks_dict = {chunk.id: chunk for chunk in all_chunks}

        # Apply metadata filters first if specified
        if metadata_filters:
            filtered_chunks = self._apply_metadata_filters(all_chunks, metadata_filters)
            chunks_dict = {chunk.id: chunk for chunk in filtered_chunks}

        # Perform vector search
        results = index_manager.search(query_embedding, k, chunks_dict)

        return results

    async def _fallback_search(
            self,
            library_id: str,
            query_embedding: List[float],
            k: int,
            metadata_filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Fallback search without index (brute force)"""

        # Get all chunks with embeddings
        all_chunks = self.storage.get_chunks_by_library(library_id)
        chunks_with_embeddings = [chunk for chunk in all_chunks if chunk.embedding]

        # Apply metadata filters
        if metadata_filters:
            chunks_with_embeddings = self._apply_metadata_filters(chunks_with_embeddings, metadata_filters)

        if not chunks_with_embeddings:
            return []

        # Calculate similarities
        from app.utils.embedding import VectorOperations

        similarities = []
        for chunk in chunks_with_embeddings:
            similarity = VectorOperations.cosine_similarity(query_embedding, chunk.embedding)
            distance = VectorOperations.euclidean_distance(query_embedding, chunk.embedding)
            similarities.append((chunk, similarity, distance))

        # Sort by similarity and take top k
        similarities.sort(key=lambda x: x[1], reverse=True)

        results = []
        for chunk, similarity, distance in similarities[:k]:
            result = SearchResult(
                chunk=chunk,
                similarity_score=similarity,
                distance=distance
            )
            results.append(result)

        return results

    def _apply_metadata_filters(
            self,
            chunks: List[Chunk],
            filters: Dict[str, Any]
    ) -> List[Chunk]:
        """Apply metadata filters to chunks"""
        if not filters:
            return chunks

        filtered_chunks = []
        for chunk in chunks:
            match = True

            for key, value in filters.items():
                # Support different filter types
                if key.startswith("created_after"):
                    # Date filter
                    if chunk.created_at <= value:
                        match = False
                        break
                elif key.startswith("created_before"):
                    if chunk.created_at >= value:
                        match = False
                        break
                elif key.endswith("_contains"):
                    # String contains filter
                    metadata_key = key.replace("_contains", "")
                    if metadata_key not in chunk.metadata:
                        match = False
                        break
                    if value.lower() not in str(chunk.metadata[metadata_key]).lower():
                        match = False
                        break
                else:
                    # Exact match filter
                    if key not in chunk.metadata or chunk.metadata[key] != value:
                        match = False
                        break

            if match:
                filtered_chunks.append(chunk)

        return filtered_chunks

# Global service instance
search_service = SearchService()