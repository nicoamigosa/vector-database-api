from typing import List, Optional, Dict, Any
from app.models import Chunk, ChunkCreate, ChunkUpdate
from app.database.storage import storage
from app.utils.embedding import embedding_service
from app.services.library_service import library_service
import logging

logger = logging.getLogger(__name__)


class ChunkService:
    """
    Service layer for chunk operations.
    Handles business logic including embedding generation and index updates.
    """

    def __init__(self):
        self.storage = storage
        self.embedding_service = embedding_service
        logger.info("ChunkService initialized")

    async def create_chunk(self, chunk_data: ChunkCreate, document_id: str) -> Chunk:
        """Create a new chunk with embedding"""
        try:
            # Verify document exists
            document = self.storage.get_document(document_id)
            if not document:
                raise ValueError(f"Document not found: {document_id}")

            # Generate embedding for the text
            embedding = await self.embedding_service.generate_embedding(chunk_data.text)

            # Create chunk object
            chunk = Chunk(
                text=chunk_data.text,
                embedding=embedding,
                metadata=chunk_data.metadata or {}
            )

            # Store in database
            created_chunk = self.storage.create_chunk(chunk, document_id)

            # Add to index if library is indexed
            if document.library_id:
                index_manager = library_service.get_index_manager(document.library_id)
                if index_manager:
                    index_manager.add_chunk(created_chunk)

            logger.info(f"Created chunk: {created_chunk.id} in document: {document_id}")
            return created_chunk

        except Exception as e:
            logger.error(f"Error creating chunk: {str(e)}")
            raise

    async def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """Get chunk by ID"""
        try:
            return self.storage.get_chunk(chunk_id)

        except Exception as e:
            logger.error(f"Error getting chunk {chunk_id}: {str(e)}")
            raise

    async def update_chunk(self, chunk_id: str, updates: ChunkUpdate) -> Optional[Chunk]:
        """Update chunk and regenerate embedding if text changed"""
        try:
            existing_chunk = self.storage.get_chunk(chunk_id)
            if not existing_chunk:
                return None

            # Convert to dict and filter None values
            update_data = {
                k: v for k, v in updates.dict().items()
                if v is not None
            }

            if not update_data:
                return existing_chunk

            # If text is being updated, regenerate embedding
            if "text" in update_data:
                new_embedding = await self.embedding_service.generate_embedding(update_data["text"])
                update_data["embedding"] = new_embedding

            # Update in storage
            updated_chunk = self.storage.update_chunk(chunk_id, update_data)

            if updated_chunk:
                # Update in index if embedding changed
                if "embedding" in update_data and updated_chunk.document_id:
                    document = self.storage.get_document(updated_chunk.document_id)
                    if document and document.library_id:
                        index_manager = library_service.get_index_manager(document.library_id)
                        if index_manager:
                            # Remove old and add new
                            index_manager.remove_chunk(chunk_id)
                            index_manager.add_chunk(updated_chunk)

                logger.info(f"Updated chunk: {chunk_id}")

            return updated_chunk

        except Exception as e:
            logger.error(f"Error updating chunk {chunk_id}: {str(e)}")
            raise

    async def delete_chunk(self, chunk_id: str) -> bool:
        """Delete chunk"""
        try:
            # Get chunk info before deletion
            chunk = self.storage.get_chunk(chunk_id)
            if not chunk:
                return False

            # Remove from index if exists
            if chunk.document_id:
                document = self.storage.get_document(chunk.document_id)
                if document and document.library_id:
                    index_manager = library_service.get_index_manager(document.library_id)
                    if index_manager:
                        index_manager.remove_chunk(chunk_id)

            # Delete from storage
            success = self.storage.delete_chunk(chunk_id)

            if success:
                logger.info(f"Deleted chunk: {chunk_id}")

            return success

        except Exception as e:
            logger.error(f"Error deleting chunk {chunk_id}: {str(e)}")
            raise

    async def get_chunks_by_document(self, document_id: str) -> List[Chunk]:
        """Get all chunks in a document"""
        try:
            return self.storage.get_chunks_by_document(document_id)

        except Exception as e:
            logger.error(f"Error getting chunks for document {document_id}: {str(e)}")
            raise

    async def get_chunks_by_library(self, library_id: str) -> List[Chunk]:
        """Get all chunks in a library"""
        try:
            return self.storage.get_chunks_by_library(library_id)

        except Exception as e:
            logger.error(f"Error getting chunks for library {library_id}: {str(e)}")
            raise

    async def filter_chunks_by_metadata(
            self,
            library_id: str,
            metadata_filters: Dict[str, Any]
    ) -> List[Chunk]:
        """Filter chunks by metadata criteria"""
        try:
            all_chunks = self.storage.get_chunks_by_library(library_id)

            if not metadata_filters:
                return all_chunks

            filtered_chunks = []
            for chunk in all_chunks:
                match = True
                for key, value in metadata_filters.items():
                    if key not in chunk.metadata or chunk.metadata[key] != value:
                        match = False
                        break

                if match:
                    filtered_chunks.append(chunk)

            return filtered_chunks

        except Exception as e:
            logger.error(f"Error filtering chunks by metadata: {str(e)}")
            raise


# Global service instance
chunk_service = ChunkService()