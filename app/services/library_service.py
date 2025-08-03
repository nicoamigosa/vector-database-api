from typing import List, Optional, Dict
from app.models import Library, LibraryCreate, LibraryUpdate
from app.database.storage import storage
from app.database.indexes import IndexManager
import logging

logger = logging.getLogger(__name__)


class LibraryService:
    """
    Service layer for library operations.
    Handles business logic and coordinates between storage and indexing.
    """

    def __init__(self):
        self.storage = storage
        self.indexes: Dict[str, IndexManager] = {}  # library_id -> IndexManager
        logger.info("LibraryService initialized")

    async def create_library(self, library_data: LibraryCreate) -> Library:
        """Create a new library"""
        try:
            # Validate index type using config
            library_data.validate_index_type()

            # Create library object
            library = Library(
                name=library_data.name,
                description=library_data.description,
                metadata=library_data.metadata or {}
            )

            # Store in database
            created_library = self.storage.create_library(library)

            # Initialize index for this library
            self.indexes[created_library.id] = IndexManager(index_type=library_data.index_type)

            logger.info(f"Created library: {created_library.id} - {created_library.name} with {library_data.index_type} index")
            return created_library

        except Exception as e:
            logger.error(f"Error creating library: {str(e)}")
            raise

    async def get_library(self, library_id: str) -> Optional[Library]:
        """Get library by ID"""
        try:
            library = self.storage.get_library(library_id)
            if library:
                # Populate with current documents
                documents = self.storage.get_documents_by_library(library_id)
                library.documents = documents

            return library

        except Exception as e:
            logger.error(f"Error getting library {library_id}: {str(e)}")
            raise

    async def update_library(self, library_id: str, updates: LibraryUpdate) -> Optional[Library]:
        """Update library"""
        try:
            # Convert to dict and filter None values
            update_data = {
                k: v for k, v in updates.model_dump().items()
                if v is not None
            }

            if not update_data:
                return await self.get_library(library_id)

            updated_library = self.storage.update_library(library_id, update_data)

            if updated_library:
                logger.info(f"Updated library: {library_id}")

            return updated_library

        except Exception as e:
            logger.error(f"Error updating library {library_id}: {str(e)}")
            raise

    async def delete_library(self, library_id: str) -> bool:
        """Delete library and all its contents"""
        try:
            # Remove index if exists
            if library_id in self.indexes:
                del self.indexes[library_id]

            # Delete from storage (cascades to documents and chunks)
            success = self.storage.delete_library(library_id)

            if success:
                logger.info(f"Deleted library: {library_id}")

            return success

        except Exception as e:
            logger.error(f"Error deleting library {library_id}: {str(e)}")
            raise

    async def list_libraries(self) -> List[Library]:
        """Get all libraries"""
        try:
            libraries = self.storage.list_libraries()

            # Populate each library with its documents
            for library in libraries:
                documents = self.storage.get_documents_by_library(library.id)
                library.documents = documents

            return libraries

        except Exception as e:
            logger.error(f"Error listing libraries: {str(e)}")
            raise

    async def index_library(self, library_id: str, index_type: str = "flat") -> bool:
        """Index all chunks in a library for vector search"""
        try:
            library = self.storage.get_library(library_id)
            if not library:
                logger.warning(f"Library not found for indexing: {library_id}")
                return False

            # Get all chunks in the library
            chunks = self.storage.get_chunks_by_library(library_id)
            chunks_with_embeddings = [chunk for chunk in chunks if chunk.embedding]

            if not chunks_with_embeddings:
                logger.warning(f"No chunks with embeddings found in library: {library_id}")
                return False

            # Create or update index
            if library_id not in self.indexes:
                self.indexes[library_id] = IndexManager(index_type=index_type)
            else:
                # Rebuild index with new type if different
                current_type = self.indexes[library_id].index_type
                if current_type != index_type:
                    self.indexes[library_id].rebuild_index(chunks_with_embeddings, index_type)
                else:
                    self.indexes[library_id].rebuild_index(chunks_with_embeddings)

            # Mark library as indexed
            self.storage.update_library(library_id, {"is_indexed": True})

            logger.info(
                f"Indexed library {library_id} with {len(chunks_with_embeddings)} chunks "
                f"using {index_type} index"
            )
            return True

        except Exception as e:
            logger.error(f"Error indexing library {library_id}: {str(e)}")
            raise

    def get_index_manager(self, library_id: str) -> Optional[IndexManager]:
        """Get index manager for a library"""
        return self.indexes.get(library_id)

# Global service instance
library_service = LibraryService()