from typing import List, Optional, Dict, Any
from app.models import Document, DocumentCreate, DocumentUpdate
from app.database.storage import storage
import logging

logger = logging.getLogger(__name__)


class DocumentService:
    """
    Service layer for document operations.
    Handles business logic for document management within libraries.
    """

    def __init__(self):
        self.storage = storage
        logger.info("DocumentService initialized")

    async def create_document(self, document_data: DocumentCreate, library_id: str) -> Document:
        """Create a new document in a library"""
        try:
            # Verify library exists
            library = self.storage.get_library(library_id)
            if not library:
                raise ValueError(f"Library not found: {library_id}")

            # Create document object
            document = Document(
                name=document_data.name,
                description=document_data.description,
                metadata=document_data.metadata or {}
            )

            # Store in database
            created_document = self.storage.create_document(document, library_id)

            logger.info(
                f"Created document: {created_document.id} - {created_document.name} in library: {library_id}")
            return created_document

        except Exception as e:
            logger.error(f"Error creating document: {str(e)}")
            raise

    async def get_document(self, document_id: str) -> Optional[Document]:
        """Get document by ID with its chunks"""
        try:
            document = self.storage.get_document(document_id)
            if document:
                # Populate with chunks
                chunks = self.storage.get_chunks_by_document(document_id)
                document.chunks = chunks

            return document

        except Exception as e:
            logger.error(f"Error getting document {document_id}: {str(e)}")
            raise

    async def update_document(self, document_id: str, updates: DocumentUpdate) -> Optional[Document]:
        """Update document"""
        try:
            # Convert to dict and filter None values
            update_data = {
                k: v for k, v in updates.model_dump().items()
                if v is not None
            }

            if not update_data:
                return await self.get_document(document_id)

            updated_document = self.storage.update_document(document_id, update_data)

            if updated_document:
                logger.info(f"Updated document: {document_id}")
                # Populate with chunks for return
                chunks = self.storage.get_chunks_by_document(document_id)
                updated_document.chunks = chunks

            return updated_document

        except Exception as e:
            logger.error(f"Error updating document {document_id}: {str(e)}")
            raise

    async def delete_document(self, document_id: str) -> bool:
        """Delete document and all its chunks"""
        try:
            success = self.storage.delete_document(document_id)

            if success:
                logger.info(f"Deleted document: {document_id}")

            return success

        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            raise

    async def get_documents_by_library(self, library_id: str) -> List[Document]:
        """Get all documents in a library"""
        try:
            documents = self.storage.get_documents_by_library(library_id)

            # Populate each document with its chunks
            for document in documents:
                chunks = self.storage.get_chunks_by_document(document.id)
                document.chunks = chunks

            return documents

        except Exception as e:
            logger.error(f"Error getting documents for library {library_id}: {str(e)}")
            raise


    async def get_documents_by_metadata(self, library_id: str, metadata_filters: Dict[str, Any]) -> List[Document]:
        """Get documents filtered by metadata"""
        try:
            all_documents = self.storage.get_documents_by_library(library_id)

            if not metadata_filters:
                # Populate all with chunks
                for document in all_documents:
                    chunks = self.storage.get_chunks_by_document(document.id)
                    document.chunks = chunks
                return all_documents

            filtered_documents = []
            for document in all_documents:
                match = True
                for key, value in metadata_filters.items():
                    if key not in document.metadata or document.metadata[key] != value:
                        match = False
                        break

                if match:
                    # Populate with chunks
                    chunks = self.storage.get_chunks_by_document(document.id)
                    document.chunks = chunks
                    filtered_documents.append(document)

            return filtered_documents

        except Exception as e:
            logger.error(f"Error filtering documents by metadata: {str(e)}")
            raise


# Global service instance
document_service = DocumentService()