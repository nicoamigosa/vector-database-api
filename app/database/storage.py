from typing import Dict, List, Optional, Any
from app.models import Library, Document, Chunk
from app.utils.concurrency import ThreadSafeDict, ReadWriteLock
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class InMemoryStorage:
    """
    Thread-safe in-memory storage system for the vector database.

    This class manages the storage of libraries, documents, and chunks
    with proper concurrency control to prevent data races.
    """

    def __init__(self):
        # Thread-safe storage containers
        self.libraries: ThreadSafeDict[Library] = ThreadSafeDict()
        self.documents: ThreadSafeDict[Document] = ThreadSafeDict()
        self.chunks: ThreadSafeDict[Chunk] = ThreadSafeDict()

        # Relationship mappings for efficient lookups
        self.library_documents: ThreadSafeDict[List[str]] = ThreadSafeDict()  # library_id -> [document_ids]
        self.document_chunks: ThreadSafeDict[List[str]] = ThreadSafeDict()  # document_id -> [chunk_ids]

        # Additional locks for complex operations
        self._global_lock = ReadWriteLock()

        logger.info("InMemoryStorage initialized")

    # ==================== LIBRARY OPERATIONS ====================

    def create_library(self, library: Library) -> Library:
        """Create a new library"""
        library.created_at = datetime.utcnow()
        library.updated_at = datetime.utcnow()

        self.libraries.set(library.id, library)
        self.library_documents.set(library.id, [])

        logger.info(f"Created library: {library.id}")
        return library

    def get_library(self, library_id: str) -> Optional[Library]:
        """Get library by ID"""
        return self.libraries.get(library_id)

    def update_library(self, library_id: str, updates: dict) -> Optional[Library]:
        """Update library with given updates"""
        library = self.libraries.get(library_id)
        if not library:
            return None

        # Apply updates
        for key, value in updates.items():
            if hasattr(library, key) and value is not None:
                setattr(library, key, value)

        library.updated_at = datetime.utcnow()
        self.libraries.set(library_id, library)

        logger.info(f"Updated library: {library_id}")
        return library

    def delete_library(self, library_id: str) -> bool:
        """Delete library and all its documents/chunks"""
        with self._global_lock.write_lock():
            library = self.libraries.get(library_id)
            if not library:
                return False

            # Get all documents in this library
            document_ids = self.library_documents.get(library_id, [])

            # Delete all chunks in all documents
            for doc_id in document_ids:
                chunk_ids = self.document_chunks.get(doc_id, [])
                for chunk_id in chunk_ids:
                    self.chunks.delete(chunk_id)
                self.document_chunks.delete(doc_id)
                self.documents.delete(doc_id)

            # Delete library
            self.libraries.delete(library_id)
            self.library_documents.delete(library_id)

            logger.info(f"Deleted library and all contents: {library_id}")
            return True

    def list_libraries(self) -> List[Library]:
        """Get all libraries"""
        return self.libraries.values()

    # ==================== DOCUMENT OPERATIONS ====================

    def create_document(self, document: Document, library_id: str) -> Document:
        """Create a new document in a library"""
        document.library_id = library_id
        document.created_at = datetime.utcnow()
        document.updated_at = datetime.utcnow()

        # Store document
        self.documents.set(document.id, document)
        self.document_chunks.set(document.id, [])

        # Update library relationship
        doc_ids = self.library_documents.get(library_id, [])
        doc_ids.append(document.id)
        self.library_documents.set(library_id, doc_ids)

        logger.info(f"Created document: {document.id} in library: {library_id}")
        return document

    def get_document(self, document_id: str) -> Optional[Document]:
        """Get document by ID"""
        return self.documents.get(document_id)

    def update_document(self, document_id: str, updates: dict) -> Optional[Document]:
        """Update document with given updates"""
        document = self.documents.get(document_id)
        if not document:
            return None

        # Apply updates
        for key, value in updates.items():
            if hasattr(document, key) and value is not None:
                setattr(document, key, value)

        document.updated_at = datetime.utcnow()
        self.documents.set(document_id, document)

        logger.info(f"Updated document: {document_id}")
        return document

    def delete_document(self, document_id: str) -> bool:
        """Delete document and all its chunks"""
        with self._global_lock.write_lock():
            document = self.documents.get(document_id)
            if not document:
                return False

            # Delete all chunks in this document
            chunk_ids = self.document_chunks.get(document_id, [])
            for chunk_id in chunk_ids:
                self.chunks.delete(chunk_id)

            # Remove from library relationship
            if document.library_id:
                doc_ids = self.library_documents.get(document.library_id, [])
                if document_id in doc_ids:
                    doc_ids.remove(document_id)
                    self.library_documents.set(document.library_id, doc_ids)

            # Delete document
            self.documents.delete(document_id)
            self.document_chunks.delete(document_id)

            logger.info(f"Deleted document and all chunks: {document_id}")
            return True

    def get_documents_by_library(self, library_id: str) -> List[Document]:
        """Get all documents in a library"""
        document_ids = self.library_documents.get(library_id, [])
        documents = []
        for doc_id in document_ids:
            doc = self.documents.get(doc_id)
            if doc:
                documents.append(doc)
        return documents

    # ==================== CHUNK OPERATIONS ====================

    def create_chunk(self, chunk: Chunk, document_id: str) -> Chunk:
        """Create a new chunk in a document"""
        chunk.document_id = document_id
        chunk.created_at = datetime.utcnow()
        chunk.updated_at = datetime.utcnow()

        # Store chunk
        self.chunks.set(chunk.id, chunk)

        # Update document relationship
        chunk_ids = self.document_chunks.get(document_id, [])
        chunk_ids.append(chunk.id)
        self.document_chunks.set(document_id, chunk_ids)

        logger.info(f"Created chunk: {chunk.id} in document: {document_id}")
        return chunk

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """Get chunk by ID"""
        return self.chunks.get(chunk_id)

    def update_chunk(self, chunk_id: str, updates: dict) -> Optional[Chunk]:
        """Update chunk with given updates"""
        chunk = self.chunks.get(chunk_id)
        if not chunk:
            return None

        # Apply updates
        for key, value in updates.items():
            if hasattr(chunk, key) and value is not None:
                setattr(chunk, key, value)

        chunk.updated_at = datetime.utcnow()
        self.chunks.set(chunk_id, chunk)

        logger.info(f"Updated chunk: {chunk_id}")
        return chunk

    def delete_chunk(self, chunk_id: str) -> bool:
        """Delete chunk"""
        chunk = self.chunks.get(chunk_id)
        if not chunk:
            return False

        # Remove from document relationship
        if chunk.document_id:
            chunk_ids = self.document_chunks.get(chunk.document_id, [])
            if chunk_id in chunk_ids:
                chunk_ids.remove(chunk_id)
                self.document_chunks.set(chunk.document_id, chunk_ids)

        # Delete chunk
        self.chunks.delete(chunk_id)

        logger.info(f"Deleted chunk: {chunk_id}")
        return True

    def get_chunks_by_document(self, document_id: str) -> List[Chunk]:
        """Get all chunks in a document"""
        chunk_ids = self.document_chunks.get(document_id, [])
        chunks = []
        for chunk_id in chunk_ids:
            chunk = self.chunks.get(chunk_id)
            if chunk:
                chunks.append(chunk)
        return chunks

    def get_chunks_by_library(self, library_id: str) -> List[Chunk]:
        """Get all chunks in a library"""
        document_ids = self.library_documents.get(library_id, [])
        all_chunks = []
        for doc_id in document_ids:
            chunks = self.get_chunks_by_document(doc_id)
            all_chunks.extend(chunks)
        return all_chunks

    # ==================== UTILITY OPERATIONS ====================

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        with self._global_lock.read_lock():
            return {
                "libraries_count": self.libraries.size(),
                "documents_count": self.documents.size(),
                "chunks_count": self.chunks.size(),
                "total_memory_items": (
                        self.libraries.size() +
                        self.documents.size() +
                        self.chunks.size()
                )
            }

    def clear_all(self) -> None:
        """Clear all data (use with caution)"""
        with self._global_lock.write_lock():
            self.libraries.clear()
            self.documents.clear()
            self.chunks.clear()
            self.library_documents.clear()
            self.document_chunks.clear()

            logger.warning("Cleared all data from storage")

    def validate_relationships(self) -> Dict[str, List[str]]:
        """Validate data integrity and return any issues found"""
        issues = []

        with self._global_lock.read_lock():
            # Check orphaned documents
            for doc_id, document in self.documents.items():
                if document.library_id:
                    if not self.libraries.exists(document.library_id):
                        issues.append(f"Document {doc_id} references non-existent library {document.library_id}")

            # Check orphaned chunks
            for chunk_id, chunk in self.chunks.items():
                if chunk.document_id:
                    if not self.documents.exists(chunk.document_id):
                        issues.append(f"Chunk {chunk_id} references non-existent document {chunk.document_id}")

            # Check relationship consistency
            for lib_id, doc_ids in self.library_documents.items():
                for doc_id in doc_ids:
                    if not self.documents.exists(doc_id):
                        issues.append(f"Library {lib_id} references non-existent document {doc_id}")

            for doc_id, chunk_ids in self.document_chunks.items():
                for chunk_id in chunk_ids:
                    if not self.chunks.exists(chunk_id):
                        issues.append(f"Document {doc_id} references non-existent chunk {chunk_id}")

        return {"issues": issues}


# Global storage instance
storage = InMemoryStorage()