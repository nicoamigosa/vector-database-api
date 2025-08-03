from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import uuid4

from app.config import validate_index_type, get_default_index_type, get_available_index_types


class ChunkCreate(BaseModel):
    """Model for creating a new chunk"""
    text: str = Field(..., description="The text content of the chunk")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class ChunkUpdate(BaseModel):
    """Model for updating a chunk"""
    text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class Chunk(BaseModel):
    """Main chunk model with all fields"""
    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique chunk identifier")
    text: str = Field(..., description="The text content of the chunk")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding of the text")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    document_id: Optional[str] = Field(None, description="ID of the parent document")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DocumentCreate(BaseModel):
    """Model for creating a new document"""
    name: str = Field(..., description="Document name")
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class DocumentUpdate(BaseModel):
    """Model for updating a document"""
    name: Optional[str] = None
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class Document(BaseModel):
    """Main document model"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(..., description="Document name")
    description: Optional[str] = None
    chunks: List[Chunk] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    library_id: Optional[str] = Field(None, description="ID of the parent library")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class LibraryCreate(BaseModel):
    """Model for creating a new library"""
    name: str = Field(..., description="Library name")
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    index_type: str = Field(
        default_factory=get_default_index_type,
        description=f"Index type. Available: {', '.join(get_available_index_types())}"
    )

    def validate_index_type(self):
        """Validate index type using config"""
        if not validate_index_type(self.index_type):
            available_types = ", ".join(get_available_index_types())
            raise ValueError(f"Index type must be one of: {available_types}")
        return self

class LibraryUpdate(BaseModel):
    """Model for updating a library"""
    name: Optional[str] = None
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class Library(BaseModel):
    """Main library model"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(..., description="Library name")
    description: Optional[str] = None
    documents: List[Document] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_indexed: bool = Field(default=False, description="Whether the library has been indexed")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SearchQuery(BaseModel):
    """Model for vector search queries"""
    query_text: Optional[str] = Field(None, description="Text to search for")
    query_embedding: Optional[List[float]] = Field(None, description="Pre-computed embedding to search with")
    k: int = Field(default=5, description="Number of nearest neighbors to return")
    metadata_filters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Metadata filters")

    def validate_query(self):
        """Ensure either query_text or query_embedding is provided"""
        if not self.query_text and not self.query_embedding:
            raise ValueError("Either query_text or query_embedding must be provided")
        return self


class SearchResult(BaseModel):
    """Model for search results"""
    chunk: Chunk
    similarity_score: float = Field(..., description="Cosine similarity score")
    distance: float = Field(..., description="Vector distance")


class SearchResponse(BaseModel):
    """Model for search response"""
    query: SearchQuery
    results: List[SearchResult]
    total_results: int
    execution_time_ms: float