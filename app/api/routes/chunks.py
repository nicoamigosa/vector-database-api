from fastapi import APIRouter, HTTPException, Query, Path
from typing import List, Optional
from app.models import Chunk, ChunkCreate, ChunkUpdate
from app.services.chunk_service import chunk_service

router = APIRouter()


@router.post("/", response_model=Chunk)
async def create_chunk(
        chunk_data: ChunkCreate,
        document_id: str = Query(..., description="Document ID to create chunk in")
):
    """Create a new chunk in a document"""
    try:
        return await chunk_service.create_chunk(chunk_data, document_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{chunk_id}", response_model=Chunk)
async def get_chunk(
        chunk_id: str = Path(..., description="Chunk ID")
):
    """Get a specific chunk by ID"""
    try:
        chunk = await chunk_service.get_chunk(chunk_id)
        if not chunk:
            raise HTTPException(status_code=404, detail="Chunk not found")
        return chunk
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{chunk_id}", response_model=Chunk)
async def update_chunk(
        chunk_data: ChunkUpdate,
        chunk_id: str = Path(..., description="Chunk ID")
):
    """Update a chunk (regenerates embedding if text changed)"""
    try:
        updated_chunk = await chunk_service.update_chunk(chunk_id, chunk_data)
        if not updated_chunk:
            raise HTTPException(status_code=404, detail="Chunk not found")
        return updated_chunk
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{chunk_id}")
async def delete_chunk(
        chunk_id: str = Path(..., description="Chunk ID")
):
    """Delete a chunk"""
    try:
        success = await chunk_service.delete_chunk(chunk_id)
        if not success:
            raise HTTPException(status_code=404, detail="Chunk not found")
        return {"message": "Chunk deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/document/{document_id}", response_model=List[Chunk])
async def get_chunks_by_document(
        document_id: str = Path(..., description="Document ID")
):
    """Get all chunks in a document"""
    try:
        return await chunk_service.get_chunks_by_document(document_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/library/{library_id}", response_model=List[Chunk])
async def get_chunks_by_library(
        library_id: str = Path(..., description="Library ID")
):
    """Get all chunks in a library"""
    try:
        return await chunk_service.get_chunks_by_library(library_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/library/{library_id}/filter")
async def filter_chunks_by_metadata(
        library_id: str = Path(..., description="Library ID"),
        metadata_filter: Optional[str] = Query(None, description="JSON string of metadata filters")
):
    """Filter chunks by metadata within a library"""
    try:
        # Parse metadata filter if provided
        metadata_filters = {}
        if metadata_filter:
            import json
            try:
                metadata_filters = json.loads(metadata_filter)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON in metadata_filter")

        chunks = await chunk_service.filter_chunks_by_metadata(library_id, metadata_filters)
        return chunks
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))