from fastapi import APIRouter, HTTPException, Query, Path
from typing import List

from app.config import get_default_index_type, get_available_index_types, validate_index_type
from app.models import Library, LibraryCreate, LibraryUpdate
from app.services.library_service import library_service

router = APIRouter()


@router.post("/", response_model=Library)
async def create_library(library_data: LibraryCreate):
    """Create a new library"""
    try:
        return await library_service.create_library(library_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/", response_model=List[Library])
async def list_libraries():
    """Get all libraries"""
    try:
        return await library_service.list_libraries()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{library_id}", response_model=Library)
async def get_library(
        library_id: str = Path(..., description="Library ID")
):
    """Get a specific library by ID"""
    try:
        library = await library_service.get_library(library_id)
        if not library:
            raise HTTPException(status_code=404, detail="Library not found")
        return library
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{library_id}", response_model=Library)
async def update_library(
        library_data: LibraryUpdate,
        library_id: str = Path(..., description="Library ID")
):
    """Update a library"""
    try:
        updated_library = await library_service.update_library(library_id, library_data)
        if not updated_library:
            raise HTTPException(status_code=404, detail="Library not found")
        return updated_library
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{library_id}")
async def delete_library(
        library_id: str = Path(..., description="Library ID")
):
    """Delete a library and all its contents"""
    try:
        success = await library_service.delete_library(library_id)
        if not success:
            raise HTTPException(status_code=404, detail="Library not found")
        return {"message": "Library deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{library_id}/index")
async def index_library(
    library_id: str = Path(..., description="Library ID"),
    index_type: str = Query(
        default_factory=get_default_index_type,
        description=f"Index type. Available: {', '.join(get_available_index_types())}"
    )
):
    """Index a library for vector search"""
    try:
        if not validate_index_type(index_type):
            available_types = ", ".join(get_available_index_types())
            raise HTTPException(
                status_code=400,
                detail=f"Invalid index type. Must be one of: {available_types}"
            )

        success = await library_service.index_library(library_id, index_type)
        if not success:
            raise HTTPException(status_code=404, detail="Library not found or no chunks to index")

        return {"message": f"Library indexed successfully with {index_type} index"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

