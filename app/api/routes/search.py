from fastapi import APIRouter, HTTPException, Query, Path
from typing import Optional
from app.models import SearchQuery, SearchResponse
from app.services.search_service import search_service

router = APIRouter()


@router.post("/libraries/{library_id}", response_model=SearchResponse)
async def search_library(
        query: SearchQuery,
        library_id: str = Path(..., description="Library ID to search in")
):
    """Perform vector search within a library"""
    try:
        response = await search_service.search_library(library_id, query)
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/libraries/{library_id}/simple")
async def simple_search(
        library_id: str = Path(..., description="Library ID to search in"),
        q: str = Query(..., description="Search query text"),
        k: int = Query(5, description="Number of results to return", ge=1, le=100),
        metadata_filter: Optional[str] = Query(None, description="JSON string of metadata filters")
):
    """Simple search endpoint with query parameters"""
    try:
        # Parse metadata filter if provided
        metadata_filters = {}
        if metadata_filter:
            import json
            try:
                metadata_filters = json.loads(metadata_filter)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON in metadata_filter")

        # Create search query
        query = SearchQuery(
            query_text=q,
            k=k,
            metadata_filters=metadata_filters
        )

        response = await search_service.search_library(library_id, query)
        return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))