from fastapi import APIRouter, HTTPException, Query, Path
from typing import List, Optional
from app.models import Document, DocumentCreate, DocumentUpdate
from app.services.document_service import document_service

router = APIRouter()


@router.post("/", response_model=Document)
async def create_document(
        document_data: DocumentCreate,
        library_id: str = Query(..., description="Library ID to create document in")
):
    """Create a new document in a library"""
    try:
        return await document_service.create_document(document_data, library_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{document_id}", response_model=Document)
async def get_document(
        document_id: str = Path(..., description="Document ID")
):
    """Get a specific document by ID"""
    try:
        document = await document_service.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        return document
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{document_id}", response_model=Document)
async def update_document(
        document_data: DocumentUpdate,
        document_id: str = Path(..., description="Document ID")
):
    """Update a document"""
    try:
        updated_document = await document_service.update_document(document_id, document_data)
        if not updated_document:
            raise HTTPException(status_code=404, detail="Document not found")
        return updated_document
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{document_id}")
async def delete_document(
        document_id: str = Path(..., description="Document ID")
):
    """Delete a document and all its chunks"""
    try:
        success = await document_service.delete_document(document_id)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        return {"message": "Document deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/library/{library_id}", response_model=List[Document])
async def get_documents_by_library(
        library_id: str = Path(..., description="Library ID")
):
    """Get all documents in a library"""
    try:
        return await document_service.get_documents_by_library(library_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/library/{library_id}/filter")
async def filter_documents_by_metadata(
        library_id: str = Path(..., description="Library ID"),
        metadata_filter: Optional[str] = Query(None, description="JSON string of metadata filters")
):
    """Filter documents by metadata"""
    try:
        # Parse metadata filter if provided
        metadata_filters = {}
        if metadata_filter:
            import json
            try:
                metadata_filters = json.loads(metadata_filter)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON in metadata_filter")

        documents = await document_service.get_documents_by_metadata(library_id, metadata_filters)
        return documents
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))