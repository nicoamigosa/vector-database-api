import pytest
from fastapi.testclient import TestClient
from app.main import app

# Test client
client = TestClient(app)


def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "version" in data


def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


class TestLibraryOperations:
    """Test library CRUD operations"""

    def test_create_library(self):
        """Test library creation"""
        library_data = {
            "name": "Test Library",
            "description": "A test library",
            "metadata": {"test": True}
        }

        response = client.post("/api/v1/libraries/", json=library_data)
        assert response.status_code == 200

        data = response.json()
        assert data["name"] == library_data["name"]
        assert data["description"] == library_data["description"]
        assert "id" in data
        assert "created_at" in data

        return data["id"]  # Return for use in other tests

    def test_list_libraries(self):
        """Test listing libraries"""
        # First create a library
        self.test_create_library()

        response = client.get("/api/v1/libraries/")
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_get_library(self):
        """Test getting specific library"""
        # Create a library first
        library_id = self.test_create_library()

        response = client.get(f"/api/v1/libraries/{library_id}")
        assert response.status_code == 200

        data = response.json()
        assert data["id"] == library_id
        assert data["name"] == "Test Library"

    def test_get_nonexistent_library(self):
        """Test getting non-existent library"""
        response = client.get("/api/v1/libraries/nonexistent-id")
        assert response.status_code == 404


class TestDocumentOperations:
    """Test document CRUD operations"""

    def setup_method(self):
        """Setup method to create a library for testing"""
        library_data = {
            "name": "Test Library for Documents",
            "description": "Library for document tests"
        }
        response = client.post("/api/v1/libraries/", json=library_data)
        self.library_id = response.json()["id"]

    def test_create_document(self):
        """Test document creation"""
        document_data = {
            "name": "Test Document",
            "description": "A test document",
            "metadata": {"type": "test"}
        }

        response = client.post(
            f"/api/v1/documents/?library_id={self.library_id}",
            json=document_data
        )
        assert response.status_code == 200

        data = response.json()
        assert data["name"] == document_data["name"]
        assert data["library_id"] == self.library_id
        assert "id" in data

        return data["id"]

    def test_get_document(self):
        """Test getting specific document"""
        document_id = self.test_create_document()

        response = client.get(f"/api/v1/documents/{document_id}")
        assert response.status_code == 200

        data = response.json()
        assert data["id"] == document_id
        assert data["name"] == "Test Document"


class TestChunkOperations:
    """Test chunk CRUD operations"""

    def setup_method(self):
        """Setup method to create library and document"""
        # Create library
        library_data = {"name": "Test Library for Chunks"}
        response = client.post("/api/v1/libraries/", json=library_data)
        self.library_id = response.json()["id"]

        # Create document
        document_data = {"name": "Test Document for Chunks"}
        response = client.post(
            f"/api/v1/documents/?library_id={self.library_id}",
            json=document_data
        )
        self.document_id = response.json()["id"]

    def test_create_chunk(self):
        """Test chunk creation"""
        chunk_data = {
            "text": "This is a test chunk with some content for testing embeddings.",
            "metadata": {"section": "test"}
        }

        response = client.post(
            f"/api/v1/chunks/?document_id={self.document_id}",
            json=chunk_data
        )
        assert response.status_code == 200

        data = response.json()
        assert data["text"] == chunk_data["text"]
        assert data["document_id"] == self.document_id
        assert "embedding" in data  # Should have embedding generated
        assert "id" in data

        return data["id"]

    def test_get_chunk(self):
        """Test getting specific chunk"""
        chunk_id = self.test_create_chunk()

        response = client.get(f"/api/v1/chunks/{chunk_id}")
        assert response.status_code == 200

        data = response.json()
        assert data["id"] == chunk_id


class TestSearchOperations:
    """Test search functionality"""

    def setup_method(self):
        """Setup method with library, document, and chunks"""
        # Create library
        library_data = {"name": "Search Test Library"}
        response = client.post("/api/v1/libraries/", json=library_data)
        self.library_id = response.json()["id"]

        # Create document
        document_data = {"name": "Search Test Document"}
        response = client.post(
            f"/api/v1/documents/?library_id={self.library_id}",
            json=document_data
        )
        self.document_id = response.json()["id"]

        # Create some test chunks
        test_chunks = [
            {"text": "Python is a programming language used for data science and web development."},
            {"text": "Machine learning algorithms can process large datasets efficiently."},
            {"text": "Vector databases store high-dimensional embeddings for similarity search."},
            {"text": "Natural language processing enables computers to understand human text."}
        ]

        for chunk_data in test_chunks:
            response = client.post(
                f"/api/v1/chunks/?document_id={self.document_id}",
                json=chunk_data
            )
            assert response.status_code == 200

        # Index the library
        response = client.post(f"/api/v1/libraries/{self.library_id}/index?index_type=lsh")
        assert response.status_code == 200

    def test_simple_search(self):
        """Test simple text search"""
        response = client.get(
            f"/api/v1/search/libraries/{self.library_id}/simple?q=programming&k=2"
        )
        assert response.status_code == 200

        data = response.json()
        assert "results" in data
        assert "query" in data
        assert "execution_time_ms" in data
        assert len(data["results"]) <= 2

class TestIndexingAlgorithms:
    """Test different indexing algorithms"""

    def setup_method(self):
        """Setup with test data"""
        # Create library
        library_data = {"name": "Indexing Test Library"}
        response = client.post("/api/v1/libraries/", json=library_data)
        self.library_id = response.json()["id"]

        # Create document
        document_data = {"name": "Indexing Test Document"}
        response = client.post(
            f"/api/v1/documents/?library_id={self.library_id}",
            json=document_data
        )
        document_id = response.json()["id"]

        # Create chunks
        # Create chunks individually
        test_chunks = [{"text": f"Test chunk number {i} with unique content."} for i in range(10)]

        for chunk_data in test_chunks:
            response = client.post(f"/api/v1/chunks/?document_id={document_id}", json=chunk_data)

    def test_lsh_index(self):
        """Test LSH indexing"""
        response = client.post(f"/api/v1/libraries/{self.library_id}/index?index_type=lsh")
        assert response.status_code == 200

    def test_ivf_index(self):
        """Test IVF indexing"""
        response = client.post(f"/api/v1/libraries/{self.library_id}/index?index_type=ivf")
        assert response.status_code == 200

    def test_change_index_type(self):
        """Test different index types work"""
        # Start with IVF
        response1 = client.post(f"/api/v1/libraries/{self.library_id}/index?index_type=ivf")
        assert response1.status_code == 200

        # Change to LSH
        response2 = client.post(f"/api/v1/libraries/{self.library_id}/index?index_type=lsh")
        assert response2.status_code == 200

if __name__ == "__main__":
    # Run specific test
    pytest.main([__file__, "-v"])