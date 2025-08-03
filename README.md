# Vector Database API

REST API for indexing and searching documents using vector databases. It implements indexing algorithms from scratch and allows k-NN searches over text embeddings.

## 🚀 Features

- **REST API** for managing libraries, documents, and chunks
- **Indexing algorithms** implemented from scratch:
  - **LSH**: Locality Sensitive Hashing for fast approximate search
  - **IVF**: Inverted File with clustering for large datasets
- **Thread-safe**: Safe concurrency management with read-write locks
- **Automatic embeddings** using Cohere API
- **k-NN vector search** with metadata filters
- **Docker containerization** ready for production

## 📁 Project Structure

```
vector_db/
├── app/
│   ├── main.py                 # FastAPI application
│   ├── config.py              # Configuration settings
│   ├── models.py              # Pydantic models
│   ├── database/
│   │   ├── storage.py         # In-memory storage
│   │   └── indexes.py         # Indexing algorithms
│   ├── services/
│   │   ├── library_service.py
│   │   ├── document_service.py
│   │   ├── chunk_service.py
│   │   └── search_service.py
│   ├── api/routes/
│   │   ├── libraries.py
│   │   ├── documents.py
│   │   ├── chunks.py
│   │   └── search.py
│   └── utils/
│       ├── embedding.py       # Cohere integration
│       └── concurrency.py     # Thread safety
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## 🛠️ Installation and Setup

### 1. Clone and set up

```bash
git clone https://github.com/nicoamigosa/vector-database-api.git
cd vector_db_api

# Create config file
cp .env.example .env
```

### 2. Set environment variables

Edit `.env` with your Cohere API key:

```bash
COHERE_API_KEY=your_cohere_api_key_here
```

### 3. Run with Docker 

```bash
# Build and run
docker-compose up --build

# Stop and remove 
docker compose down
```

### 4. Run locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
uvicorn app.main:app --reload
```

API will be available at: http://localhost:8000

## 📚 API Usage

### Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Basic Flow

1. **Create a library**
```bash
curl -X POST "http://localhost:8000/api/v1/libraries/"   -H "Content-Type: application/json"   -d '{"name": "My Library", "description": "Test documents"}'
```

2. **Create a document**
```bash
curl -X POST "http://localhost:8000/api/v1/documents/?library_id=<library_id>"   -H "Content-Type: application/json"   -d '{"name": "Document 1", "description": "First document"}'
```

3. **Add text chunks**
```bash
curl -X POST "http://localhost:8000/api/v1/chunks/?document_id=<document_id>"   -H "Content-Type: application/json"   -d '{"text": "This is the chunk content", "metadata": {"type": "paragraph"}}'
```

4. **Index the library**
```bash
curl -X POST "http://localhost:8000/api/v1/libraries/<library_id>/index?index_type=flat"
```

5. **Search the library**
```bash
curl -X GET "http://localhost:8000/api/v1/search/libraries/<library_id>/simple?q=content&k=5"
```

### Advanced Usage Examples

#### Search with metadata filters
```bash
curl -X GET "http://localhost:8000/api/v1/search/libraries/<library_id>/simple?q=text&metadata_filter={"type":"paragraph"}"
```

#### Change index type
```bash
curl -X PUT "http://localhost:8000/api/v1/libraries/<library_id>/index-type?index_type=lsh"
```

## 🎮 Interactive Demo

The project includes a comprehensive demo script that showcases all the Vector Database API capabilities in action.

### Running the Demo

```bash
# 1. Start the API server
docker-compose up -d
# Or locally: uvicorn app.main:app --reload

# 2. Run the interactive demo
python demo.py
```

## 🔍 Indexing Algorithms

### LSH (Locality Sensitive Hashing)
- **Complexity**: O(L×k + C) search, O(L×k) insert
- **Use case**: Medium datasets (1K–100K vectors)
- **Pros**: Fast approximate searches
- **Cons**: Approximate results

### IVF (Inverted File)
- **Complexity**: O(d×nprobe + C) search, O(d×nlist) insert
- **Use case**: Large datasets (>10K vectors)
- **Pros**: Great balance of speed/accuracy
- **Cons**: Requires initial training



## 🧪 Testing

### Run tests
```bash
# With pytest
pytest

# With coverage
pytest --cov=app tests/
```

## 🏗️ Architecture and Design

### Design Principles

1. **Separation of concerns**: API → Services → Storage
2. **Thread safety**: Read-write locks for concurrency
3. **Extensibility**: Easy to add new algorithms
4. **Performance**: Algorithms optimized for dataset sizes

### Concurrency Decisions

- **ReadWriteLock**: Multiple concurrent readers, exclusive writers
- **ThreadSafeDict**: Thread-safe containers for storage
- **AsyncRateLimiter**: Rate control for external APIs

### Embedding Management

- **Batch processing**: Efficient multiple embedding generation
- **Caching**: In-memory storage for fast access


## 📈 Performance Optimization

### Dataset size recommendations:

- **100–10K chunks**: LSH index  
- **> 10K chunks**: IVF index

