"""
Demo script to showcase the capabilities of the Vector Database API
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, Any

# Configuration
BASE_URL = "http://localhost:8000"
API_BASE = f"{BASE_URL}/api/v1"


class VectorDBDemo:
    """Demo class to showcase functionalities"""

    def __init__(self):
        self.session = None
        self.library_id = None
        self.document_ids = []
        self.chunk_ids = []

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def make_request(self, method: str, endpoint: str, data: Dict = None) -> Dict[Any, Any]:
        """Helper to make HTTP requests"""
        url = f"{API_BASE}{endpoint}"

        async with self.session.request(
                method,
                url,
                json=data,
                headers={"Content-Type": "application/json"}
        ) as response:
            if response.status >= 400:
                text = await response.text()
                raise Exception(f"Request failed {response.status}: {text}")

            return await response.json()

    async def step_1_create_library(self):
        """Step 1: Create a library"""
        print("📚 Step 1: Creating library...")

        library_data = {
            "name": "Technical Documentation Library",
            "description": "Collection of documents about programming and AI",
            "metadata": {
                "category": "technical",
                "language": "es",
                "created_by": "demo_script"
            }
        }

        result = await self.make_request("POST", "/libraries/", library_data)
        self.library_id = result["id"]

        print(f"✅ Library created: {result['name']} (ID: {self.library_id})")
        return result

    async def step_2_create_documents(self):
        """Step 2: Create documents"""
        print("\n📄 Step 2: Creating documents...")

        documents = [
            {
                "name": "Introduction to Python",
                "description": "Basic programming concepts in Python",
                "metadata": {"topic": "python", "level": "beginner"}
            },
            {
                "name": "Machine Learning Fundamentals",
                "description": "Fundamentals of machine learning",
                "metadata": {"topic": "ml", "level": "intermediate"}
            },
            {
                "name": "Vector Databases Explained",
                "description": "Comprehensive guide on vector databases",
                "metadata": {"topic": "databases", "level": "advanced"}
            }
        ]

        for doc_data in documents:
            result = await self.make_request(
                "POST",
                f"/documents/?library_id={self.library_id}",
                doc_data
            )
            self.document_ids.append(result["id"])
            print(f"✅ Document created: {result['name']}")

        print(f"📊 Total documents created: {len(self.document_ids)}")

    async def step_3_add_chunks(self):
        """Step 3: Add chunks with content"""
        print("\n📝 Step 3: Adding text chunks...")

        # Content for different documents
        content_sets = [
            # Basic Python
            [
                "Python is an interpreted, interactive, and object-oriented programming language.",
                "Variables in Python do not need to be explicitly declared. They are created automatically when a value is assigned.",
                "Basic data types in Python include integers, floats, strings, and booleans.",
                "Lists are ordered, mutable data structures that can hold elements of different types.",
                "Dictionaries are collections of key-value pairs that allow fast access to data."
            ],
            # Machine Learning
            [
                "Machine learning is a branch of artificial intelligence that enables machines to learn without explicit programming.",
                "Supervised algorithms learn from labeled data to make predictions on new data.",
                "Unsupervised algorithms find hidden patterns in unlabeled data.",
                "Overfitting occurs when a model learns the training data too well but fails on new data.",
                "Cross-validation is a technique to evaluate a model's generalization ability."
            ],
            # Vector Databases
            [
                "Vector databases are optimized to store and query high-dimensional vectors.",
                "Embeddings are dense numeric representations of data such as text, images, or audio.",
                "Cosine similarity search is a common metric for finding similar vectors.",
                "Indexes like LSH and IVF speed up searches in large vector spaces.",
                "Vector databases are essential for AI applications like RAG and semantic search."
            ]
        ]

        total_chunks = 0

        for i, (doc_id, chunks_content) in enumerate(zip(self.document_ids, content_sets)):
            created_chunks = []
            for j, content in enumerate(chunks_content):
                chunk_data = {
                    "text": content,
                    "metadata": {
                        "chunk_index": j,
                        "topic": ["python", "ml", "databases"][i],
                        "word_count": len(content.split())
                    }
                }
                result = await self.make_request(
                    "POST",
                    f"/chunks/?document_id={doc_id}",
                    chunk_data
                )

                created_chunks.append(result)
                self.chunk_ids.append(result["id"])
                print(f"✅ Created chunk {j + 1}/{len(chunks_content)} in document {i + 1}")

            total_chunks += len(result)
            print(f"✅ {len(result)} chunks created in document {i + 1}")

        print(f"📊 Total chunks created: {total_chunks}")

    async def step_4_index_library(self):
        """Step 4: Index library"""
        print("\n🔍 Step 4: Indexing library...")

        index_type = "lsh"
        print(f"📊 Creating {index_type.upper()} index...")

        start_time = time.time()
        await self.make_request(
            "POST",
            f"/libraries/{self.library_id}/index?index_type={index_type}"
        )
        end_time = time.time()

        print(f"✅ Index {index_type} created in {(end_time - start_time) * 1000:.2f}ms")

    async def step_5_search_examples(self):
        """Step 5: Search examples"""
        print("\n🔎 Step 5: Performing example searches...")

        search_queries = [
            "What are variables in programming?",
            "Explain overfitting in machine learning",
            "How do embeddings work?",
            "Vector indexing algorithms"
        ]

        for i, query in enumerate(search_queries):
            print(f"\n🔍 Search {i + 1}: '{query}'")

            start_time = time.time()
            result = await self.make_request(
                "GET",
                f"/search/libraries/{self.library_id}/simple?q={query}&k=3"
            )
            end_time = time.time()

            print(f"⏱️  Time: {result['execution_time_ms']:.2f}ms")
            print(f"📊 Results found: {len(result['results'])}")

            for j, search_result in enumerate(result['results'][:2]):  # Show only top 2
                chunk = search_result['chunk']
                score = search_result['similarity_score']
                print(f"   {j + 1}. Score: {score:.3f} | {chunk['text'][:100]}...")

    async def step_6_metadata_filtering(self):
        """Step 6: Search with metadata filters"""
        print("\n🎯 Step 6: Search with metadata filters...")

        # Search only in Python content
        metadata_filter = json.dumps({"topic": "python"})
        result = await self.make_request(
            "GET",
            f"/search/libraries/{self.library_id}/simple?q=variables&k=5&metadata_filter={metadata_filter}"
        )

        print(f"🔍 Filtered search (topic=python): {len(result['results'])} results")
        for search_result in result['results']:
            chunk = search_result['chunk']
            print(f"   📝 {chunk['text'][:80]}...")

    async def run_complete_demo(self):
        """Run complete demo"""
        print("🚀 Starting full Vector Database API demo")
        print("=" * 60)

        try:
            await self.step_1_create_library()
            await self.step_2_create_documents()
            await self.step_3_add_chunks()
            await self.step_4_index_library()
            await self.step_5_search_examples()
            await self.step_6_metadata_filtering()

            print("\n✅ Demo completed successfully!")
            print("🌐 Visit http://localhost:8000/docs for more information")

        except Exception as e:
            print(f"\n❌ Demo error: {str(e)}")
            raise


async def main():
    """Main function"""
    print("Vector Database API - Demo Script")
    print("Make sure the server is running at http://localhost:8000")

    # Check that the server is up
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{BASE_URL}/health") as response:
                if response.status != 200:
                    raise Exception("Server not available")
                print("✅ Server connected successfully\n")
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("💡 Run: uvicorn app.main:app --reload")
        return

    # Run demo
    async with VectorDBDemo() as demo:
        await demo.run_complete_demo()


if __name__ == "__main__":
    asyncio.run(main())
