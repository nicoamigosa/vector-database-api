import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from abc import ABC, abstractmethod

from app.config import settings, get_available_index_types
from app.models import Chunk, SearchResult
from app.utils.embedding import VectorOperations
from app.utils.concurrency import ReadWriteLock
import logging
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


class VectorIndex(ABC):
    """Abstract base class for vector indexes"""

    @abstractmethod
    def add_vector(self, chunk_id: str, vector: List[float]) -> None:
        """Add a vector to the index"""
        pass

    @abstractmethod
    def remove_vector(self, chunk_id: str) -> bool:
        """Remove a vector from the index"""
        pass

    @abstractmethod
    def search(self, query_vector: List[float], k: int, chunks_dict: Dict[str, Chunk]) -> List[SearchResult]:
        """Search for k nearest neighbors"""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear the index"""
        pass

    @abstractmethod
    def size(self) -> int:
        """Get number of vectors in index"""
        pass


class LSHIndex(VectorIndex):
    """
    Locality Sensitive Hashing (LSH) Index

    Time Complexity:
    - Add: O(L*k) where L=number of hash tables, k=hash length
    - Remove: O(L*k)
    - Search: O(L*k + C) where C=average bucket size

    Space Complexity: O(L*2^k + n*d)

    LSH hash similar vectors into the same buckets to speed up similarity search.
    Trade-off between speed and accuracy. Good for large datasets.
    """

    def __init__(self, num_tables: int = 8, hash_length: int = 12, vector_dim: int = 1024):
        self.num_tables = num_tables
        self.hash_length = hash_length
        self.vector_dim = vector_dim

        # Generate random hyperplanes for hashing
        self.hyperplanes = []
        for _ in range(num_tables):
            table_planes = []
            for _ in range(hash_length):
                # Random hyperplane (normal vector)
                plane = np.random.normal(0, 1, vector_dim)
                plane = plane / np.linalg.norm(plane)  # Normalize
                table_planes.append(plane)
            self.hyperplanes.append(table_planes)

        # Hash tables - each table maps hash -> set of chunk_ids
        self.hash_tables: List[Dict[str, Set[str]]] = [
            {} for _ in range(num_tables)
        ]

        # Store actual vectors for final similarity calculation
        self.vectors: Dict[str, List[float]] = {}
        self._lock = ReadWriteLock()

        logger.info(f"LSHIndex initialized with {num_tables} tables, hash length {hash_length}")

    def _hash_vector(self, vector: List[float]) -> List[str]:
        """Generate LSH hashes for a vector"""
        hashes = []
        vector_np = np.array(vector)

        for table_idx in range(self.num_tables):
            hash_bits = []
            for plane in self.hyperplanes[table_idx]:
                # Project vector onto hyperplane and check sign
                projection = np.dot(vector_np, plane)
                hash_bits.append('1' if projection >= 0 else '0')

            hash_str = ''.join(hash_bits)
            hashes.append(hash_str)

        return hashes

    def add_vector(self, chunk_id: str, vector: List[float]) -> None:
        """Add vector to LSH index"""
        with self._lock.write_lock():
            # Store the actual vector
            self.vectors[chunk_id] = vector.copy()

            # Generate hashes and add to tables
            hashes = self._hash_vector(vector)
            for table_idx, hash_val in enumerate(hashes):
                if hash_val not in self.hash_tables[table_idx]:
                    self.hash_tables[table_idx][hash_val] = set()
                self.hash_tables[table_idx][hash_val].add(chunk_id)

    def remove_vector(self, chunk_id: str) -> bool:
        """Remove vector from LSH index"""
        with self._lock.write_lock():
            if chunk_id not in self.vectors:
                return False

            vector = self.vectors[chunk_id]
            hashes = self._hash_vector(vector)

            # Remove from hash tables
            for table_idx, hash_val in enumerate(hashes):
                if hash_val in self.hash_tables[table_idx]:
                    self.hash_tables[table_idx][hash_val].discard(chunk_id)
                    # Clean up empty buckets
                    if not self.hash_tables[table_idx][hash_val]:
                        del self.hash_tables[table_idx][hash_val]

            # Remove actual vector
            del self.vectors[chunk_id]
            return True

    def search(self, query_vector: List[float], k: int, chunks_dict: Dict[str, Chunk]) -> List[SearchResult]:
        """Search using LSH"""
        with self._lock.read_lock():
            if not self.vectors:
                return []

            # Get candidate chunk IDs from hash tables
            candidates = set()
            query_hashes = self._hash_vector(query_vector)

            for table_idx, hash_val in enumerate(query_hashes):
                if hash_val in self.hash_tables[table_idx]:
                    candidates.update(self.hash_tables[table_idx][hash_val])

            # If no candidates found, fall back to checking all vectors
            if not candidates:
                candidates = set(self.vectors.keys())

            # Calculate exact similarities for candidates
            similarities = []
            for chunk_id in candidates:
                if chunk_id in chunks_dict and chunk_id in self.vectors:
                    vector = self.vectors[chunk_id]
                    similarity = VectorOperations.cosine_similarity(query_vector, vector)
                    distance = VectorOperations.euclidean_distance(query_vector, vector)
                    similarities.append((chunk_id, similarity, distance))

            # Sort and return top k
            similarities.sort(key=lambda x: x[1], reverse=True)

            results = []
            for chunk_id, similarity, distance in similarities[:k]:
                chunk = chunks_dict.get(chunk_id)
                if chunk:
                    result = SearchResult(
                        chunk=chunk,
                        similarity_score=similarity,
                        distance=distance
                    )
                    results.append(result)

            return results

    def clear(self) -> None:
        """Clear all data"""
        with self._lock.write_lock():
            self.vectors.clear()
            for table in self.hash_tables:
                table.clear()

    def size(self) -> int:
        """Get number of vectors"""
        with self._lock.read_lock():
            return len(self.vectors)


class IVFIndex(VectorIndex):
    """
    Inverted File (IVF) Index

    Time Complexity:
    - Add: O(d*nlist) where d=vector dimension, nlist=number of clusters
    - Remove: O(d*nlist)
    - Search: O(d*nprobe + C*d) where nprobe=clusters to search, C=avg cluster size

    Space Complexity: O(n*d + nlist*d)

    IVF uses k-means clustering to organize similar vectors into the same cluster.
    Search only examines vectors in the nearest clusters.    """

    def __init__(self, nlist: int = 100, nprobe: int = 5, vector_dim: int = 1024):
        self.nlist = nlist  # Number of clusters
        self.nprobe = nprobe  # Number of clusters to search
        self.vector_dim = vector_dim

        # Cluster centroids
        self.centroids: Optional[np.ndarray] = None

        # Inverted lists - cluster_id -> list of (chunk_id, vector)
        self.inverted_lists: List[List[Tuple[str, List[float]]]] = [
            [] for _ in range(nlist)
        ]

        # Keep track of which cluster each chunk belongs to
        self.chunk_to_cluster: Dict[str, int] = {}

        self._lock = ReadWriteLock()
        self._is_trained = False

        logger.info(f"IVFIndex initialized with {nlist} clusters, searching {nprobe} clusters")

    def _train_centroids(self, vectors: List[List[float]]) -> None:
        """Train k-means centroids on the data"""
        if len(vectors) < self.nlist:
            # Not enough data for clustering, use random centroids
            self.centroids = np.random.normal(0, 1, (self.nlist, self.vector_dim))
            for i in range(self.nlist):
                self.centroids[i] = self.centroids[i] / np.linalg.norm(self.centroids[i])
        else:
            # Simple k-means implementation
            vectors_np = np.array(vectors)

            # Initialize centroids randomly
            indices = np.random.choice(len(vectors), self.nlist, replace=False)
            self.centroids = vectors_np[indices].copy()

            # K-means iterations
            for iteration in range(10):  # Max 10 iterations
                # Assign points to clusters
                distances = cdist(vectors_np, self.centroids)
                assignments = np.argmin(distances, axis=1)

                # Update centroids
                new_centroids = np.zeros_like(self.centroids)
                for cluster_id in range(self.nlist):
                    cluster_points = vectors_np[assignments == cluster_id]
                    if len(cluster_points) > 0:
                        new_centroids[cluster_id] = np.mean(cluster_points, axis=0)
                    else:
                        new_centroids[cluster_id] = self.centroids[cluster_id]

                # Check convergence
                if np.allclose(self.centroids, new_centroids, rtol=1e-4):
                    break

                self.centroids = new_centroids

        self._is_trained = True
        logger.info("IVF centroids trained")

    def _find_nearest_cluster(self, vector: List[float]) -> int:
        """Find the nearest cluster for a vector"""
        if not self._is_trained:
            return 0

        vector_np = np.array(vector)
        distances = np.linalg.norm(self.centroids - vector_np, axis=1)
        return int(np.argmin(distances))

    def _retrain_if_needed(self):
        """Retrain centroids if we have enough new data"""
        total_vectors = sum(len(inv_list) for inv_list in self.inverted_lists)

        if total_vectors >= self.nlist and not self._is_trained:
            # Collect all vectors for training
            all_vectors = []
            for inv_list in self.inverted_lists:
                for _, vector in inv_list:
                    all_vectors.append(vector)

            # Retrain centroids
            self._train_centroids(all_vectors)

            # Reassign all vectors to new clusters
            self._reassign_all_vectors()

    def _reassign_all_vectors(self):
        """Reassign all vectors to clusters after retraining"""
        # Collect all vectors
        all_items = []
        for inv_list in self.inverted_lists:
            all_items.extend(inv_list)

        # Clear inverted lists
        for inv_list in self.inverted_lists:
            inv_list.clear()

        # Reassign to new clusters
        for chunk_id, vector in all_items:
            cluster_id = self._find_nearest_cluster(vector)
            self.inverted_lists[cluster_id].append((chunk_id, vector))
            self.chunk_to_cluster[chunk_id] = cluster_id

    def add_vector(self, chunk_id: str, vector: List[float]) -> None:
        """Add vector to IVF index"""
        with self._lock.write_lock():
            # Find nearest cluster
            cluster_id = self._find_nearest_cluster(vector)

            # Add to inverted list
            self.inverted_lists[cluster_id].append((chunk_id, vector.copy()))
            self.chunk_to_cluster[chunk_id] = cluster_id

            # Retrain if needed
            self._retrain_if_needed()

    def remove_vector(self, chunk_id: str) -> bool:
        """Remove vector from IVF index"""
        with self._lock.write_lock():
            if chunk_id not in self.chunk_to_cluster:
                return False

            cluster_id = self.chunk_to_cluster[chunk_id]

            # Remove from inverted list
            self.inverted_lists[cluster_id] = [
                (cid, vec) for cid, vec in self.inverted_lists[cluster_id]
                if cid != chunk_id
            ]

            del self.chunk_to_cluster[chunk_id]
            return True

    def search(self, query_vector: List[float], k: int, chunks_dict: Dict[str, Chunk]) -> List[SearchResult]:
        """Search using IVF"""
        with self._lock.read_lock():
            if not self._is_trained or not any(self.inverted_lists):
                return []

            # Find nearest clusters to search
            query_np = np.array(query_vector)
            distances = np.linalg.norm(self.centroids - query_np, axis=1)
            nearest_clusters = np.argsort(distances)[:self.nprobe]

            # Collect candidates from nearest clusters
            candidates = []
            for cluster_id in nearest_clusters:
                for chunk_id, vector in self.inverted_lists[cluster_id]:
                    if chunk_id in chunks_dict:
                        candidates.append((chunk_id, vector))

            # Calculate similarities
            similarities = []
            for chunk_id, vector in candidates:
                similarity = VectorOperations.cosine_similarity(query_vector, vector)
                distance = VectorOperations.euclidean_distance(query_vector, vector)
                similarities.append((chunk_id, similarity, distance))

            # Sort and return top k
            similarities.sort(key=lambda x: x[1], reverse=True)

            results = []
            for chunk_id, similarity, distance in similarities[:k]:
                chunk = chunks_dict.get(chunk_id)
                if chunk:
                    result = SearchResult(
                        chunk=chunk,
                        similarity_score=similarity,
                        distance=distance
                    )
                    results.append(result)

            return results

    def clear(self) -> None:
        """Clear all data"""
        with self._lock.write_lock():
            for inv_list in self.inverted_lists:
                inv_list.clear()
            self.chunk_to_cluster.clear()
            self.centroids = None
            self._is_trained = False

    def size(self) -> int:
        """Get number of vectors"""
        with self._lock.read_lock():
            return len(self.chunk_to_cluster)


class IndexManager:
    """
    Manager class to handle multiple index types and provide unified interface
    """

    def __init__(self, index_type: str = "flat"):
        self.index_type = index_type
        self.index = self._create_index(index_type)
        self._lock = ReadWriteLock()

        logger.info(f"IndexManager initialized with {index_type} index")

    def _create_index(self, index_type: str) -> VectorIndex:
        """Create index based on type using config"""
        if index_type == "lsh":
            return LSHIndex(
                num_tables=settings.lsh_num_tables,
                hash_length=settings.lsh_hash_length
            )
        elif index_type == "ivf":
            return IVFIndex(
                nlist=settings.ivf_nlist,
                nprobe=settings.ivf_nprobe
            )
        else:
            available_types = ", ".join(get_available_index_types())
            raise ValueError(f"Unknown index type: {index_type}. Available: {available_types}")

    def add_chunk(self, chunk: Chunk) -> None:
        """Add chunk to index"""
        if chunk.embedding:
            self.index.add_vector(chunk.id, chunk.embedding)

    def remove_chunk(self, chunk_id: str) -> bool:
        """Remove chunk from index"""
        return self.index.remove_vector(chunk_id)

    def search(self, query_vector: List[float], k: int, chunks_dict: Dict[str, Chunk]) -> List[SearchResult]:
        """Search for similar vectors"""
        return self.index.search(query_vector, k, chunks_dict)

    def rebuild_index(self, chunks: List[Chunk], new_index_type: str = None) -> None:
        """Rebuild index with new type or refresh current"""
        with self._lock.write_lock():
            if new_index_type and new_index_type != self.index_type:
                self.index_type = new_index_type
                self.index = self._create_index(new_index_type)
                logger.info(f"Switched to {new_index_type} index")
            else:
                self.index.clear()
                logger.info(f"Cleared {self.index_type} index")

            # Re-add all chunks with embeddings
            for chunk in chunks:
                if chunk.embedding:
                    self.index.add_vector(chunk.id, chunk.embedding)

            logger.info(f"Rebuilt index with {len(chunks)} chunks")