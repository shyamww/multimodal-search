import os
import json
import faiss
import numpy as np
import logging
import time
from typing import List, Dict, Union, Tuple, Optional
import traceback

# Configure logging
logger = logging.getLogger('search_app.vector_storage')


class VectorStorage:
    """Class for managing vector storage using FAISS"""

    def __init__(self, index_path: str, metadata_path: str, dimension: int = 512):
        """
        Initialize the vector storage

        Args:
            index_path: Path to store the FAISS index
            metadata_path: Path to store the metadata
            dimension: Dimension of the embedding vectors
        """
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.dimension = dimension

        # Ensure directories exist
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

        # Initialize storage
        self.initialize_storage()

        # Performance metrics
        self.metrics = {
            "add_times": [],
            "search_times": [],
            "errors": 0
        }

    def initialize_storage(self):
        """Initialize or load the vector index and metadata"""
        try:
            # Check if index exists
            if os.path.exists(self.index_path):
                logger.info(f"Loading existing index from {self.index_path}")
                self.index = faiss.read_index(self.index_path)
                logger.info(f"Index loaded with {self.index.ntotal} vectors")
            else:
                logger.info(f"Creating new index with dimension {self.dimension}")
                # Create a new index - using L2 distance and flat index for simplicity
                self.index = faiss.IndexFlatL2(self.dimension)
                faiss.write_index(self.index, self.index_path)
                logger.info("New index created")

            # Load metadata if exists
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"Metadata loaded with {len(self.metadata)} entries")
            else:
                self.metadata = {
                    "ids": [],
                    "items": {}
                }
                with open(self.metadata_path, 'w') as f:
                    json.dump(self.metadata, f)
                logger.info("New metadata file created")

        except Exception as e:
            logger.error(f"Error initializing storage: {str(e)}")
            logger.error(traceback.format_exc())
            self.metrics["errors"] += 1
            raise

    def save(self):
        """Save the index and metadata to disk"""
        try:
            faiss.write_index(self.index, self.index_path)

            with open(self.metadata_path, 'w') as f:
                json.dump(self.metadata, f)

            logger.info(f"Index and metadata saved. Total vectors: {self.index.ntotal}")

        except Exception as e:
            logger.error(f"Error saving storage: {str(e)}")
            logger.error(traceback.format_exc())
            self.metrics["errors"] += 1
            raise

    def add_item(self, item_id: str,
                 embedding: np.ndarray,
                 metadata: Dict) -> bool:
        """
        Add an item to the vector storage

        Args:
            item_id: Unique identifier for the item
            embedding: Embedding vector (normalized)
            metadata: Associated metadata

        Returns:
            bool: Success status
        """
        try:
            start_time = time.time()

            # Convert to numpy array if needed
            embedding_array = np.array(embedding).astype('float32').reshape(1, -1)

            # Add to FAISS index
            self.index.add(embedding_array)

            # Get the index ID (position in the index)
            index_id = self.index.ntotal - 1

            # Add to metadata
            self.metadata["ids"].append(item_id)
            self.metadata["items"][item_id] = {
                "metadata": metadata,
                "index_id": index_id
            }

            # Save changes
            self.save()

            # Track metrics
            add_time = time.time() - start_time
            self.metrics["add_times"].append(add_time)

            logger.debug(f"Item {item_id} added in {add_time:.4f} seconds")
            return True

        except Exception as e:
            logger.error(f"Error adding item {item_id}: {str(e)}")
            logger.error(traceback.format_exc())
            self.metrics["errors"] += 1
            return False

    def add_batch(self, items: List[Dict]) -> bool:
        """
        Add a batch of items to the vector storage

        Args:
            items: List of dictionaries with 'id', 'embedding', and 'metadata' keys

        Returns:
            bool: Success status
        """
        try:
            start_time = time.time()

            # Prepare batch data
            batch_ids = []
            batch_embeddings = []

            for item in items:
                batch_ids.append(item['id'])
                batch_embeddings.append(item['embedding'])

            # Convert to numpy array
            batch_embeddings_array = np.array(batch_embeddings).astype('float32')

            # Add to FAISS index
            prev_total = self.index.ntotal
            self.index.add(batch_embeddings_array)

            # Add to metadata
            for i, item_id in enumerate(batch_ids):
                index_id = prev_total + i
                self.metadata["ids"].append(item_id)
                self.metadata["items"][item_id] = {
                    "metadata": items[i]['metadata'],
                    "index_id": index_id
                }

            # Save changes
            self.save()

            # Track metrics
            batch_time = time.time() - start_time
            avg_time = batch_time / len(items)
            self.metrics["add_times"].append(avg_time)

            logger.info(f"Batch of {len(items)} items added in {batch_time:.4f} seconds "
                        f"({avg_time:.4f} s/item)")
            return True

        except Exception as e:
            logger.error(f"Error adding batch: {str(e)}")
            logger.error(traceback.format_exc())
            self.metrics["errors"] += 1
            return False

    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Dict]:
        """
        Search for similar items

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return

        Returns:
            List of dictionaries with search results
        """
        try:
            start_time = time.time()

            # Ensure we don't request more items than we have
            k = min(k, self.index.ntotal)

            if k == 0:
                logger.warning("No items in index, returning empty results")
                return []

            # Convert to numpy array if needed
            query_array = np.array(query_embedding).astype('float32').reshape(1, -1)

            # Perform search
            distances, indices = self.index.search(query_array, k)

            # Build results
            results = []
            for i, idx in enumerate(indices[0]):
                # Get the item ID from the index position
                item_id = self.metadata["ids"][idx]

                # Get item metadata
                item_data = self.metadata["items"][item_id]

                # Create result object
                result = {
                    "id": item_id,
                    "distance": float(distances[0][i]),  # Convert numpy float to Python float
                    "metadata": item_data["metadata"]
                }

                results.append(result)

            # Track metrics
            search_time = time.time() - start_time
            self.metrics["search_times"].append(search_time)

            logger.debug(f"Search completed in {search_time:.4f} seconds")
            return results

        except Exception as e:
            logger.error(f"Error searching: {str(e)}")
            logger.error(traceback.format_exc())
            self.metrics["errors"] += 1
            return []

    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics for the vector storage"""
        stats = {}

        if self.metrics["add_times"]:
            stats["avg_add_time"] = np.mean(self.metrics["add_times"])
            stats["min_add_time"] = np.min(self.metrics["add_times"])
            stats["max_add_time"] = np.max(self.metrics["add_times"])

        if self.metrics["search_times"]:
            stats["avg_search_time"] = np.mean(self.metrics["search_times"])
            stats["min_search_time"] = np.min(self.metrics["search_times"])
            stats["max_search_time"] = np.max(self.metrics["search_times"])

        stats["total_errors"] = self.metrics["errors"]
        stats["total_items"] = self.index.ntotal

        return stats

    def get_item_count(self) -> int:
        """Get the number of items in the storage"""
        return self.index.ntotal