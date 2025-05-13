import time
import logging
import psutil
import os
import json
from typing import List, Dict, Optional, Union
import traceback

# Configure logging
logger = logging.getLogger('search_app.search_engine')


class SearchEngine:
    """Main search engine class that coordinates embedding generation and vector search"""

    def __init__(self, embedding_generator, vector_storage):
        """
        Initialize the search engine

        Args:
            embedding_generator: EmbeddingGenerator instance
            vector_storage: VectorStorage instance
        """
        self.embedding_generator = embedding_generator
        self.vector_storage = vector_storage

        # Performance metrics
        self.metrics = {
            "queries_processed": 0,
            "query_times": [],
            "total_results": 0,
            "errors": 0,
            "memory_usage": []
        }

    def search_by_text(self, query_text: str, k: int = 10) -> Dict:
        """
        Search for items matching a text query

        Args:
            query_text: Text query
            k: Number of results to return

        Returns:
            Dictionary with search results and metadata
        """
        try:
            start_time = time.time()

            # Track memory usage before query
            memory_before = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB

            # Generate query embedding
            query_embedding = self.embedding_generator.generate_text_embedding(query_text)

            # Search vector storage
            search_results = self.vector_storage.search(query_embedding, k=k)

            # Track memory usage after query
            memory_after = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
            memory_delta = memory_after - memory_before

            # Prepare response
            query_time = time.time() - start_time
            response = {
                "query": query_text,
                "results": search_results,
                "result_count": len(search_results),
                "query_time_ms": query_time * 1000,
                "performance": {
                    "embedding_time_ms": self.embedding_generator.metrics["text_embedding_times"][-1] * 1000 if
                    self.embedding_generator.metrics["text_embedding_times"] else 0,
                    "search_time_ms": self.vector_storage.metrics["search_times"][-1] * 1000 if
                    self.vector_storage.metrics["search_times"] else 0,
                    "memory_used_mb": memory_delta
                }
            }

            # Update metrics
            self.metrics["queries_processed"] += 1
            self.metrics["query_times"].append(query_time)
            self.metrics["total_results"] += len(search_results)
            self.metrics["memory_usage"].append(memory_delta)

            logger.info(f"Text search for '{query_text}' completed in {query_time:.4f} seconds")
            logger.info(f"Found {len(search_results)} results")

            return response

        except Exception as e:
            logger.error(f"Error in text search: {str(e)}")
            logger.error(traceback.format_exc())
            self.metrics["errors"] += 1

            return {
                "query": query_text,
                "results": [],
                "result_count": 0,
                "query_time_ms": 0,
                "error": str(e)
            }

    def search_by_image(self, image_path: str, k: int = 10) -> Dict:
        """
        Search for items matching an image query

        Args:
            image_path: Path to query image
            k: Number of results to return

        Returns:
            Dictionary with search results and metadata
        """
        try:
            start_time = time.time()

            # Track memory usage before query
            memory_before = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB

            # Generate query embedding
            query_embedding = self.embedding_generator.generate_image_embedding(image_path)

            # Search vector storage
            search_results = self.vector_storage.search(query_embedding, k=k)

            # Track memory usage after query
            memory_after = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
            memory_delta = memory_after - memory_before

            # Prepare response
            query_time = time.time() - start_time
            response = {
                "query_image": image_path,
                "results": search_results,
                "result_count": len(search_results),
                "query_time_ms": query_time * 1000,
                "performance": {
                    "embedding_time_ms": self.embedding_generator.metrics["image_embedding_times"][-1] * 1000 if
                    self.embedding_generator.metrics["image_embedding_times"] else 0,
                    "search_time_ms": self.vector_storage.metrics["search_times"][-1] * 1000 if
                    self.vector_storage.metrics["search_times"] else 0,
                    "memory_used_mb": memory_delta
                }
            }

            # Update metrics
            self.metrics["queries_processed"] += 1
            self.metrics["query_times"].append(query_time)
            self.metrics["total_results"] += len(search_results)
            self.metrics["memory_usage"].append(memory_delta)

            logger.info(f"Image search completed in {query_time:.4f} seconds")
            logger.info(f"Found {len(search_results)} results")

            return response

        except Exception as e:
            logger.error(f"Error in image search: {str(e)}")
            logger.error(traceback.format_exc())
            self.metrics["errors"] += 1

            return {
                "query_image": image_path,
                "results": [],
                "result_count": 0,
                "query_time_ms": 0,
                "error": str(e)
            }

    def get_performance_stats(self) -> Dict:
        """Get performance statistics for the search engine"""

        stats = {
            "queries_processed": self.metrics["queries_processed"],
            "errors": self.metrics["errors"]
        }

        if self.metrics["query_times"]:
            import numpy as np
            stats["avg_query_time_ms"] = np.mean(self.metrics["query_times"]) * 1000
            stats["min_query_time_ms"] = np.min(self.metrics["query_times"]) * 1000
            stats["max_query_time_ms"] = np.max(self.metrics["query_times"]) * 1000

        if self.metrics["memory_usage"]:
            import numpy as np
            stats["avg_memory_usage_mb"] = np.mean(self.metrics["memory_usage"])
            stats["total_memory_usage_mb"] = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

        # Add embedding and vector storage stats
        stats["embedding_stats"] = self.embedding_generator.get_performance_stats()
        stats["storage_stats"] = self.vector_storage.get_performance_stats()

        return stats

    def get_system_info(self) -> Dict:
        """Get system information for monitoring"""

        # Get CPU info
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count()

        # Get memory info
        memory = psutil.virtual_memory()
        memory_used_percent = memory.percent
        memory_used_gb = memory.used / 1024 / 1024 / 1024
        memory_total_gb = memory.total / 1024 / 1024 / 1024

        # Get disk info
        disk = psutil.disk_usage('/')
        disk_used_percent = disk.percent
        disk_used_gb = disk.used / 1024 / 1024 / 1024
        disk_total_gb = disk.total / 1024 / 1024 / 1024

        return {
            "cpu": {
                "percent": cpu_percent,
                "count": cpu_count
            },
            "memory": {
                "percent": memory_used_percent,
                "used_gb": memory_used_gb,
                "total_gb": memory_total_gb
            },
            "disk": {
                "percent": disk_used_percent,
                "used_gb": disk_used_gb,
                "total_gb": disk_total_gb
            },
            "timestamp": time.time()
        }