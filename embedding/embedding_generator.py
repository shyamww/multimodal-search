import os
import torch
import numpy as np
import time
import logging
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from pathlib import Path
import torch.nn.functional as F
from typing import List, Dict, Union, Tuple, Optional
import traceback

# Configure logging
logger = logging.getLogger('search_app.embedding')


class EmbeddingGenerator:
    """Class for generating embeddings using CLIP model"""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32",
                 device: Optional[str] = None,
                 batch_size: int = 16):
        """
        Initialize the embedding generator

        Args:
            model_name: Name of the model to use
            device: Device to use (cpu or cuda)
            batch_size: Batch size for processing
        """
        self.model_name = model_name
        self.batch_size = batch_size

        # Determine device (M3 Mac uses MPS)
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                logger.info("Using MPS device for embeddings")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info("Using CUDA device for embeddings")
            else:
                self.device = torch.device("cpu")
                logger.info("Using CPU device for embeddings")
        else:
            self.device = torch.device(device)

        # Load model and processor
        self.load_model()

        # Track performance metrics
        self.metrics = {
            "text_embedding_times": [],
            "image_embedding_times": [],
            "batch_embedding_times": [],
            "errors": 0
        }

    def load_model(self):
        """Load the CLIP model and processor"""
        try:
            start_time = time.time()
            logger.info(f"Loading model {self.model_name}...")

            self.model = CLIPModel.from_pretrained(self.model_name)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)

            # Move model to device
            self.model.to(self.device)

            load_time = time.time() - start_time
            logger.info(f"Model loaded in {load_time:.2f} seconds")

            # Get embedding dimension from model
            self.embedding_dim = self.model.config.text_config.hidden_size
            logger.info(f"Embedding dimension: {self.embedding_dim}")

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def generate_text_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a text input

        Args:
            text: Input text

        Returns:
            numpy array of embedding vector
        """
        try:
            start_time = time.time()

            # Preprocess text
            inputs = self.processor(text=text, return_tensors="pt",
                                    padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate embedding
            with torch.no_grad():
                outputs = self.model.get_text_features(**inputs)

            # Normalize embedding
            embedding = F.normalize(outputs, p=2, dim=1).cpu().numpy()

            # Track metrics
            process_time = time.time() - start_time
            self.metrics["text_embedding_times"].append(process_time)

            logger.debug(f"Text embedding generated in {process_time:.4f} seconds")
            return embedding[0]  # Return the first (and only) embedding

        except Exception as e:
            logger.error(f"Error generating text embedding: {str(e)}")
            logger.error(traceback.format_exc())
            self.metrics["errors"] += 1
            raise

    def generate_image_embedding(self, image_path: str) -> np.ndarray:
        """
        Generate embedding for an image input

        Args:
            image_path: Path to image file

        Returns:
            numpy array of embedding vector
        """
        try:
            start_time = time.time()

            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate embedding
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)

            # Normalize embedding
            embedding = F.normalize(outputs, p=2, dim=1).cpu().numpy()

            # Track metrics
            process_time = time.time() - start_time
            self.metrics["image_embedding_times"].append(process_time)

            logger.debug(f"Image embedding generated in {process_time:.4f} seconds")
            return embedding[0]  # Return the first (and only) embedding

        except Exception as e:
            logger.error(f"Error generating image embedding: {str(e)}")
            logger.error(traceback.format_exc())
            self.metrics["errors"] += 1
            raise

    def generate_batch_embeddings(self,
                                  items: List[Dict[str, str]]) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for a batch of items

        Args:
            items: List of dictionaries with 'id', 'text', and 'image_path' keys

        Returns:
            Dictionary mapping item IDs to embeddings (text and image)
        """
        try:
            batch_start_time = time.time()
            results = {}

            # Process in batches
            for i in range(0, len(items), self.batch_size):
                batch = items[i:i + self.batch_size]

                # Process text embeddings
                batch_texts = [item['text'] for item in batch]
                text_inputs = self.processor(text=batch_texts, return_tensors="pt",
                                             padding=True, truncation=True)
                text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

                # Process image embeddings
                batch_images = [Image.open(item['image_path']).convert("RGB") for item in batch]
                image_inputs = self.processor(images=batch_images, return_tensors="pt")
                image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}

                # Generate embeddings
                with torch.no_grad():
                    text_outputs = self.model.get_text_features(**text_inputs)
                    image_outputs = self.model.get_image_features(**image_inputs)

                # Normalize embeddings
                text_embeddings = F.normalize(text_outputs, p=2, dim=1).cpu().numpy()
                image_embeddings = F.normalize(image_outputs, p=2, dim=1).cpu().numpy()

                # Store results
                for j, item in enumerate(batch):
                    item_id = item['id']
                    results[item_id] = {
                        'text_embedding': text_embeddings[j],
                        'image_embedding': image_embeddings[j]
                    }

            # Track metrics
            batch_time = time.time() - batch_start_time
            self.metrics["batch_embedding_times"].append(batch_time)

            avg_time_per_item = batch_time / len(items)
            logger.info(f"Batch processed {len(items)} items in {batch_time:.2f} seconds "
                        f"({avg_time_per_item:.4f} s/item)")

            return results

        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            logger.error(traceback.format_exc())
            self.metrics["errors"] += 1
            raise

    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics for the embedding generator"""
        stats = {}

        if self.metrics["text_embedding_times"]:
            stats["avg_text_embedding_time"] = np.mean(self.metrics["text_embedding_times"])
            stats["min_text_embedding_time"] = np.min(self.metrics["text_embedding_times"])
            stats["max_text_embedding_time"] = np.max(self.metrics["text_embedding_times"])

        if self.metrics["image_embedding_times"]:
            stats["avg_image_embedding_time"] = np.mean(self.metrics["image_embedding_times"])
            stats["min_image_embedding_time"] = np.min(self.metrics["image_embedding_times"])
            stats["max_image_embedding_time"] = np.max(self.metrics["image_embedding_times"])

        if self.metrics["batch_embedding_times"]:
            stats["avg_batch_time"] = np.mean(self.metrics["batch_embedding_times"])
            stats["total_batch_time"] = np.sum(self.metrics["batch_embedding_times"])

        stats["total_errors"] = self.metrics["errors"]

        return stats