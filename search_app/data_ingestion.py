import os
import json
import pandas as pd
import logging
import time
import shutil
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import traceback
from pathlib import Path
import requests
from PIL import Image
from io import BytesIO

# Configure logging
logger = logging.getLogger('search_app.data_ingestion')


class DataIngestor:
    """Class for handling data ingestion from various sources"""

    def __init__(self,
                 data_dir: str,
                 embedding_generator,
                 vector_storage):
        """
        Initialize the data ingestor

        Args:
            data_dir: Directory to store processed data
            embedding_generator: EmbeddingGenerator instance
            vector_storage: VectorStorage instance
        """
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, 'raw')
        self.processed_dir = os.path.join(data_dir, 'processed')
        self.images_dir = os.path.join(data_dir, 'images')

        # Ensure directories exist
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)

        # Store references to embedding and storage components
        self.embedding_generator = embedding_generator
        self.vector_storage = vector_storage

        # Performance metrics
        self.metrics = {
            "files_processed": 0,
            "items_processed": 0,
            "processing_times": [],
            "errors": 0
        }

    def load_coco_captions(self, annotations_file: str, images_dir: str, limit: int = 1000) -> List[Dict]:
        """
        Load data from COCO Captions dataset

        Args:
            annotations_file: Path to annotations JSON file
            images_dir: Directory containing images
            limit: Maximum number of items to load

        Returns:
            List of dictionaries with loaded data
        """
        try:
            start_time = time.time()

            logger.info(f"Loading COCO Captions from {annotations_file}")

            # Load annotations
            with open(annotations_file, 'r') as f:
                data = json.load(f)

            # Create lookup dictionaries
            image_dict = {img['id']: img for img in data['images']}

            # Process annotations
            items = []
            used_image_ids = set()

            for ann in tqdm(data['annotations'], desc="Processing COCO annotations"):
                image_id = ann['image_id']

                # Skip if we've processed enough items
                if len(items) >= limit:
                    break

                # Skip if we've already used this image (to avoid duplicates)
                if image_id in used_image_ids:
                    continue

                # Get image info
                image_info = image_dict.get(image_id)
                if not image_info:
                    continue

                # Build image path
                image_filename = image_info['file_name']
                image_path = os.path.join(images_dir, image_filename)

                # Skip if image doesn't exist
                if not os.path.exists(image_path):
                    logger.warning(f"Image not found: {image_path}")
                    continue

                # Create a new local copy in our images directory
                local_image_path = os.path.join(self.images_dir, image_filename)
                shutil.copy(image_path, local_image_path)

                # Build item
                item = {
                    'id': str(image_id),
                    'text': ann['caption'],
                    'image_path': local_image_path,
                    'metadata': {
                        'title': f"COCO Image {image_id}",
                        'source': 'coco',
                        'image_id': image_id,
                        'width': image_info.get('width'),
                        'height': image_info.get('height'),
                        'date_captured': image_info.get('date_captured')
                    }
                }

                items.append(item)
                used_image_ids.add(image_id)

            # Update metrics
            processing_time = time.time() - start_time
            self.metrics["files_processed"] += 1
            self.metrics["items_processed"] += len(items)
            self.metrics["processing_times"].append(processing_time)

            logger.info(f"Loaded {len(items)} items from COCO Captions in {processing_time:.2f} seconds")
            return items

        except Exception as e:
            logger.error(f"Error loading COCO Captions: {str(e)}")
            logger.error(traceback.format_exc())
            self.metrics["errors"] += 1
            return []

    def load_unsplash_dataset(self, csv_path: str, limit: int = 1000) -> List[Dict]:
        """
        Load data from Unsplash dataset

        Args:
            csv_path: Path to Unsplash CSV file
            limit: Maximum number of items to load

        Returns:
            List of dictionaries with loaded data
        """
        try:
            start_time = time.time()

            logger.info(f"Loading Unsplash dataset from {csv_path}")

            # Load CSV data
            df = pd.read_csv(csv_path)

            # Take a sample up to the limit
            if len(df) > limit:
                df = df.sample(limit)

            # Process items
            items = []

            for _, row in tqdm(df.iterrows(), desc="Processing Unsplash data", total=len(df)):
                # Download image if URL is provided
                image_path = None
                if 'photo_image_url' in row and row['photo_image_url']:
                    try:
                        # Create a filename from the photo ID
                        photo_id = row.get('photo_id', f"unsplash_{len(items)}")
                        image_filename = f"unsplash_{photo_id}.jpg"
                        image_path = os.path.join(self.images_dir, image_filename)

                        # Download image if it doesn't exist
                        if not os.path.exists(image_path):
                            response = requests.get(row['photo_image_url'])
                            img = Image.open(BytesIO(response.content))
                            img.save(image_path)
                            logger.debug(f"Downloaded image to {image_path}")
                    except Exception as e:
                        logger.warning(f"Could not download image: {str(e)}")
                        continue

                # Skip if we couldn't get an image
                if not image_path:
                    continue

                # Build metadata
                metadata = {}
                for col in df.columns:
                    if col not in ['photo_description', 'photo_image_url'] and not pd.isna(row[col]):
                        metadata[col] = row[col]

                # Build item
                item = {
                    'id': f"unsplash_{metadata.get('photo_id', len(items))}",
                    'text': row.get('photo_description', ''),
                    'image_path': image_path,
                    'metadata': {
                        'title': row.get('photo_description', '')[:50] + '...' if len(
                            row.get('photo_description', '')) > 50 else row.get('photo_description', ''),
                        'source': 'unsplash',
                        **metadata
                    }
                }

                items.append(item)

            # Update metrics
            processing_time = time.time() - start_time
            self.metrics["files_processed"] += 1
            self.metrics["items_processed"] += len(items)
            self.metrics["processing_times"].append(processing_time)

            logger.info(f"Loaded {len(items)} items from Unsplash in {processing_time:.2f} seconds")
            return items

        except Exception as e:
            logger.error(f"Error loading Unsplash dataset: {str(e)}")
            logger.error(traceback.format_exc())
            self.metrics["errors"] += 1
            return []

    def process_batch(self, items: List[Dict], batch_size: int = 16) -> bool:
        """
        Process a batch of items (generate embeddings and store)

        Args:
            items: List of dictionaries with 'id', 'text', and 'image_path' keys
            batch_size: Size of mini-batches for processing

        Returns:
            bool: Success status
        """
        try:
            start_time = time.time()

            logger.info(f"Processing batch of {len(items)} items")

            # Process in mini-batches
            for i in range(0, len(items), batch_size):
                mini_batch = items[i:i + batch_size]

                # Generate embeddings
                embeddings = self.embedding_generator.generate_batch_embeddings(mini_batch)

                # Prepare storage items
                storage_items = []

                for item in mini_batch:
                    item_id = item['id']

                    if item_id not in embeddings:
                        logger.warning(f"No embeddings generated for item {item_id}")
                        continue

                    # Average text and image embeddings for a combined representation
                    # This is a simple approach - more sophisticated fusion is possible
                    combined_embedding = (
                                                 embeddings[item_id]['text_embedding'] +
                                                 embeddings[item_id]['image_embedding']
                                         ) / 2.0

                    storage_items.append({
                        'id': item_id,
                        'embedding': combined_embedding,
                        'metadata': {
                            'text': item['text'],
                            'image_path': item['image_path'],
                            **item['metadata']
                        }
                    })

                # Add to storage
                if storage_items:
                    self.vector_storage.add_batch(storage_items)
                    logger.debug(f"Added {len(storage_items)} items to storage")

            # Update metrics
            processing_time = time.time() - start_time
            avg_time_per_item = processing_time / len(items)
            logger.info(f"Processed {len(items)} items in {processing_time:.2f} seconds "
                        f"({avg_time_per_item:.4f} s/item)")

            return True

        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            logger.error(traceback.format_exc())
            self.metrics["errors"] += 1
            return False

    def process_single_item(self, item: Dict) -> bool:
        """
        Process a single item (generate embeddings and store)

        Args:
            item: Dictionary with 'id', 'text', and 'image_path' keys

        Returns:
            bool: Success status
        """
        try:
            # Generate embeddings
            text_embedding = self.embedding_generator.generate_text_embedding(item['text'])
            image_embedding = self.embedding_generator.generate_image_embedding(item['image_path'])

            # Average embeddings for a combined representation
            combined_embedding = (text_embedding + image_embedding) / 2.0

            # Add to storage
            success = self.vector_storage.add_item(
                item_id=item['id'],
                embedding=combined_embedding,
                metadata={
                    'text': item['text'],
                    'image_path': item['image_path'],
                    **item['metadata']
                }
            )

            return success

        except Exception as e:
            logger.error(f"Error processing item {item.get('id', 'unknown')}: {str(e)}")
            logger.error(traceback.format_exc())
            self.metrics["errors"] += 1
            return False

    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics for the data ingestor"""
        stats = {
            "files_processed": self.metrics["files_processed"],
            "items_processed": self.metrics["items_processed"],
            "total_errors": self.metrics["errors"]
        }

        if self.metrics["processing_times"]:
            stats["avg_processing_time"] = sum(self.metrics["processing_times"]) / len(self.metrics["processing_times"])
            stats["total_processing_time"] = sum(self.metrics["processing_times"])

        return stats