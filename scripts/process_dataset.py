import os
import sys
import argparse
import logging
import time
from pathlib import Path

# Add the project root to the Python path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR, "logs", "data_processing.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('data_processing')

# Import project components
from embedding.embedding_generator import EmbeddingGenerator
from storage.vector_storage import VectorStorage
from search_app.data_ingestion import DataIngestor


def main():
    parser = argparse.ArgumentParser(description='Process and index datasets for multimodal search')
    parser.add_argument('--dataset', choices=['coco', 'unsplash'], required=True,
                        help='Dataset type to process')
    parser.add_argument('--path', required=True,
                        help='Path to dataset file')
    parser.add_argument('--images-dir',
                        help='Path to images directory (required for COCO dataset)')
    parser.add_argument('--limit', type=int, default=1000,
                        help='Maximum number of items to process')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for processing')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default=None,
                        help='Device to use for embedding generation')

    args = parser.parse_args()

    # Validate arguments
    if args.dataset == 'coco' and not args.images_dir:
        parser.error("--images-dir is required for COCO dataset")

    # Setup components
    logger.info("Initializing components...")

    data_dir = os.path.join(BASE_DIR, "data")

    # Initialize embedding generator
    embedding_generator = EmbeddingGenerator(
        model_name="openai/clip-vit-base-patch32",
        device=args.device,
        batch_size=args.batch_size
    )

    # Initialize vector storage
    vector_storage = VectorStorage(
        index_path=os.path.join(data_dir, "vector_db", "index.faiss"),
        metadata_path=os.path.join(data_dir, "vector_db", "metadata.json"),
        dimension=embedding_generator.embedding_dim
    )

    # Initialize data ingestor
    data_ingestor = DataIngestor(
        data_dir=data_dir,
        embedding_generator=embedding_generator,
        vector_storage=vector_storage
    )

    # Process dataset
    logger.info(f"Processing {args.dataset} dataset from {args.path}")
    start_time = time.time()

    if args.dataset == 'coco':
        items = data_ingestor.load_coco_captions(args.path, args.images_dir, limit=args.limit)
    elif args.dataset == 'unsplash':
        items = data_ingestor.load_unsplash_dataset(args.path, limit=args.limit)

    if items:
        logger.info(f"Processing {len(items)} items...")
        data_ingestor.process_batch(items, batch_size=args.batch_size)

        # Log performance statistics
        processing_time = time.time() - start_time
        logger.info(f"Finished processing in {processing_time:.2f} seconds")

        # Log performance metrics
        embedding_stats = embedding_generator.get_performance_stats()
        storage_stats = vector_storage.get_performance_stats()

        logger.info(f"Embedding statistics: {embedding_stats}")
        logger.info(f"Storage statistics: {storage_stats}")

        logger.info(f"Total items in index: {vector_storage.get_item_count()}")
    else:
        logger.warning("No items loaded from dataset")


if __name__ == "__main__":
    main()