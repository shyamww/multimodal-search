import os
import time
import tempfile
import logging
from typing import List, Dict, Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import uvicorn
import json
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('search_api')

# Import your search engine components
from embedding.embedding_generator import EmbeddingGenerator
from storage.vector_storage import VectorStorage
from search_app.search_engine import SearchEngine
from search_app.data_ingestion import DataIngestor

# Create the API app
app = FastAPI(
    title="Multimodal Search API",
    description="API for searching text and images using embeddings",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = os.path.join(BASE_DIR, "data")
VECTOR_DB_PATH = os.path.join(DATA_DIR, "vector_db", "index.faiss")
METADATA_PATH = os.path.join(DATA_DIR, "vector_db", "metadata.json")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
MEDIA_DIR = os.path.join(BASE_DIR, "media")

# Ensure directories exist
os.makedirs(os.path.join(DATA_DIR, "vector_db"), exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MEDIA_DIR, exist_ok=True)

# Initialize components
embedding_generator = None
vector_storage = None
search_engine = None
data_ingestor = None


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global embedding_generator, vector_storage, search_engine, data_ingestor

    logger.info("Initializing search API components...")

    # Initialize embedding generator (use MPS for M3 Mac)
    logger.info("Loading embedding model...")
    embedding_generator = EmbeddingGenerator(
        model_name="openai/clip-vit-base-patch32",
        batch_size=16
    )

    # Initialize vector storage
    logger.info("Initializing vector storage...")
    vector_storage = VectorStorage(
        index_path=VECTOR_DB_PATH,
        metadata_path=METADATA_PATH,
        dimension=embedding_generator.embedding_dim
    )

    # Initialize search engine
    logger.info("Setting up search engine...")
    search_engine = SearchEngine(
        embedding_generator=embedding_generator,
        vector_storage=vector_storage
    )

    # Initialize data ingestor
    logger.info("Setting up data ingestor...")
    data_ingestor = DataIngestor(
        data_dir=DATA_DIR,
        embedding_generator=embedding_generator,
        vector_storage=vector_storage
    )

    logger.info(f"Initialization complete. Vector storage has {vector_storage.get_item_count()} items.")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Multimodal Search API is running",
        "endpoints": {
            "search_text": "/api/search/text?query=your_query",
            "search_image": "/api/search/image [POST]",
            "upload_item": "/api/upload [POST]",
            "status": "/api/status",
            "docs": "/docs"
        }
    }


@app.get("/api/search/text")
async def search_text(query: str, limit: int = Query(10, ge=1, le=100)):
    """
    Search by text query

    Args:
        query: Text query string
        limit: Maximum number of results to return

    Returns:
        Search results
    """
    if not query:
        raise HTTPException(status_code=400, detail="Query text is required")

    start_time = time.time()
    results = search_engine.search_by_text(query, k=limit)

    # Add API processing time
    results["api_time_ms"] = (time.time() - start_time) * 1000

    return results


@app.post("/api/search/image")
async def search_image(
        image: UploadFile = File(...),
        limit: int = Form(10)
):
    """
    Search by image query

    Args:
        image: Uploaded image file
        limit: Maximum number of results to return

    Returns:
        Search results
    """
    if not image:
        raise HTTPException(status_code=400, detail="Image file is required")

    start_time = time.time()

    # Save uploaded image to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_image:
        shutil.copyfileobj(image.file, temp_image)
        temp_image_path = temp_image.name

    try:
        # Search using the image
        results = search_engine.search_by_image(temp_image_path, k=limit)

        # Add API processing time
        results["api_time_ms"] = (time.time() - start_time) * 1000

        return results

    finally:
        # Clean up temporary file
        if os.path.exists(temp_image_path):
            os.unlink(temp_image_path)


@app.post("/api/upload")
async def upload_item(
        background_tasks: BackgroundTasks,
        image: UploadFile = File(...),
        text: str = Form(...),
        title: str = Form(None),
        metadata: str = Form("{}")
):
    """
    Upload a new item to the search index

    Args:
        image: Image file
        text: Text description
        title: Optional title
        metadata: Optional JSON metadata

    Returns:
        Upload status
    """
    if not image or not text:
        raise HTTPException(status_code=400, detail="Both image and text are required")

    start_time = time.time()

    try:
        # Parse metadata JSON
        metadata_dict = json.loads(metadata)

        # Save uploaded image
        image_filename = f"{int(time.time())}_{image.filename}"
        image_path = os.path.join(UPLOAD_DIR, image_filename)

        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # Create item
        item = {
            'id': f"upload_{int(time.time())}",
            'text': text,
            'image_path': image_path,
            'metadata': {
                'title': title or 'Uploaded item',
                'source': 'upload',
                'upload_time': time.time(),
                **metadata_dict
            }
        }

        # Process item in background
        def process_uploaded_item(item):
            logger.info(f"Processing uploaded item {item['id']}")
            data_ingestor.process_single_item(item)
            logger.info(f"Finished processing item {item['id']}")

        background_tasks.add_task(process_uploaded_item, item)

        return {
            "status": "success",
            "message": "Item uploaded and being processed",
            "item_id": item['id'],
            "processing_time_ms": (time.time() - start_time) * 1000
        }

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid metadata JSON")
    except Exception as e:
        logger.error(f"Error uploading item: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/status")
async def get_status():
    """
    Get system status and statistics

    Returns:
        Status information and statistics
    """
    # Get item count
    item_count = vector_storage.get_item_count()

    # Get performance stats
    search_stats = search_engine.get_performance_stats()
    system_info = search_engine.get_system_info()

    return {
        "status": "running",
        "items_indexed": item_count,
        "performance": search_stats,
        "system": system_info
    }


@app.get("/api/image/{image_path:path}")
async def get_image(image_path: str):
    """
    Serve an image file

    Args:
        image_path: Path to image file

    Returns:
        Image file
    """
    # Extract just the filename if it's a path
    if os.path.sep in image_path:
        image_path = os.path.basename(image_path)

    # Check in various locations for the image
    search_paths = [
        os.path.join(DATA_DIR, "images", image_path),
        os.path.join(UPLOAD_DIR, image_path),
        os.path.join(DATA_DIR, "raw", "coco", "val2017", image_path)
    ]

    for path in search_paths:
        if os.path.exists(path) and os.path.isfile(path):
            return FileResponse(path)

    # Log the issue
    logger.warning(f"Image not found: {image_path}")
    logger.warning(f"Searched in: {search_paths}")

    # Not found
    raise HTTPException(status_code=404, detail=f"Image not found: {image_path}")


@app.post("/api/batch/process")
async def process_batch(
        background_tasks: BackgroundTasks,
        dataset_type: str = Form(...),
        dataset_path: str = Form(...),
        limit: int = Form(1000)
):
    """
    Process a dataset in batch mode

    Args:
        dataset_type: Type of dataset ('coco' or 'unsplash')
        dataset_path: Path to dataset file
        limit: Maximum number of items to process

    Returns:
        Processing status
    """
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=400, detail=f"Dataset file not found: {dataset_path}")

    def process_dataset():
        try:
            logger.info(f"Processing {dataset_type} dataset from {dataset_path}")

            if dataset_type.lower() == 'coco':
                # For COCO dataset, need both annotations and images directory
                if not os.path.isfile(dataset_path):
                    logger.error("COCO annotations file not found")
                    return

                # Assuming images directory is in the same parent directory
                images_dir = os.path.join(os.path.dirname(os.path.dirname(dataset_path)), "images")
                if not os.path.isdir(images_dir):
                    logger.error(f"COCO images directory not found: {images_dir}")
                    return

                items = data_ingestor.load_coco_captions(dataset_path, images_dir, limit=limit)

            elif dataset_type.lower() == 'unsplash':
                items = data_ingestor.load_unsplash_dataset(dataset_path, limit=limit)

            else:
                logger.error(f"Unknown dataset type: {dataset_type}")
                return

            if items:
                logger.info(f"Loaded {len(items)} items, processing...")
                data_ingestor.process_batch(items)
                logger.info(f"Finished processing {len(items)} items")
            else:
                logger.warning("No items loaded from dataset")

        except Exception as e:
            logger.error(f"Error processing dataset: {str(e)}")

    # Start processing in background
    background_tasks.add_task(process_dataset)

    return {
        "status": "processing",
        "message": f"Started processing {dataset_type} dataset in the background",
        "dataset": dataset_path,
        "limit": limit
    }


# Mount static files for the frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

# Run the server when executed directly
if __name__ == "__main__":
    uvicorn.run("search_app.api:app", host="0.0.0.0", port=8000, reload=True)