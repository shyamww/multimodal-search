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

# Configuration - Define these BEFORE logging setup
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = os.path.join(BASE_DIR, "data")
VECTOR_DB_PATH = os.path.join(DATA_DIR, "vector_db", "index.faiss")
METADATA_PATH = os.path.join(DATA_DIR, "vector_db", "metadata.json")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
MEDIA_DIR = os.path.join(BASE_DIR, "media")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Ensure directories exist
os.makedirs(os.path.join(DATA_DIR, "vector_db"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "images"), exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MEDIA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Configure logging - Now LOGS_DIR is defined
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "api.log")),
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

# Initialize components
embedding_generator = None
vector_storage = None
search_engine = None
data_ingestor = None

# Run the server when executed directly
if __name__ == "__main__":
    uvicorn.run("search_app.api:app", host="0.0.0.0", port=8000, reload=True)