# Multi-Modal Search System

A scalable and cost-efficient search system for text and images, built with a focus on performance optimization for the Apple M3 platform.

## 🔍 Overview

This system provides a high-performance search engine for image-text pairs, leveraging CLIP embeddings and FAISS vector search. It's designed to be efficient on Apple Silicon (M3) hardware while maintaining low latency and high accuracy.

## 🚀 Features

- **Multi-modal search**: Find images using natural language queries
- **Optimized for M3 MacBook**: Uses Metal Performance Shaders (MPS) acceleration
- **High-performance vector search**: Fast ANN search using FAISS
- **Comprehensive metrics**: Detailed performance monitoring
- **API + User Interface**: Both programmatic and visual search capabilities
- **Batch and streaming support**: Process large datasets or individual items

## 📊 Performance

- **Search latency**: ~140ms for text queries on a 1,000-item index
- **Embedding generation**: ~139ms per query (CLIP model on M3 chip)
- **Vector search**: <1ms for 1,000 vectors
- **Memory footprint**: Optimized batching for efficient memory usage

## 🛠️ Technology Stack

- **Backend**: FastAPI + Django
- **Embedding Models**: CLIP (OpenAI)
- **Vector Storage**: FAISS
- **Web Interface**: Bootstrap UI
- **Performance Monitoring**: Built-in logging and metrics
- **Acceleration**: MPS (Metal Performance Shaders) for M3 Mac

## 📋 Requirements

- Python 3.8+
- MacOS with M3 chip (optimized for Apple Silicon)
- 8GB+ RAM
- 1GB+ disk space

## 🚀 Quick Start

### Installation

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# need to run
pip install requests  

# Initialize database
python manage.py migrate
```

### Dataset Preperation
```bash
# Create a directory for the dataset
mkdir -p data/raw/coco
cd data/raw/coco

# Download COCO validation images and annotations
curl -O http://images.cocodataset.org/zips/val2017.zip
curl -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# Extract files
unzip val2017.zip
unzip annotations_trainval2017.zip

# Move back to project root directory
cd ../../../
```


### Running the System


```bash
python setup.py

# Start the API server
uvicorn search_app.api:app --host 0.0.0.0 --port 8000

# need to run this if pandas is missing
# pip install pandas

# In a separate terminal, start the web interface
python manage.py runserver 0.0.0.0:8080
```

### API Testing

```bash
# Search for "car" with text query
curl -X GET "http://localhost:8000/api/search/text?query=car&limit=10" | python -m json.tool

# Get system status and metrics
curl -X GET "http://localhost:8000/api/status" | python -m json.tool
```

### Web Interface

Access the web interface at http://localhost:8080


### Performance report gerenate
```bash
curl http://localhost:8000/api/status | python -m json.tool > performance_data.json
```
## 📂 Project Structure

```
multimodal_search/
├── api/                 # FastAPI endpoints
├── data/                # Data storage
│   ├── images/          # Processed images
│   ├── raw/             # Raw data files
│   ├── processed/       # Intermediate files
│   ├── uploads/         # User uploads
│   └── vector_db/       # FAISS indexes
├── embedding/           # Embedding generation
├── search_app/          # Django web application
├── storage/             # Vector storage
├── templates/           # HTML templates
└── scripts/             # Utility scripts
```

## 🔍 Core Components

### 1. Data Ingestion Pipeline

Processes image-text pairs from datasets like COCO Captions. Supports both batch processing for large datasets and streaming for individual items.

### 2. Embedding Generation

Uses CLIP to create embeddings for both text and images. Optimized for Apple Silicon with MPS acceleration for maximum performance.

### 3. Vector Storage

FAISS-based vector storage for efficient similarity search. Includes metadata storage for comprehensive search results.

### 4. Search API

FastAPI endpoints that deliver fast, relevant search results with comprehensive performance metrics.

### 5. Web Interface

A simple Django-based UI for searching and visualizing results.

## 💡 Design Decisions

### Performance Optimization

- **MPS Acceleration**: Utilizes Apple's Metal Performance Shaders for faster tensor operations
- **Batch Processing**: Optimal batch sizes for M3 processor
- **Memory Efficiency**: Careful management of tensor operations to minimize memory footprint
- **Response Format**: Streamlined JSON responses with minimal overhead

### Scaling Considerations

- **Vector Search**: FAISS flat index for maximum accuracy with reasonable performance at this scale
- **API Design**: Asynchronous endpoints for better concurrency
- **Metadata Storage**: Efficient storage format for quick retrieval

## 📝 Usage Notes

- The web interface provides a visual way to search and view results
- For programmatic access, use the API endpoints
- System metrics are available through the API and displayed on the dashboard

## 🔧 Known Limitations

- Image display in the UI requires correct path configuration
- Currently optimized for the COCO Captions dataset structure

## 📈 Future Improvements

- Enhanced filtering based on metadata
- Image query support
- Quantization for larger indexes
- Distributed search capabilities