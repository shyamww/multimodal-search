from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib import messages
from django.conf import settings
import os
import json
import logging
import requests

# Configure logging
logger = logging.getLogger('search_app.views')


def index(request):
    """
    Main search interface
    """
    # Get API status to show in the UI
    try:
        response = requests.get('http://localhost:8000/api/status')
        status = response.json()
        item_count = status.get('items_indexed', 0)
    except:
        status = None
        item_count = 0

    context = {
        'item_count': item_count,
        'api_status': status
    }
    return render(request, 'search_app/index.html', context)


def search_view(request):
    """
    Handle search requests
    """
    if request.method == 'GET':
        query = request.GET.get('query', '')

        if not query:
            return render(request, 'search_app/search.html', {'results': None})

        try:
            # Call the FastAPI endpoint
            response = requests.get(f'http://localhost:8000/api/search/text?query={query}')
            data = response.json()

            # Process results to add full image URLs
            for result in data.get('results', []):
                if 'metadata' in result and 'image_path' in result['metadata']:
                    # Extract filename from path
                    filename = os.path.basename(result['metadata']['image_path'])
                    result['image_url'] = f'/api/image/{filename}'

            context = {
                'query': query,
                'results': data.get('results', []),
                'count': data.get('result_count', 0),
                'query_time': data.get('query_time_ms', 0),
                'performance': data.get('performance', {})
            }
            return render(request, 'search_app/search.html', context)

        except Exception as e:
            logger.error(f"Error searching: {str(e)}")
            messages.error(request, f"Search error: {str(e)}")
            return render(request, 'search_app/search.html', {'error': str(e)})

    return render(request, 'search_app/search.html', {})


@csrf_exempt
def upload_view(request):
    """
    Handle file uploads
    """
    if request.method == 'POST':
        try:
            # Get form data
            image = request.FILES.get('image')
            text = request.POST.get('text')
            title = request.POST.get('title', '')
            tags = request.POST.get('tags', '')

            if not image or not text:
                messages.error(request, "Both image and text are required")
                return render(request, 'search_app/upload.html', {})

            # Prepare metadata
            metadata = {
                'tags': [tag.strip() for tag in tags.split(',') if tag.strip()]
            }

            # Create multipart form data
            files = {'image': (image.name, image, image.content_type)}
            data = {
                'text': text,
                'title': title,
                'metadata': json.dumps(metadata)
            }

            # Send to FastAPI endpoint
            response = requests.post('http://localhost:8000/api/upload', files=files, data=data)

            if response.status_code == 200:
                result = response.json()
                messages.success(request, "Upload successful! Your item is being processed.")
                return redirect('search')
            else:
                messages.error(request, f"Upload failed: {response.text}")

        except Exception as e:
            logger.error(f"Error uploading: {str(e)}")
            messages.error(request, f"Upload error: {str(e)}")

    return render(request, 'search_app/upload.html', {})


def dashboard_view(request):
    """
    Dashboard with system metrics
    """
    try:
        # Get API status
        response = requests.get('http://localhost:8000/api/status')
        status = response.json()

        context = {
            'status': status,
            'item_count': status.get('items_indexed', 0),
            'performance': status.get('performance', {}),
            'system': status.get('system', {})
        }
        return render(request, 'search_app/dashboard.html', context)

    except Exception as e:
        logger.error(f"Error getting dashboard data: {str(e)}")
        messages.error(request, f"Dashboard error: {str(e)}")
        return render(request, 'search_app/dashboard.html', {'error': str(e)})


def batch_upload_view(request):
    """
    Batch upload interface
    """
    if request.method == 'POST':
        try:
            dataset_type = request.POST.get('dataset_type')
            dataset_path = request.POST.get('dataset_path')
            limit = int(request.POST.get('limit', 1000))

            if not dataset_type or not dataset_path:
                messages.error(request, "Dataset type and path are required")
                return render(request, 'search_app/batch_upload.html', {})

            # Create form data
            data = {
                'dataset_type': dataset_type,
                'dataset_path': dataset_path,
                'limit': limit
            }

            # Send to FastAPI endpoint
            response = requests.post('http://localhost:8000/api/batch/process', data=data)

            if response.status_code == 200:
                result = response.json()
                messages.success(request,
                                 f"Batch processing started! Processing {limit} items from {dataset_type} dataset.")
                return redirect('dashboard')
            else:
                messages.error(request, f"Batch processing failed: {response.text}")

        except Exception as e:
            logger.error(f"Error starting batch process: {str(e)}")
            messages.error(request, f"Batch processing error: {str(e)}")

    return render(request, 'search_app/batch_upload.html', {})