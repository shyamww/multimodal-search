from django.db import models
import uuid
import json


class ImageTextPair(models.Model):
    """Model to store image-text pairs and their metadata"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    image_path = models.CharField(max_length=255)
    text = models.TextField()
    title = models.CharField(max_length=255, blank=True, null=True)

    # Store metadata as JSON
    metadata_json = models.TextField(blank=True, null=True)

    # Store embedding vector path
    embedding_path = models.CharField(max_length=255, blank=True, null=True)

    # Track processing status
    processed = models.BooleanField(default=False)
    date_created = models.DateTimeField(auto_now_add=True)

    @property
    def metadata(self):
        """Convert JSON metadata to dictionary"""
        if self.metadata_json:
            return json.loads(self.metadata_json)
        return {}

    @metadata.setter
    def metadata(self, value):
        """Convert dictionary to JSON metadata"""
        self.metadata_json = json.dumps(value)

    def __str__(self):
        return f"{self.title or 'Untitled'}: {self.text[:50]}..."


class SearchQuery(models.Model):
    """Model to track search queries for analytics"""
    query_text = models.TextField()
    query_image = models.CharField(max_length=255, blank=True, null=True)
    results_count = models.IntegerField(default=0)
    latency_ms = models.FloatField(default=0)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.query_text[:50]} ({self.timestamp})"