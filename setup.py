import os
import sys


def setup_environment():
    """
    Set up the necessary directories and environment for the multimodal search system
    """
    print("Setting up multimodal search environment...")

    # Define base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Create required directories
    directories = [
        os.path.join(base_dir, "logs"),
        os.path.join(base_dir, "data"),
        os.path.join(base_dir, "data", "images"),
        os.path.join(base_dir, "data", "raw"),
        os.path.join(base_dir, "data", "processed"),
        os.path.join(base_dir, "data", "uploads"),
        os.path.join(base_dir, "data", "vector_db"),
        os.path.join(base_dir, "media"),
        os.path.join(base_dir, "static", "images"),
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

    print("\nEnvironment setup complete!")
    print("\nTo run the system:")
    print("1. Start the API server: uvicorn search_app.api:app --host 0.0.0.0 --port 8000")
    print("2. Start the web interface: python manage.py runserver 0.0.0.0:8080")
    print("\nNote: You'll need to process a dataset before searching. See README.md for details.")


if __name__ == "__main__":
    setup_environment()