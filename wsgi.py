"""WSGI entry point for Render / Gunicorn."""
from app import app

if __name__ == "__main__":
    app.run()
