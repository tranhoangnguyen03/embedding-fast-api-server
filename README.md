# Embedding FastAPI Server

This is a FastAPI server for generating sentence embeddings using Transformer models like BAAI/bge-base-en-v1.5 and sentence-transformers/all-mpnet-base-v2.

## Features

-   Generate embeddings for text input using HTTP API
-   Support for multiple models with different endpoints
-   Async generation using `run_in_threadpool`
-   Dockerized for easy deployment

## Usage

### Local Development
```
# Install dependencies
pip install -r requirements.txt

# Start server
uvicorn app:app
```
The server will be running at http://127.0.0.1:8000.

You can make requests like:
```
curl -X POST "http://127.0.0.1:8000/v1/embeddings/baai" -H  "accept: application/json" -H  "Content-Type: application/json" -d "{\"input\": \"Hello world\"}"
```

### Docker
```
# Build image
docker build -t embedding-server .

# Start container
docker run -p 80:80 embedding-server
```
The server will be running at http://localhost

## Models
### Currently supported models:
-   `baai`: BAAI/bge-base-en-v1.5
-   `st`: sentence-transformers/all-mpnet-base-v2

###  Add a new model by:
1.  Loading model in `load_model` in app.py
2.  Adding to `MODELS` dictionary
3.  Creating a new endpoint

## License
MIT
