import torch
from transformers import AutoTokenizer, AutoModel
from starlette.concurrency import run_in_threadpool
from typing import List, Union
from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings
from loguru import logger

router = APIRouter()

# Initiate the app
def create_app():
    app = FastAPI(title="Embeddings Fast API Server", version="0.1")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app

# Define request and response models
class CreateEmbeddingRequest(BaseModel):
    input: Union[str, List[str]] = Field(..., description="The input to embed.")

class Embedding(BaseModel):
    embedding: List[List[float]] = Field(..., max_items=2048)

class CreateEmbeddingResponse(BaseModel):
    data: List[Embedding]

# Separate models loading logic for dynamic handling
def load_model(model_name):
    if model_name == "BAAI/bge-base-en-v1.5":
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        model = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            cache_folder=f'./models'
        )
    elif model_name ==  "sentence-transformers/all-mpnet-base-v2":
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            cache_folder=f'./models'
        )
    return model

def _create_embedding(input: Union[str, List[str]], model):
    if not isinstance(input, list):
        input = [input]
    data = [Embedding(embedding= model.embed_documents(i)) for i in input]
    return CreateEmbeddingResponse(data=data)

# Initialize all models
MODELS = {
    "baai": load_model("BAAI/bge-base-en-v1.5"),
    "st": load_model("sentence-transformers/all-mpnet-base-v2")
}

# Endpoints for different models
@router.post("/v1/embeddings/baai", response_model=CreateEmbeddingResponse)
async def create_embedding_baai(request: CreateEmbeddingRequest):
    return await run_in_threadpool(
        _create_embedding, 
        **request.dict(), 
        model=MODELS["baai"]
    )

@router.post("/v1/embeddings/st", response_model=CreateEmbeddingResponse)
async def create_embedding_st(request: CreateEmbeddingRequest):
    return await run_in_threadpool(
        _create_embedding, 
        **request.dict(), 
        model=MODELS["st"]
    )