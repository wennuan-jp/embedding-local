from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.engine import EmbeddingEngine
from app.schemas import (
    EmbedContentRequest,
    EmbedContentResponse,
    ContentEmbedding,
    BatchEmbedContentsRequest,
    BatchEmbedContentsResponse,
    SimpleEmbedRequest,
    SimpleEmbedResponse,
    OpenAIEmbeddingRequest,
    OpenAIEmbeddingResponse,
    OpenAIEmbeddingItem,
    OpenAIUsage,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Preload and warm up model on startup
    engine = EmbeddingEngine.get_instance()
    # Warmup inference
    await engine.embed_single("warmup")
    print("Embedding service is warmed up and ready to serve.")
    yield

app = FastAPI(
    title="Qwen3 Local Embedding Service",
    description="Google AI (Gemini) and OpenAI compatible embedding web service powered by MLX",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model": settings.DEFAULT_MODEL_NAME,
        "model_path": settings.MODEL_PATH,
    }


@app.get("/v1beta/models")
@app.get("/v1/models")
async def list_models():
    return {
        "models": [
            {
                "name": settings.DEFAULT_MODEL_NAME,
                "version": "1.0",
                "displayName": "Qwen3 Embedding 4B (MLX 4-bit)",
                "description": "High performance local text embedding model",
                "inputTokenLimit": settings.MAX_SEQUENCE_LENGTH,
                "outputDimensionality": 2560,
                "supportedGenerationMethods": ["embedContent", "batchEmbedContents"],
            }
        ]
    }


# -------------------------------------------------------------
# Google AI API: embedContent
# -------------------------------------------------------------

async def _process_embed_content(req: EmbedContentRequest) -> EmbedContentResponse:
    text = req.extract_text()
    if not text:
        raise HTTPException(status_code=400, detail="No text found in content or text field.")

    engine = EmbeddingEngine.get_instance()
    values = await engine.embed_single(
        text=text,
        task_type=req.get_task_type(),
        title=req.title,
        custom_instruction=req.instruction,
        output_dim=req.get_output_dim(),
    )
    return EmbedContentResponse(embedding=ContentEmbedding(values=values))


@app.post("/v1beta/models/{model_id:path}:embedContent", response_model=EmbedContentResponse)
@app.post("/v1beta/{model_id:path}:embedContent", response_model=EmbedContentResponse)
@app.post("/v1/models/{model_id:path}:embedContent", response_model=EmbedContentResponse)
@app.post("/models/{model_id:path}:embedContent", response_model=EmbedContentResponse)
@app.post("/embedContent", response_model=EmbedContentResponse)
async def embed_content_endpoint(
    request: EmbedContentRequest,
    model_id: Optional[str] = None,
):
    return await _process_embed_content(request)


# -------------------------------------------------------------
# Google AI API: batchEmbedContents
# -------------------------------------------------------------

async def _process_batch_embed(req: BatchEmbedContentsRequest) -> BatchEmbedContentsResponse:
    if not req.requests:
        raise HTTPException(status_code=400, detail="requests list cannot be empty.")

    engine = EmbeddingEngine.get_instance()
    items = []
    for r in req.requests:
        text = r.extract_text()
        if not text:
            raise HTTPException(status_code=400, detail="Empty text in one of the batch requests.")
        items.append({
            "text": text,
            "task_type": r.get_task_type(),
            "title": r.title,
            "custom_instruction": r.instruction,
            "output_dim": r.get_output_dim(),
        })

    vectors = await engine.embed_batch(items)
    embeddings = [ContentEmbedding(values=v) for v in vectors]
    return BatchEmbedContentsResponse(embeddings=embeddings)


@app.post("/v1beta/models/{model_id:path}:batchEmbedContents", response_model=BatchEmbedContentsResponse)
@app.post("/v1beta/{model_id:path}:batchEmbedContents", response_model=BatchEmbedContentsResponse)
@app.post("/v1/models/{model_id:path}:batchEmbedContents", response_model=BatchEmbedContentsResponse)
@app.post("/batchEmbedContents", response_model=BatchEmbedContentsResponse)
async def batch_embed_endpoint(
    request: BatchEmbedContentsRequest,
    model_id: Optional[str] = None,
):
    return await _process_batch_embed(request)


# -------------------------------------------------------------
# Convenience Endpoint: /embed
# -------------------------------------------------------------

@app.post("/embed", response_model=SimpleEmbedResponse)
async def simple_embed(req: SimpleEmbedRequest):
    if not req.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    engine = EmbeddingEngine.get_instance()
    vec = await engine.embed_single(
        text=req.text,
        task_type=req.task_type,
        custom_instruction=req.instruction,
        output_dim=req.output_dim,
    )
    return SimpleEmbedResponse(embedding=vec, dimension=len(vec))


# -------------------------------------------------------------
# OpenAI Compatible API: /v1/embeddings
# -------------------------------------------------------------

@app.post("/v1/embeddings", response_model=OpenAIEmbeddingResponse)
async def openai_embeddings(req: OpenAIEmbeddingRequest):
    inputs = [req.input] if isinstance(req.input, str) else req.input
    if not inputs:
        raise HTTPException(status_code=400, detail="input cannot be empty.")

    engine = EmbeddingEngine.get_instance()
    items = [{"text": t, "output_dim": req.dimensions} for t in inputs]
    vectors = await engine.embed_batch(items, default_output_dim=req.dimensions)

    total_tokens = sum(len(engine.tokenizer.encode(t)) for t in inputs)
    data = [
        OpenAIEmbeddingItem(index=idx, embedding=vec)
        for idx, vec in enumerate(vectors)
    ]

    return OpenAIEmbeddingResponse(
        data=data,
        model=req.model or "qwen3-embedding-4b",
        usage=OpenAIUsage(prompt_tokens=total_tokens, total_tokens=total_tokens),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=False)
