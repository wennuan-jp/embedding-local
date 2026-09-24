# Local Qwen3-Embedding Service (Google AI & OpenAI Compatible)

High-performance local text embedding microservice powered by Apple Silicon **MLX** and **Qwen3-Embedding-4B-4bit-DWQ**.

This service runs locally on macOS (Apple Silicon) with hardware acceleration (~70ms latency per embedding) and exposes REST APIs compatible with:
1. **Google AI (Gemini) Generative Language API** (`embedContent` & `batchEmbedContents`)
2. **OpenAI Embeddings API** (`/v1/embeddings` — for LangChain, LlamaIndex, LiteLLM, ChromaDB)
3. **Simple Direct REST API** (`/embed`)

---

## Table of Contents

- [System Requirements & Architecture](#system-requirements--architecture)
- [How to Start the Service](#how-to-start-the-service)
  - [Foreground Mode](#1-foreground-mode)
  - [Background / Daemon Mode](#2-background--daemon-mode)
  - [Stopping the Service](#3-stopping-the-service)
  - [Environment Variables](#4-environment-variables)
- [Health Check & Model Discovery](#health-check--model-discovery)
- [Complete Endpoint Reference](#complete-endpoint-reference)
  - [1. Google AI: Single Embedding (`embedContent`)](#1-google-ai-single-embedding-embedcontent)
  - [2. Google AI: Batch Embedding (`batchEmbedContents`)](#2-google-ai-batch-embedding-batchembedcontents)
  - [3. OpenAI Compatible: (`/v1/embeddings`)](#3-openai-compatible-v1embeddings)
  - [4. Simple Direct: (`/embed`)](#4-simple-direct-embed)
- [Client Integration Guides](#client-integration-guides)
  - [Python (requests / httpx)](#python-requests--httpx)
  - [Python (OpenAI SDK)](#python-openai-sdk)
  - [Python (LangChain)](#python-langchain)
  - [Node.js / TypeScript (fetch)](#nodejs--typescript-fetch)
- [Matryoshka Representation Learning (MRL)](#matryoshka-representation-learning-mrl)
- [Troubleshooting & Verification](#troubleshooting--verification)

---

## System Requirements & Architecture

- **OS**: macOS (Apple Silicon: M1 / M2 / M3 / M4)
- **Model Path**: `/Users/wennuan/.lmstudio/models/mlx-community/Qwen3-Embedding-4B-4bit-DWQ/`
- **Native Dimensions**: 2560 (MRL-truncatable to 1536, 1024, 768, 512, 256, etc.)
- **Default Base URL**: `http://localhost:8000`
- **Normalization**: All outputs are L2-normalized (`||vector|| = 1.0`), ready for cosine similarity.

---

## How to Start the Service

### 1. Foreground Mode

```bash
cd /Users/wennuan/dev/infra/embedding-local

# Using the startup script (default port 8000, host 0.0.0.0)
./start.sh

# Or directly with uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 2. Background / Daemon Mode

To keep the service running in the background for other local services to connect:

```bash
cd /Users/wennuan/dev/infra/embedding-local
nohup ./start.sh > service.log 2>&1 &
```

Check the log to ensure the model loaded:
```bash
tail -f /Users/wennuan/dev/infra/embedding-local/service.log
```

### 3. Stopping the Service

```bash
pkill -f "uvicorn app.main:app"
```

### 4. Environment Variables

You can customize port, host, or model path by passing environment variables:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `HOST` | `0.0.0.0` | IP to bind (use `0.0.0.0` for LAN access, `127.0.0.1` for local only) |
| `PORT` | `8000` | Port to listen on |
| `MODEL_PATH` | `/Users/wennuan/.lmstudio/models/...` | Path to MLX model directory |
| `MAX_SEQUENCE_LENGTH` | `8192` | Maximum token limit per text sequence |

Example:
```bash
PORT=9000 HOST=127.0.0.1 ./start.sh
```

---

## Health Check & Model Discovery

### Check Service Health
```bash
curl http://localhost:8000/health
```

**Response (200 OK)**:
```json
{
  "status": "ok",
  "model": "models/qwen3-embedding-4b",
  "model_path": "/Users/wennuan/.lmstudio/models/mlx-community/Qwen3-Embedding-4B-4bit-DWQ/"
}
```

### List Models
```bash
curl http://localhost:8000/v1beta/models
```

**Response (200 OK)**:
```json
{
  "models": [
    {
      "name": "models/qwen3-embedding-4b",
      "version": "1.0",
      "displayName": "Qwen3 Embedding 4B (MLX 4-bit)",
      "description": "High performance local text embedding model",
      "inputTokenLimit": 8192,
      "outputDimensionality": 2560,
      "supportedGenerationMethods": ["embedContent", "batchEmbedContents"]
    }
  ]
}
```

---

## Complete Endpoint Reference

| Method | Endpoint Path | Protocol / Style | Purpose |
| :--- | :--- | :--- | :--- |
| `POST` | `/v1beta/models/{model}:embedContent` | Google AI (Gemini) | Embed single text with Google AI schema |
| `POST` | `/v1beta/models/{model}:batchEmbedContents` | Google AI (Gemini) | Batch embed multiple items |
| `POST` | `/v1/embeddings` | OpenAI Compatible | Drop-in for OpenAI SDK & LangChain |
| `POST` | `/embed` | Simple REST | Direct JSON payload (`{"text": "..."}`) |
| `GET` | `/health` | Utility | Service health check |
| `GET` | `/v1beta/models` | Google AI | Model metadata |

> **Note on Model Names in URL**: Any string passed in `{model}` (e.g. `text-embedding-004`, `models/text-embedding-004`, `qwen3-embedding-4b`) is accepted and served by the loaded Qwen3 model.

---

### 1. Google AI: Single Embedding (`embedContent`)

**Endpoint**: `POST /v1beta/models/{model}:embedContent`  
*(Also supported on `/v1/models/{model}:embedContent`, `/v1beta/{model}:embedContent`, `/embedContent`)*

#### Request Schema

```json
{
  "content": {
    "parts": [
      { "text": "Your plain text here" }
    ]
  },
  "taskType": "RETRIEVAL_QUERY",
  "title": "Optional title (recommended for RETRIEVAL_DOCUMENT)",
  "outputDimensionality": 768
}
```

- **`content`** *(object or string, required)*: Contains `parts` array with `text`. (Shorthand: `"text": "..."` or `"content": "..."` also accepted).
- **`taskType`** *(string, optional)*: Optimizes task instructions. Supported values:
  - `RETRIEVAL_QUERY`: Prepends query retrieval instruction.
  - `RETRIEVAL_DOCUMENT`: Formats text as document (uses `title` if provided).
  - `SEMANTIC_SIMILARITY`: Semantic similarity clustering/matching.
  - `CLASSIFICATION`: Classification task.
  - `CLUSTERING`: Clustering task.
  - `QUESTION_ANSWERING`: Q&A passage retrieval.
  - `FACT_VERIFICATION`: Fact checking.
- **`outputDimensionality`** *(integer, optional)*: Truncate vector to lower dimension (e.g. `768`, `512`, `256`) via Matryoshka Representation Learning. Native is `2560`.

#### cURL Example
```bash
curl -X POST http://localhost:8000/v1beta/models/text-embedding-004:embedContent \
  -H "Content-Type: application/json" \
  -d '{
    "content": {
      "parts": [
        {"text": "What are the latest advances in artificial intelligence?"}
      ]
    },
    "taskType": "RETRIEVAL_QUERY",
    "outputDimensionality": 768
  }'
```

#### Response (200 OK)
```json
{
  "embedding": {
    "values": [
      -0.003421,
      0.015234,
      -0.021098,
      "..."
    ]
  }
}
```

---

### 2. Google AI: Batch Embedding (`batchEmbedContents`)

**Endpoint**: `POST /v1beta/models/{model}:batchEmbedContents`  
*(Also supported on `/v1/models/{model}:batchEmbedContents`, `/batchEmbedContents`)*

#### Request Schema

```json
{
  "requests": [
    {
      "model": "models/text-embedding-004",
      "content": {
        "parts": [{ "text": "First passage to embed" }]
      },
      "taskType": "RETRIEVAL_DOCUMENT",
      "outputDimensionality": 512
    },
    {
      "model": "models/text-embedding-004",
      "content": {
        "parts": [{ "text": "Second passage to embed" }]
      },
      "taskType": "RETRIEVAL_DOCUMENT",
      "outputDimensionality": 512
    }
  ]
}
```

#### cURL Example
```bash
curl -X POST http://localhost:8000/v1beta/models/text-embedding-004:batchEmbedContents \
  -H "Content-Type: application/json" \
  -d '{
    "requests": [
      {
        "content": {"parts": [{"text": "Apple Silicon Unified Memory Architecture"}]},
        "outputDimensionality": 512
      },
      {
        "content": {"parts": [{"text": "Fast vector embeddings using Apple MLX"}]},
        "outputDimensionality": 512
      }
    ]
  }'
```

#### Response (200 OK)
```json
{
  "embeddings": [
    {
      "values": [-0.0123, 0.0456, "..."]
    },
    {
      "values": [0.0089, -0.0234, "..."]
    }
  ]
}
```

---

### 3. OpenAI Compatible: (`/v1/embeddings`)

**Endpoint**: `POST /v1/embeddings`

Designed as a drop-in endpoint for any tool or SDK expecting an OpenAI-compatible embedding service (LangChain, LlamaIndex, LiteLLM, Ollama clients, ChromaDB).

#### Request Schema

```json
{
  "input": "Single string" // OR array of strings: ["Text 1", "Text 2"],
  "model": "qwen3-embedding-4b", // optional
  "dimensions": 768 // optional MRL dimension truncation
}
```

#### cURL Example
```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": [
      "Artificial intelligence and local embeddings",
      "FastAPI server with MLX acceleration"
    ],
    "dimensions": 768
  }'
```

#### Response (200 OK)
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [-0.0023, 0.0451, "..."]
    },
    {
      "object": "embedding",
      "index": 1,
      "embedding": [0.0128, -0.0384, "..."]
    }
  ],
  "model": "qwen3-embedding-4b",
  "usage": {
    "prompt_tokens": 16,
    "total_tokens": 16
  }
}
```

---

### 4. Simple Direct: (`/embed`)

**Endpoint**: `POST /embed`

Minimalist endpoint for quick scripts and internal microservices.

#### Request Schema
```json
{
  "text": "Plain text string to embed",
  "output_dim": 512, // optional (default: 2560)
  "task_type": "RETRIEVAL_QUERY" // optional
}
```

#### cURL Example
```bash
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world from MLX local embedding",
    "output_dim": 512
  }'
```

#### Response (200 OK)
```json
{
  "embedding": [-0.0105, 0.0385, "..."],
  "dimension": 512
}
```

---

## Client Integration Guides

### Python (`requests` / `httpx`)

```python
import requests

# 1. Connect to local service (Google AI style)
res = requests.post(
    "http://localhost:8000/v1beta/models/text-embedding-004:embedContent",
    json={
        "content": {"parts": [{"text": "Embed this sentence"}]},
        "taskType": "RETRIEVAL_QUERY",
        "outputDimensionality": 768
    }
)
res.raise_for_status()
vector = res.json()["embedding"]["values"]
print("Vector dimension:", len(vector))
```

### Python (`openai` SDK)

Any existing code using the OpenAI library can switch to this service by changing `base_url`:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # Local server does not require authentication
)

response = client.embeddings.create(
    input=["Machine learning embeddings", "Vector search in database"],
    model="qwen3-embedding-4b",
    dimensions=768
)

for item in response.data:
    print(f"Index {item.index}: dim={len(item.embedding)}")
```

### Python (`langchain`)

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
    model="qwen3-embedding-4b",
    dimensions=768
)

doc_vector = embeddings.embed_query("Query text for semantic search")
print("Embedded query length:", len(doc_vector))
```

### Node.js / TypeScript (`fetch`)

```typescript
async function getEmbedding(text: string): Promise<number[]> {
  const response = await fetch("http://localhost:8000/v1beta/models/text-embedding-004:embedContent", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      content: { parts: [{ text }] },
      outputDimensionality: 768
    })
  });

  if (!response.ok) {
    throw new Error(`Embedding request failed: ${response.statusText}`);
  }

  const data = await response.json();
  return data.embedding.values;
}

// Usage
getEmbedding("Hello world from Node.js").then((vec) => {
  console.log("Vector length:", vec.length);
});
```

---

## Matryoshka Representation Learning (MRL)

`Qwen3-Embedding-4B` was trained with Matryoshka Representation Learning (MRL). This allows users to truncate the output dimension to save database storage and index memory while preserving high retrieval accuracy.

| Dimension | Typical Use Case | Storage Savings vs Native |
| :--- | :--- | :--- |
| **2560** | Native full precision, highest benchmark scores | 0% (baseline) |
| **1536** | Drop-in replacement for OpenAI `text-embedding-3-small` | 40% reduction |
| **1024** | Balanced general-purpose vector search | 60% reduction |
| **768** | Standard Google AI (`text-embedding-004`) / BERT size | 70% reduction |
| **512** | Ultra high-throughput vector search | 80% reduction |

Whenever an `outputDimensionality` or `dimensions` parameter is provided, the service slices the first $N$ dimensions and re-normalizes using L2 norm (`||v|| = 1.0`).

---

## Troubleshooting & Verification

### Run Automated Unit & Integration Tests

```bash
cd /Users/wennuan/dev/infra/embedding-local
pytest -v test_service.py
```

### Common Issues

1. **Connection Refused (`curl: (7) Failed to connect to localhost port 8000`)**:
   - Check if the service is running: `pgrep -fl "uvicorn app.main:app"`
   - If not running, start with `./start.sh` or check `service.log`.

2. **Port Conflict (`error: address already in use`)**:
   - Choose a different port: `PORT=8001 ./start.sh`

3. **Wrong Tokenizer or Model Path**:
   - Verify the model path exists: `ls -la /Users/wennuan/.lmstudio/models/mlx-community/Qwen3-Embedding-4B-4bit-DWQ/`
   - Override with `MODEL_PATH=/path/to/model ./start.sh` if moved.
