import math
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "models/qwen3-embedding-4b" in data["model"]

def test_list_models():
    res = client.get("/v1beta/models")
    assert res.status_code == 200
    data = res.json()
    assert len(data["models"]) >= 1
    assert "embedContent" in data["models"][0]["supportedGenerationMethods"]

def test_google_ai_embed_content():
    payload = {
        "content": {
            "parts": [
                {"text": "Artificial Intelligence and Machine Learning"}
            ]
        }
    }
    res = client.post("/v1beta/models/text-embedding-004:embedContent", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "embedding" in data
    assert "values" in data["embedding"]
    values = data["embedding"]["values"]
    assert len(values) == 2560
    # Check L2 normalization
    norm = math.sqrt(sum(x * x for x in values))
    assert abs(norm - 1.0) < 1e-2

def test_google_ai_embed_content_with_mrl_dim():
    payload = {
        "content": {
            "parts": [
                {"text": "Test embedding with custom output dimensionality"}
            ]
        },
        "outputDimensionality": 768
    }
    res = client.post("/v1beta/models/text-embedding-004:embedContent", json=payload)
    assert res.status_code == 200
    data = res.json()
    values = data["embedding"]["values"]
    assert len(values) == 768
    norm = math.sqrt(sum(x * x for x in values))
    assert abs(norm - 1.0) < 1e-2

def test_google_ai_batch_embed_contents():
    payload = {
        "requests": [
            {
                "content": {"parts": [{"text": "First passage to embed"}]},
                "outputDimensionality": 512
            },
            {
                "content": {"parts": [{"text": "Second passage to embed"}]},
                "outputDimensionality": 512
            }
        ]
    }
    res = client.post("/v1beta/models/text-embedding-004:batchEmbedContents", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "embeddings" in data
    assert len(data["embeddings"]) == 2
    for emb in data["embeddings"]:
        assert len(emb["values"]) == 512

def test_simple_embed():
    payload = {"text": "Simple test string", "output_dim": 256}
    res = client.post("/embed", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["dimension"] == 256
    assert len(data["embedding"]) == 256

def test_openai_compatibility():
    payload = {
        "input": ["First sentence", "Second sentence"],
        "model": "qwen3-embedding-4b",
        "dimensions": 1024
    }
    res = client.post("/v1/embeddings", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 2
    assert len(data["data"][0]["embedding"]) == 1024
    assert len(data["data"][1]["embedding"]) == 1024

if __name__ == "__main__":
    pytest.main(["-v", __file__])
