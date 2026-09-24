from typing import Any, List, Optional, Union
from pydantic import BaseModel, Field, ConfigDict

# -------------------------------------------------------------
# Google AI API Schemas (Generative Language API)
# -------------------------------------------------------------

class Part(BaseModel):
    text: Optional[str] = None

class Content(BaseModel):
    role: Optional[str] = None
    parts: Optional[List[Part]] = None

class EmbedContentRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    model: Optional[str] = None
    content: Optional[Union[Content, str, dict]] = None
    # Flexible fallbacks for developer convenience
    contents: Optional[Union[Content, str, dict, List[Any]]] = None
    text: Optional[str] = None
    taskType: Optional[str] = Field(default=None, alias="taskType")
    task_type: Optional[str] = None
    title: Optional[str] = None
    outputDimensionality: Optional[int] = Field(default=None, alias="outputDimensionality")
    output_dimensionality: Optional[int] = None
    instruction: Optional[str] = None

    def extract_text(self) -> str:
        """Extract plain text from content/contents/parts/text."""
        if self.text:
            return self.text

        # Check content
        target = self.content if self.content is not None else self.contents
        if isinstance(target, str):
            return target
        if isinstance(target, dict):
            # Might be {"parts": [{"text": "..."}]} or {"text": "..."}
            if "text" in target and isinstance(target["text"], str):
                return target["text"]
            parts = target.get("parts", [])
            extracted = [p.get("text", "") for p in parts if isinstance(p, dict) and "text" in p]
            if extracted:
                return "\n".join(extracted)
        if isinstance(target, Content) and target.parts:
            extracted = [p.text for p in target.parts if p.text]
            if extracted:
                return "\n".join(extracted)
        if isinstance(target, list):
            texts = []
            for item in target:
                if isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, dict) and "text" in item:
                    texts.append(item["text"])
            if texts:
                return "\n".join(texts)

        return ""

    def get_output_dim(self) -> Optional[int]:
        return self.outputDimensionality or self.output_dimensionality

    def get_task_type(self) -> Optional[str]:
        return self.taskType or self.task_type


class ContentEmbedding(BaseModel):
    values: List[float]


class EmbedContentResponse(BaseModel):
    embedding: ContentEmbedding


class BatchEmbedContentsRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    requests: List[EmbedContentRequest]


class BatchEmbedContentsResponse(BaseModel):
    embeddings: List[ContentEmbedding]


# -------------------------------------------------------------
# Convenience & OpenAI-Compatible Schemas
# -------------------------------------------------------------

class SimpleEmbedRequest(BaseModel):
    text: str
    output_dim: Optional[int] = None
    task_type: Optional[str] = None
    instruction: Optional[str] = None

class SimpleEmbedResponse(BaseModel):
    embedding: List[float]
    dimension: int

class OpenAIEmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: Optional[str] = "qwen3-embedding-4b"
    dimensions: Optional[int] = None

class OpenAIEmbeddingItem(BaseModel):
    object: str = "embedding"
    index: int
    embedding: List[float]

class OpenAIUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int

class OpenAIEmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[OpenAIEmbeddingItem]
    model: str
    usage: OpenAIUsage
