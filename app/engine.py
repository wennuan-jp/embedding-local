import asyncio
from typing import List, Optional
import mlx.core as mx
import mlx_lm
from app.config import settings

TASK_INSTRUCTIONS = {
    "RETRIEVAL_QUERY": "Given a web search query, retrieve relevant passages that answer the query",
    "SEMANTIC_SIMILARITY": "Retrieve semantically similar text.",
    "CLASSIFICATION": "Classify the input text.",
    "CLUSTERING": "Identify the topic or cluster of the input text.",
    "QUESTION_ANSWERING": "Given a question, retrieve relevant documents that answer the question",
    "FACT_VERIFICATION": "Given a claim, retrieve documents that support or refute the claim",
}

class EmbeddingEngine:
    _instance: Optional["EmbeddingEngine"] = None

    def __init__(self, model_path: str = settings.MODEL_PATH):
        print(f"Loading Qwen3-Embedding model from {model_path}...")
        self.model, self.tokenizer = mlx_lm.load(model_path)
        self.lock = asyncio.Lock()
        print("Model loaded successfully.")

    @classmethod
    def get_instance(cls) -> "EmbeddingEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def format_input(
        self,
        text: str,
        task_type: Optional[str] = None,
        title: Optional[str] = None,
        custom_instruction: Optional[str] = None,
    ) -> str:
        """Apply instruction template according to task_type or custom instruction."""
        # If document has title (Google AI style)
        if title:
            text = f"{title}\n{text}"

        instruction = custom_instruction
        if not instruction and task_type:
            instruction = TASK_INSTRUCTIONS.get(task_type.upper())

        if instruction:
            return f"Instruct: {instruction}\nQuery: {text}"
        return text

    def _embed_single(self, text: str, output_dim: Optional[int] = None) -> List[float]:
        tokens = self.tokenizer.encode(text)
        max_len = settings.MAX_SEQUENCE_LENGTH
        if len(tokens) > max_len:
            tokens = tokens[:max_len]

        input_ids = mx.array([tokens])
        # Forward pass through transformer backbone
        hidden = self.model.model(input_ids)

        # Last token pooling for decoder-only Qwen3 embedding
        last_hidden = hidden[0, -1, :]

        # Matryoshka Representation Learning (MRL) dimension truncation
        if output_dim is not None and output_dim > 0 and output_dim < last_hidden.shape[-1]:
            last_hidden = last_hidden[:output_dim]

        # L2 Normalization
        norm = mx.linalg.norm(last_hidden)
        emb = last_hidden / mx.maximum(norm, 1e-12)
        mx.eval(emb)

        return emb.tolist()

    async def embed_single(
        self,
        text: str,
        task_type: Optional[str] = None,
        title: Optional[str] = None,
        custom_instruction: Optional[str] = None,
        output_dim: Optional[int] = None,
    ) -> List[float]:
        formatted_text = self.format_input(
            text=text,
            task_type=task_type,
            title=title,
            custom_instruction=custom_instruction,
        )
        async with self.lock:
            return self._embed_single(formatted_text, output_dim=output_dim)

    async def embed_batch(
        self,
        items: List[dict],
        default_output_dim: Optional[int] = None,
    ) -> List[List[float]]:
        results = []
        async with self.lock:
            for item in items:
                formatted_text = self.format_input(
                    text=item.get("text", ""),
                    task_type=item.get("task_type"),
                    title=item.get("title"),
                    custom_instruction=item.get("custom_instruction"),
                )
                dim = item.get("output_dim") or default_output_dim
                results.append(self._embed_single(formatted_text, output_dim=dim))
        return results
