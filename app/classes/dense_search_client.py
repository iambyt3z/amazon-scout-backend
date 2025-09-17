import os
from datetime import datetime
from typing import List, Optional, Union

import numpy as np
import onnxruntime as ort
from dotenv import load_dotenv
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client import models as qm

# Load environment variables from .env file
load_dotenv()


def ensure_dense_collection(
    client: QdrantClient,
    collection_name: str,
    dense_dim: int,
    distance: qm.Distance = qm.Distance.COSINE,
    shards: int = 4,
    recreate_if_missing: bool = False,
):
    """
    Make sure `collection_name` has a vector named "dense".
    If collection exists but misses "dense", either raise or recreate (controlled by `recreate_if_missing`).
    """
    try:
        info = client.get_collection(collection_name)
        have_dense = "dense" in (info.vectors_count_by_name or {})
        if have_dense:
            return
        if not recreate_if_missing:
            raise RuntimeError(
                f"Collection `{collection_name}` exists but has no 'dense' vector. "
                f"Recreate the collection with a 'dense' vector or set recreate_if_missing=True."
            )
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config={"dense": qm.VectorParams(size=dense_dim, distance=distance)},
            hnsw_config=qm.HnswConfigDiff(m=16),  # reasonable default; change if you build later
            shard_number=shards,
        )
    except Exception:
        # Not found -> create if allowed, else raise
        if not recreate_if_missing:
            raise
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config={"dense": qm.VectorParams(size=dense_dim, distance=distance)},
            hnsw_config=qm.HnswConfigDiff(m=16),
            shard_number=shards,
        )


class DenseSearchClient:
    """
    Dense-only search against a Qdrant collection with a vector named "dense".
    """

    def __init__(
        self,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        collection_name: str = "hybrid-search2",  # set to your ingested collection
        dense_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_gpu: bool = True,
        deterministic: bool = False,
        random_seed: Optional[int] = None,
        ensure_schema: bool = False,  # set True if you want this client to create/repair the collection schema
        shards: int = 4,
    ):
        # Credentials
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
        self.qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
        if not self.qdrant_url or not self.qdrant_api_key:
            raise ValueError(
                "Qdrant URL and API key must be provided (env QDRANT_URL/QDRANT_API_KEY or params)."
            )

        self.collection_name = collection_name
        self.deterministic = deterministic

        # Seeds for determinism (note: embedding model itself is deterministic)
        if random_seed is not None:
            np.random.seed(random_seed)
            import random

            random.seed(random_seed)

        # ONNX providers
        avail = ort.get_available_providers()
        if deterministic:
            self.providers = ["CPUExecutionProvider"]
            self.threads = 1
        elif use_gpu and "CUDAExecutionProvider" in avail:
            self.providers = ["CUDAExecutionProvider"]
            self.threads = None
        else:
            self.providers = ["CPUExecutionProvider"]
            self.threads = None

        # Embedding model (FastEmbed)
        self.dense_model = TextEmbedding(
            dense_model, providers=self.providers, threads=self.threads
        )

        # Qdrant client
        self.client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            prefer_grpc=True,
            timeout=60,
        )

        # Optionally ensure schema
        if ensure_schema:
            dim = len(next(self.dense_model.embed(["probe"])))
            ensure_dense_collection(
                client=self.client,
                collection_name=self.collection_name,
                dense_dim=dim,
                shards=shards,
            )

    # ---------- filters ----------
    def _to_filter(self, filters: Optional[dict]) -> Optional[qm.Filter]:
        if not filters:
            return None
        must = []
        for k, v in filters.items():
            if isinstance(v, dict):
                rng_kwargs = {key: v[key] for key in ("gt", "gte", "lt", "lte") if key in v}
                if rng_kwargs:
                    must.append(qm.FieldCondition(key=k, range=qm.Range(**rng_kwargs)))
                    continue
            must.append(qm.FieldCondition(key=k, match=qm.MatchValue(value=v)))
        return qm.Filter(must=must) if must else None

    def save_results_to_file(
        self,
        results,
        query: str,
        filename: Optional[str] = None,
        include_scores: bool = True,
        include_payload: bool = True,
        max_doc_length: int = 200,
    ) -> str:
        """
        Save Qdrant search results to a text file.

        Args:
            results: Qdrant query result object
            query: The original search query
            filename: Output filename (auto-generated if None)
            include_scores: Whether to include relevance scores
            include_payload: Whether to include payload information
            max_doc_length: Maximum length of document text to include

        Returns:
            The filename of the saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dense_search_results_{timestamp}.txt"

        with open(filename, "w", encoding="utf-8") as f:
            # Header
            f.write("=" * 80 + "\n")
            f.write("DENSE SEARCH RESULTS\n")
            f.write("=" * 80 + "\n")
            f.write(f"Query: {query}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Collection: {self.collection_name}\n")
            f.write(f"Total Results: {len(results.points)}\n")
            f.write(f"Deterministic Mode: {'Yes' if self.deterministic else 'No'}\n")
            f.write("=" * 80 + "\n\n")

            # Results
            for i, point in enumerate(results.points, 1):
                f.write(f"RESULT #{i}\n")
                f.write("-" * 40 + "\n")

                if include_scores:
                    f.write(f"Score: {point.score:.6f}\n")

                if include_payload and point.payload:
                    f.write("Payload:\n")
                    for key, value in point.payload.items():
                        if (
                            key == "document"
                            and isinstance(value, str)
                            and len(value) > max_doc_length
                        ):
                            f.write(f"  {key}: {value[:max_doc_length]}...\n")
                        else:
                            f.write(f"  {key}: {value}\n")

                f.write(f"Point ID: {point.id}\n")
                f.write("\n")

            # Footer
            f.write("=" * 80 + "\n")
            f.write("END OF RESULTS\n")
            f.write("=" * 80 + "\n")

        return filename

    # ---------- search ----------
    def search(
        self,
        query: str,
        topk: int = 10,
        filters: Optional[dict] = None,
        with_payload: bool = True,
    ):
        """
        Dense vector search on the "dense" field.
        """
        q_vec = next(self.dense_model.embed([query]))
        return self.client.query_points(
            collection_name=self.collection_name,
            query=q_vec,
            using="dense",  # important when collection has multiple vectors
            limit=topk,
            query_filter=self._to_filter(filters),
            with_payload=with_payload,
        )
