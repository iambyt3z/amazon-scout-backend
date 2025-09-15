import os
from typing import List, Optional, Tuple, Union

import numpy as np
import onnxruntime as ort
from fastembed import LateInteractionTextEmbedding, SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client import models as qm


class HybridSearchClient:
    def __init__(
        self,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        collection_name: str = "hybrid-search",
        dense_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        sparse_model: str = "Qdrant/bm25",
        colbert_model: str = "colbert-ir/colbertv2.0",
        use_gpu: bool = True,
    ):
        """
        Initialize the HybridSearchClient.

        Args:
            qdrant_url: Qdrant database URL (defaults to QDRANT_URL env var)
            qdrant_api_key: Qdrant API key (defaults to QDRANT_API_KEY env var)
            collection_name: Name of the Qdrant collection
            dense_model: Dense embedding model name
            sparse_model: Sparse embedding model name
            colbert_model: ColBERT model for reranking
            use_gpu: Whether to use GPU acceleration if available
        """
        # Get credentials from environment if not provided
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
        self.qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")

        if not self.qdrant_url or not self.qdrant_api_key:
            raise ValueError(
                "Qdrant URL and API key must be provided via parameters or environment variables"
            )

        self.collection_name = collection_name

        # Setup ONNX providers
        available_providers = ort.get_available_providers()
        if use_gpu and "CUDAExecutionProvider" in available_providers:
            self.providers = ["CUDAExecutionProvider"]
        else:
            self.providers = ["CPUExecutionProvider"]

        # Initialize clients and models
        self.client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            prefer_grpc=False,
            timeout=60,  # Increase timeout to 60 seconds
        )

        self.dense_model = TextEmbedding(dense_model, providers=self.providers, threads=None)
        self.sparse_model = SparseTextEmbedding(sparse_model)
        self.colbert_model = LateInteractionTextEmbedding(colbert_model, providers=self.providers)

    def _to_filter(self, filters: Optional[dict]) -> Optional[qm.Filter]:
        """Convert dictionary filters to Qdrant Filter format."""
        if not filters:
            return None

        must = []
        for k, v in filters.items():
            if isinstance(v, dict):
                # Handle range filters
                rng_kwargs = {key: v[key] for key in ("gt", "gte", "lt", "lte") if key in v}
                if rng_kwargs:
                    must.append(qm.FieldCondition(key=k, range=qm.Range(**rng_kwargs)))
                    continue
            # Handle equality filters
            must.append(qm.FieldCondition(key=k, match=qm.MatchValue(value=v)))

        return qm.Filter(must=must) if must else None

    def hybrid_search(
        self,
        query: str,
        topk: int = 10,
        prefetch_k: int = 100,
        fusion: str = "DBSF",
        filters: Optional[dict] = None,
    ):
        """
        Perform hybrid search combining dense and sparse embeddings.

        Args:
            query: Search query text
            topk: Number of results to return
            prefetch_k: Number of candidates to prefetch for each embedding type
            fusion: Fusion method - "RRF" (Reciprocal Rank Fusion) or "DBSF"
            filters: Optional filters to apply to search results

        Returns:
            Qdrant query result with points and scores
        """
        # Generate embeddings
        q_dense = next(self.dense_model.embed([query]))
        q_sparse = next(self.sparse_model.embed([query])).as_object()

        # Setup prefetch queries
        prefetch = [
            qm.Prefetch(query=q_sparse, using="bm25", limit=prefetch_k),
            qm.Prefetch(query=q_dense, using="dense", limit=prefetch_k),
        ]

        # Choose fusion method
        fusion_enum = qm.Fusion.RRF if fusion.upper() == "RRF" else qm.Fusion.DBSF

        return self.client.query_points(
            collection_name=self.collection_name,
            prefetch=prefetch,
            query=qm.FusionQuery(fusion=fusion_enum),
            limit=topk,
            query_filter=self._to_filter(filters),
            with_payload=True,
        )

    def _normalize_rows(self, x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """L2 normalize rows for cosine similarity computation."""
        n = np.linalg.norm(x, axis=1, keepdims=True) + eps
        return x / n

    def _colbert_score(self, q_tok: np.ndarray, d_tok: np.ndarray) -> float:
        """
        Compute ColBERT score between query and document token embeddings.

        Args:
            q_tok: Query token embeddings [Q, d]
            d_tok: Document token embeddings [D, d]

        Returns:
            ColBERT score: sum_i max_j cosine(q_i, d_j)
        """
        qn = self._normalize_rows(q_tok)
        dn = self._normalize_rows(d_tok)
        sims = qn @ dn.T  # [Q, D]
        return float(sims.max(axis=1).sum())

    def rerank_colbert(
        self,
        query_text: str,
        qdrant_result,
        topk: int = 10,
        rerank_k: int = 100,
        return_scores: bool = False,
    ) -> Union[List, List[Tuple]]:
        """
        Rerank search results using ColBERT scoring.

        Args:
            query_text: Original search query
            qdrant_result: Result from hybrid_search()
            topk: Number of final results to return
            rerank_k: Number of candidates to rerank with ColBERT
            return_scores: Whether to return scores along with results

        Returns:
            List of reranked points, optionally with scores
        """
        points = qdrant_result.points[:rerank_k]
        if not points:
            return []

        # Extract document texts
        docs = [p.payload.get("document", "") or "" for p in points]

        # Generate embeddings
        q_tok = next(self.colbert_model.query_embed(query_text))  # [Q, d]
        d_tok_list = list(self.colbert_model.embed(docs))  # List of [D_i, d]

        # Compute scores
        scores = []
        for d_tok in d_tok_list:
            if d_tok.size == 0:
                scores.append(-1e9)
            else:
                scores.append(self._colbert_score(q_tok, d_tok))

        # Sort by score
        order = np.argsort(-np.asarray(scores))

        if return_scores:
            return [(points[i], float(scores[i])) for i in order[:topk]]
        return [points[i] for i in order[:topk]]

    def search_and_rerank(
        self,
        query: str,
        topk: int = 10,
        prefetch_k: int = 64,
        rerank_k: int = 100,
        fusion: str = "RRF",
        filters: Optional[dict] = None,
        return_scores: bool = False,
    ) -> Union[List, List[Tuple]]:
        """
        Convenience method that performs hybrid search followed by ColBERT reranking.

        Args:
            query: Search query text
            topk: Number of final results to return
            prefetch_k: Number of candidates to prefetch for each embedding type
            rerank_k: Number of candidates to rerank with ColBERT
            fusion: Fusion method - "RRF" or "DBSF"
            filters: Optional filters to apply
            return_scores: Whether to return ColBERT scores

        Returns:
            List of reranked results, optionally with scores
        """
        # Perform hybrid search
        search_results = self.hybrid_search(
            query=query,
            topk=rerank_k,  # Get more candidates for reranking
            prefetch_k=prefetch_k,
            fusion=fusion,
            filters=filters,
        )

        # Rerank with ColBERT
        return self.rerank_colbert(
            query_text=query,
            qdrant_result=search_results,
            topk=topk,
            rerank_k=rerank_k,
            return_scores=return_scores,
        )
