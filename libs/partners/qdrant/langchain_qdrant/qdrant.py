from __future__ import annotations

import uuid
from enum import Enum
from itertools import islice
from operator import itemgetter
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from qdrant_client import QdrantClient, models

from langchain_qdrant._utils import maximal_marginal_relevance
from langchain_qdrant.sparse_embeddings import SparseEmbeddings


class QdrantVectorStoreError(Exception):
    """`QdrantVectorStore` related exceptions."""


class RetrievalMode(str, Enum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


class QdrantVectorStore(VectorStore):
    """`QdrantVectorStore` - Vector store implementation using https://qdrant.tech/

    Example:
        .. code-block:: python
        from langchain_qdrant import QdrantVectorStore

        store = QdrantVectorStore.from_existing_collection("my-collection", embedding, url="http://localhost:6333")
    """

    CONTENT_KEY: str = "page_content"
    METADATA_KEY: str = "metadata"
    VECTOR_NAME: str = ""  # The default/unnamed vector - https://qdrant.tech/documentation/concepts/collections/#create-a-collection
    SPARSE_VECTOR_NAME: str = "langchain-sparse"

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        embedding: Optional[Embeddings] = None,
        retrieval_mode: RetrievalMode = RetrievalMode.DENSE,
        vector_name: str = VECTOR_NAME,
        content_payload_key: str = CONTENT_KEY,
        metadata_payload_key: str = METADATA_KEY,
        distance: models.Distance = models.Distance.COSINE,
        sparse_embedding: Optional[SparseEmbeddings] = None,
        sparse_vector_name: str = SPARSE_VECTOR_NAME,
        validate_embeddings: bool = True,
        validate_collection_config: bool = True,
    ):
        if validate_embeddings:
            self._validate_embeddings(retrieval_mode, embedding, sparse_embedding)

        if validate_collection_config:
            self._validate_collection_config(
                client,
                collection_name,
                retrieval_mode,
                vector_name,
                sparse_vector_name,
                distance,
                embedding,
            )

        self._client = client
        self.collection_name = collection_name
        self._embeddings = embedding
        self.retrieval_mode = retrieval_mode
        self.vector_name = vector_name
        self.content_payload_key = content_payload_key
        self.metadata_payload_key = metadata_payload_key
        self.distance = distance
        self._sparse_embeddings = sparse_embedding
        self.sparse_vector_name = sparse_vector_name

    @property
    def client(self) -> QdrantClient:
        return self._client

    @property
    def embeddings(self) -> Embeddings:
        if self._embeddings is None:
            raise ValueError(
                "Embeddings are not set. Please set embeddings using the `embedding` parameter."
            )
        return self._embeddings

    @property
    def sparse_embeddings(self) -> SparseEmbeddings:
        if self._sparse_embeddings is None:
            raise ValueError(
                "Sparse embeddings are not set. Please set sparse embeddings using the `sparse_embedding` parameter."
            )
        return self._sparse_embeddings

    @classmethod
    def from_texts(
        cls: Type[QdrantVectorStore],
        texts: List[str],
        embedding: Optional[Embeddings] = None,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[Sequence[str | int]] = None,
        collection_name: Optional[str] = None,
        location: Optional[str] = None,
        url: Optional[str] = None,
        port: Optional[int] = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,
        https: Optional[bool] = None,
        api_key: Optional[str] = None,
        prefix: Optional[str] = None,
        timeout: Optional[int] = None,
        host: Optional[str] = None,
        path: Optional[str] = None,
        distance: models.Distance = models.Distance.COSINE,
        content_payload_key: str = CONTENT_KEY,
        metadata_payload_key: str = METADATA_KEY,
        vector_name: str = VECTOR_NAME,
        retrieval_mode: RetrievalMode = RetrievalMode.DENSE,
        sparse_embedding: Optional[SparseEmbeddings] = None,
        sparse_vector_name: str = SPARSE_VECTOR_NAME,
        collection_create_options: Dict[str, Any] = {},
        vector_params: Dict[str, Any] = {},
        sparse_vector_params: Dict[str, Any] = {},
        batch_size: int = 64,
        force_recreate: bool = False,
        validate_embeddings: bool = True,
        validate_collection_config: bool = True,
        **kwargs: Any,
    ) -> QdrantVectorStore:
        client_options = {
            "location": location,
            "url": url,
            "port": port,
            "grpc_port": grpc_port,
            "prefer_grpc": prefer_grpc,
            "https": https,
            "api_key": api_key,
            "prefix": prefix,
            "timeout": timeout,
            "host": host,
            "path": path,
            **kwargs,
        }

        qdrant = cls.construct_instance(
            embedding,
            retrieval_mode,
            sparse_embedding,
            client_options,
            collection_name,
            distance,
            content_payload_key,
            metadata_payload_key,
            vector_name,
            sparse_vector_name,
            force_recreate,
            collection_create_options,
            vector_params,
            sparse_vector_params,
            validate_embeddings,
            validate_collection_config,
        )
        qdrant.add_texts(texts, metadatas, ids, batch_size)
        return qdrant

    @classmethod
    def from_existing_collection(
        cls: Type[QdrantVectorStore],
        collection_name: str,
        embedding: Optional[Embeddings] = None,
        retrieval_mode: RetrievalMode = RetrievalMode.DENSE,
        location: Optional[str] = None,
        url: Optional[str] = None,
        port: Optional[int] = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,
        https: Optional[bool] = None,
        api_key: Optional[str] = None,
        prefix: Optional[str] = None,
        timeout: Optional[int] = None,
        host: Optional[str] = None,
        path: Optional[str] = None,
        distance: models.Distance = models.Distance.COSINE,
        content_payload_key: str = CONTENT_KEY,
        metadata_payload_key: str = METADATA_KEY,
        vector_name: str = VECTOR_NAME,
        sparse_vector_name: str = SPARSE_VECTOR_NAME,
        sparse_embedding: Optional[SparseEmbeddings] = None,
        validate_embeddings: bool = True,
        validate_collection_config: bool = True,
        **kwargs: Any,
    ) -> QdrantVectorStore:
        client = QdrantClient(
            location=location,
            url=url,
            port=port,
            grpc_port=grpc_port,
            prefer_grpc=prefer_grpc,
            https=https,
            api_key=api_key,
            prefix=prefix,
            timeout=timeout,
            host=host,
            path=path,
            **kwargs,
        )

        return cls(
            client=client,
            collection_name=collection_name,
            embedding=embedding,
            retrieval_mode=retrieval_mode,
            content_payload_key=content_payload_key,
            metadata_payload_key=metadata_payload_key,
            distance=distance,
            vector_name=vector_name,
            sparse_embedding=sparse_embedding,
            sparse_vector_name=sparse_vector_name,
            validate_embeddings=validate_embeddings,
            validate_collection_config=validate_collection_config,
        )

    def add_texts(  # type: ignore
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[Sequence[str | int]] = None,
        batch_size: int = 64,
        **kwargs: Any,
    ) -> List[str | int]:
        added_ids = []
        for batch_ids, points in self._generate_batches(
            texts, metadatas, ids, batch_size
        ):
            self.client.upsert(
                collection_name=self.collection_name, points=points, **kwargs
            )
            added_ids.extend(batch_ids)

        return added_ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[models.Filter] = None,
        search_params: Optional[models.SearchParams] = None,
        offset: int = 0,
        score_threshold: Optional[float] = None,
        consistency: Optional[models.ReadConsistency] = None,
        hybrid_fusion: Optional[models.FusionQuery] = None,
        **kwargs: Any,
    ) -> List[Document]:
        results = self.similarity_search_with_score(
            query,
            k,
            filter=filter,
            search_params=search_params,
            offset=offset,
            score_threshold=score_threshold,
            consistency=consistency,
            hybrid_fusion=hybrid_fusion,
            **kwargs,
        )
        return list(map(itemgetter(0), results))

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[models.Filter] = None,
        search_params: Optional[models.SearchParams] = None,
        offset: int = 0,
        score_threshold: Optional[float] = None,
        consistency: Optional[models.ReadConsistency] = None,
        hybrid_fusion: Optional[models.FusionQuery] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        query_options = {
            "collection_name": self.collection_name,
            "query_filter": filter,
            "search_params": search_params,
            "limit": k,
            "offset": offset,
            "with_payload": True,
            "with_vectors": False,
            "score_threshold": score_threshold,
            "consistency": consistency,
            **kwargs,
        }
        if self.retrieval_mode == RetrievalMode.DENSE:
            query_dense_embedding = self.embeddings.embed_query(query)
            results = self.client.query_points(
                query=query_dense_embedding,
                using=self.vector_name,
                **query_options,
            ).points

        elif self.retrieval_mode == RetrievalMode.SPARSE:
            query_sparse_embedding = self.sparse_embeddings.embed_query(query)
            results = self.client.query_points(
                query=models.SparseVector(
                    indices=query_sparse_embedding.indices,
                    values=query_sparse_embedding.values,
                ),
                using=self.sparse_vector_name,
                **query_options,
            ).points

        elif self.retrieval_mode == RetrievalMode.HYBRID:
            query_dense_embedding = self.embeddings.embed_query(query)
            query_sparse_embedding = self.sparse_embeddings.embed_query(query)
            results = self.client.query_points(
                prefetch=[
                    models.Prefetch(
                        using=self.vector_name,
                        query=query_dense_embedding,
                        filter=filter,
                        limit=k,
                        params=search_params,
                    ),
                    models.Prefetch(
                        using=self.sparse_vector_name,
                        query=models.SparseVector(
                            indices=query_sparse_embedding.indices,
                            values=query_sparse_embedding.values,
                        ),
                        filter=filter,
                        limit=k,
                        params=search_params,
                    ),
                ],
                query=hybrid_fusion or models.FusionQuery(fusion=models.Fusion.RRF),
                **query_options,
            ).points

        else:
            raise ValueError(f"Invalid retrieval mode. {self.retrieval_mode}.")
        return [
            (
                self._document_from_point(
                    result,
                    self.collection_name,
                    self.content_payload_key,
                    self.metadata_payload_key,
                ),
                result.score,
            )
            for result in results
        ]

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[models.Filter] = None,
        search_params: Optional[models.SearchParams] = None,
        offset: int = 0,
        score_threshold: Optional[float] = None,
        consistency: Optional[models.ReadConsistency] = None,
        **kwargs: Any,
    ) -> List[Document]:
        qdrant_filter = filter

        self._validate_collection_for_dense(
            client=self.client,
            collection_name=self.collection_name,
            vector_name=self.vector_name,
            distance=self.distance,
            dense_embeddings=embedding,
        )
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=embedding,
            using=self.vector_name,
            query_filter=qdrant_filter,
            search_params=search_params,
            limit=k,
            offset=offset,
            with_payload=True,
            with_vectors=False,
            score_threshold=score_threshold,
            consistency=consistency,
            **kwargs,
        ).points

        return [
            self._document_from_point(
                result,
                self.collection_name,
                self.content_payload_key,
                self.metadata_payload_key,
            )
            for result in results
        ]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[models.Filter] = None,
        search_params: Optional[models.SearchParams] = None,
        score_threshold: Optional[float] = None,
        consistency: Optional[models.ReadConsistency] = None,
        **kwargs: Any,
    ) -> List[Document]:
        self._validate_collection_for_dense(
            self.client,
            self.collection_name,
            self.vector_name,
            self.distance,
            self.embeddings,
        )

        query_embedding = self.embeddings.embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            query_embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            search_params=search_params,
            score_threshold=score_threshold,
            consistency=consistency,
            **kwargs,
        )

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[models.Filter] = None,
        search_params: Optional[models.SearchParams] = None,
        score_threshold: Optional[float] = None,
        consistency: Optional[models.ReadConsistency] = None,
        **kwargs: Any,
    ) -> List[Document]:
        results = self.max_marginal_relevance_search_with_score_by_vector(
            embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            search_params=search_params,
            score_threshold=score_threshold,
            consistency=consistency,
            **kwargs,
        )
        return list(map(itemgetter(0), results))

    def max_marginal_relevance_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[models.Filter] = None,
        search_params: Optional[models.SearchParams] = None,
        score_threshold: Optional[float] = None,
        consistency: Optional[models.ReadConsistency] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=embedding,
            query_filter=filter,
            search_params=search_params,
            limit=fetch_k,
            with_payload=True,
            with_vectors=True,
            score_threshold=score_threshold,
            consistency=consistency,
            using=self.vector_name,
            **kwargs,
        ).points

        embeddings = [
            result.vector
            if isinstance(result.vector, list)
            else result.vector.get(self.vector_name)  # type: ignore
            for result in results
        ]
        mmr_selected = maximal_marginal_relevance(
            np.array(embedding), embeddings, k=k, lambda_mult=lambda_mult
        )
        return [
            (
                self._document_from_point(
                    results[i],
                    self.collection_name,
                    self.content_payload_key,
                    self.metadata_payload_key,
                ),
                results[i].score,
            )
            for i in mmr_selected
        ]

    def delete(  # type: ignore
        self,
        ids: Optional[List[str | int]] = None,
        **kwargs: Any,
    ) -> Optional[bool]:
        result = self.client.delete(
            collection_name=self.collection_name,
            points_selector=ids,
        )
        return result.status == models.UpdateStatus.COMPLETED

    def get_by_ids(self, ids: Sequence[str | int], /) -> List[Document]:
        results = self.client.retrieve(self.collection_name, ids, with_payload=True)

        return [
            self._document_from_point(
                result,
                self.collection_name,
                self.content_payload_key,
                self.metadata_payload_key,
            )
            for result in results
        ]

    @classmethod
    def construct_instance(
        cls: Type[QdrantVectorStore],
        embedding: Optional[Embeddings] = None,
        retrieval_mode: RetrievalMode = RetrievalMode.DENSE,
        sparse_embedding: Optional[SparseEmbeddings] = None,
        client_options: Dict[str, Any] = {},
        collection_name: Optional[str] = None,
        distance: models.Distance = models.Distance.COSINE,
        content_payload_key: str = CONTENT_KEY,
        metadata_payload_key: str = METADATA_KEY,
        vector_name: str = VECTOR_NAME,
        sparse_vector_name: str = SPARSE_VECTOR_NAME,
        force_recreate: bool = False,
        collection_create_options: Dict[str, Any] = {},
        vector_params: Dict[str, Any] = {},
        sparse_vector_params: Dict[str, Any] = {},
        validate_embeddings: bool = True,
        validate_collection_config: bool = True,
    ) -> QdrantVectorStore:
        if validate_embeddings:
            cls._validate_embeddings(retrieval_mode, embedding, sparse_embedding)
        collection_name = collection_name or uuid.uuid4().hex
        client = QdrantClient(**client_options)

        collection_exists = client.collection_exists(collection_name)

        if collection_exists and force_recreate:
            client.delete_collection(collection_name)
            collection_exists = False
        if collection_exists:
            if validate_collection_config:
                cls._validate_collection_config(
                    client,
                    collection_name,
                    retrieval_mode,
                    vector_name,
                    sparse_vector_name,
                    distance,
                    embedding,
                )
        else:
            vectors_config, sparse_vectors_config = {}, {}
            if retrieval_mode == RetrievalMode.DENSE:
                partial_embeddings = embedding.embed_documents(["dummy_text"])  # type: ignore

                vector_params["size"] = len(partial_embeddings[0])
                vector_params["distance"] = distance

                vectors_config = {
                    vector_name: models.VectorParams(
                        **vector_params,
                    )
                }

            elif retrieval_mode == RetrievalMode.SPARSE:
                sparse_vectors_config = {
                    sparse_vector_name: models.SparseVectorParams(
                        **sparse_vector_params
                    )
                }

            elif retrieval_mode == RetrievalMode.HYBRID:
                partial_embeddings = embedding.embed_documents(["dummy_text"])  # type: ignore

                vector_params["size"] = len(partial_embeddings[0])
                vector_params["distance"] = distance

                vectors_config = {
                    vector_name: models.VectorParams(
                        **vector_params,
                    )
                }

                sparse_vectors_config = {
                    sparse_vector_name: models.SparseVectorParams(
                        **sparse_vector_params
                    )
                }

            collection_create_options["collection_name"] = collection_name
            collection_create_options["vectors_config"] = vectors_config
            collection_create_options["sparse_vectors_config"] = sparse_vectors_config

            client.create_collection(**collection_create_options)

        qdrant = cls(
            client=client,
            collection_name=collection_name,
            embedding=embedding,
            retrieval_mode=retrieval_mode,
            content_payload_key=content_payload_key,
            metadata_payload_key=metadata_payload_key,
            distance=distance,
            vector_name=vector_name,
            sparse_embedding=sparse_embedding,
            sparse_vector_name=sparse_vector_name,
            validate_embeddings=False,
            validate_collection_config=False,
        )
        return qdrant

    @staticmethod
    def _cosine_relevance_score_fn(distance: float) -> float:
        """Normalize the distance to a score on a scale [0, 1]."""
        return (distance + 1.0) / 2.0

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """
        The 'correct' relevance function
        may differ depending on a few things, including:
        - the distance / similarity metric used by the VectorStore
        - the scale of your embeddings (OpenAI's are unit normed. Many others are not!)
        - embedding dimensionality
        - etc.
        """

        if self.distance == models.Distance.COSINE:
            return self._cosine_relevance_score_fn
        elif self.distance == models.Distance.DOT:
            return self._max_inner_product_relevance_score_fn
        elif self.distance == models.Distance.EUCLID:
            return self._euclidean_relevance_score_fn
        else:
            raise ValueError(
                "Unknown distance strategy, must be COSINE, " "DOT, EUCLID OR L1."
            )

    @classmethod
    def _document_from_point(
        cls,
        scored_point: Any,
        collection_name: str,
        content_payload_key: str,
        metadata_payload_key: str,
    ) -> Document:
        metadata = scored_point.payload.get(metadata_payload_key) or {}
        metadata["_id"] = scored_point.id
        metadata["_collection_name"] = collection_name
        return Document(
            page_content=scored_point.payload.get(content_payload_key, ""),
            metadata=metadata,
        )

    def _generate_batches(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[Sequence[str | int]] = None,
        batch_size: int = 64,
    ) -> Generator[tuple[list[str | int], list[models.PointStruct]], Any, None]:
        texts_iterator = iter(texts)
        metadatas_iterator = iter(metadatas or [])
        ids_iterator = iter(ids or [uuid.uuid4().hex for _ in iter(texts)])

        while batch_texts := list(islice(texts_iterator, batch_size)):
            batch_metadatas = list(islice(metadatas_iterator, batch_size)) or None
            batch_ids = list(islice(ids_iterator, batch_size))
            points = [
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
                for point_id, vector, payload in zip(
                    batch_ids,
                    self._build_vectors(batch_texts),
                    self._build_payloads(
                        batch_texts,
                        batch_metadatas,
                        self.content_payload_key,
                        self.metadata_payload_key,
                    ),
                )
            ]

            yield batch_ids, points

    def _build_payloads(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]],
        content_payload_key: str,
        metadata_payload_key: str,
    ) -> List[dict]:
        payloads = []
        for i, text in enumerate(texts):
            if text is None:
                raise ValueError(
                    "At least one of the texts is None. Please remove it before "
                    "calling .from_texts or .add_texts."
                )
            metadata = metadatas[i] if metadatas is not None else None
            payloads.append(
                {
                    content_payload_key: text,
                    metadata_payload_key: metadata,
                }
            )

        return payloads

    def _build_vectors(
        self,
        texts: Iterable[str],
    ) -> List[models.VectorStruct]:
        if self.retrieval_mode == RetrievalMode.DENSE:
            batch_embeddings = self.embeddings.embed_documents(list(texts))
            return [
                {
                    self.vector_name: vector,
                }
                for vector in batch_embeddings
            ]

        elif self.retrieval_mode == RetrievalMode.SPARSE:
            batch_sparse_embeddings = self.sparse_embeddings.embed_documents(
                list(texts)
            )
            return [
                {
                    self.sparse_vector_name: models.SparseVector(
                        values=vector.values, indices=vector.indices
                    )
                }
                for vector in batch_sparse_embeddings
            ]

        elif self.retrieval_mode == RetrievalMode.HYBRID:
            dense_embeddings = self.embeddings.embed_documents(list(texts))
            sparse_embeddings = self.sparse_embeddings.embed_documents(list(texts))

            assert len(dense_embeddings) == len(
                sparse_embeddings
            ), "Mismatched length between dense and sparse embeddings."

            return [
                {
                    self.vector_name: dense_vector,
                    self.sparse_vector_name: models.SparseVector(
                        values=sparse_vector.values, indices=sparse_vector.indices
                    ),
                }
                for dense_vector, sparse_vector in zip(
                    dense_embeddings, sparse_embeddings
                )
            ]

        else:
            raise ValueError(
                f"Unknown retrieval mode. {self.retrieval_mode} to build vectors."
            )

    @classmethod
    def _validate_collection_config(
        cls: Type[QdrantVectorStore],
        client: QdrantClient,
        collection_name: str,
        retrieval_mode: RetrievalMode,
        vector_name: str,
        sparse_vector_name: str,
        distance: models.Distance,
        embedding: Optional[Embeddings],
    ) -> None:
        if retrieval_mode == RetrievalMode.DENSE:
            cls._validate_collection_for_dense(
                client, collection_name, vector_name, distance, embedding
            )

        elif retrieval_mode == RetrievalMode.SPARSE:
            cls._validate_collection_for_sparse(
                client, collection_name, sparse_vector_name
            )

        elif retrieval_mode == RetrievalMode.HYBRID:
            cls._validate_collection_for_dense(
                client, collection_name, vector_name, distance, embedding
            )
            cls._validate_collection_for_sparse(
                client, collection_name, sparse_vector_name
            )

    @classmethod
    def _validate_collection_for_dense(
        cls: Type[QdrantVectorStore],
        client: QdrantClient,
        collection_name: str,
        vector_name: str,
        distance: models.Distance,
        dense_embeddings: Union[Embeddings, List[float], None],
    ) -> None:
        collection_info = client.get_collection(collection_name=collection_name)
        vector_config = collection_info.config.params.vectors

        if isinstance(vector_config, models.VectorParams) and vector_name != "":
            # qdrant-client returns a single VectorParams object in case of a single unnamed vector

            raise QdrantVectorStoreError(
                f"Existing Qdrant collection {collection_name} is built with unnamed dense vector. "
                f"If you want to reuse it, please set `vector_name` to ''(empty string)."
                f"If you want to recreate the collection, set `force_recreate` to `True`."
            )

        else:
            # vector_config is a Dict[str, VectorParams]
            if isinstance(vector_config, dict) and vector_name not in vector_config:
                raise QdrantVectorStoreError(
                    f"Existing Qdrant collection {collection_name} does not "
                    f"contain dense vector named {vector_name}. Did you mean one of the "
                    f"existing vectors: {', '.join(vector_config.keys())}? "  # type: ignore
                    f"If you want to recreate the collection, set `force_recreate` "
                    f"parameter to `True`."
                )

            # Get the VectorParams object for the specified vector_name
            vector_config = vector_config[vector_name]  # type: ignore

        if isinstance(dense_embeddings, Embeddings):
            vector_size = len(dense_embeddings.embed_documents(["dummy_text"])[0])
        elif isinstance(dense_embeddings, list):
            vector_size = len(dense_embeddings)
        else:
            raise ValueError("Invalid `embeddings` type.")

        if vector_config.size != vector_size:
            raise QdrantVectorStoreError(
                f"Existing Qdrant collection is configured for dense vectors with "
                f"{vector_config.size} dimensions. "
                f"Selected embeddings are {vector_size}-dimensional. "
                f"If you want to recreate the collection, set `force_recreate` "
                f"parameter to `True`."
            )

        if vector_config.distance != distance:
            raise QdrantVectorStoreError(
                f"Existing Qdrant collection is configured for "
                f"{vector_config.distance.name} similarity, but requested "
                f"{distance.upper()}. Please set `distance` parameter to "
                f"`{vector_config.distance.name}` if you want to reuse it. "
                f"If you want to recreate the collection, set `force_recreate` "
                f"parameter to `True`."
            )

    @classmethod
    def _validate_collection_for_sparse(
        cls: Type[QdrantVectorStore],
        client: QdrantClient,
        collection_name: str,
        sparse_vector_name: str,
    ) -> None:
        collection_info = client.get_collection(collection_name=collection_name)
        sparse_vector_config = collection_info.config.params.sparse_vectors

        if (
            sparse_vector_config is None
            or sparse_vector_name not in sparse_vector_config
        ):
            raise QdrantVectorStoreError(
                f"Existing Qdrant collection {collection_name} does not "
                f"contain sparse vectors named {sparse_vector_config}. "
                f"If you want to recreate the collection, set `force_recreate` "
                f"parameter to `True`."
            )

    @classmethod
    def _validate_embeddings(
        cls: Type[QdrantVectorStore],
        retrieval_mode: RetrievalMode,
        embedding: Optional[Embeddings],
        sparse_embedding: Optional[SparseEmbeddings],
    ) -> None:
        if retrieval_mode == RetrievalMode.DENSE and embedding is None:
            raise ValueError(
                "'embedding' cannot be None when retrieval mode is 'dense'"
            )

        elif retrieval_mode == RetrievalMode.SPARSE and sparse_embedding is None:
            raise ValueError(
                "'sparse_embedding' cannot be None when retrieval mode is 'sparse'"
            )

        elif retrieval_mode == RetrievalMode.HYBRID and any(
            [embedding is None, sparse_embedding is None]
        ):
            raise ValueError(
                "Both 'embedding' and 'sparse_embedding' cannot be None when retrieval mode is 'hybrid'"
            )
