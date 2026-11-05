"""LangChain retriever over a sqlite-graph-memory index.

Паспорт (for the people who maintain this):
  in:  a query string; an index built by `sgm-index` (two files: _brain_e5.npy,
       _brain_e5_meta.pkl) in `index_dir`.
  out: list[Document], best first. page_content = the matched chunk; metadata =
       path, title, date, score, rank, via_graph.
  who calls: any LangChain / LangGraph app (`retriever.invoke(q)`, a tool, a chain).
  what breaks it: sqlite-graph-memory changing the signatures of load_index,
       query_sims, retrieve_candidates, expand_1hop, rerank_candidates. The test
       suite pins that contract with fake models, so a drift fails CI, not a user.
  рельса: none, no LLM call. Embedding + cross-encoder run locally.

It runs the SAME pipeline functions `sgm-ask` runs (imported from
sqlite_graph_memory.brain_ask), so recall through LangChain cannot drift away from
recall through the CLI or the MCP server.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Literal, Optional, Union

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field, PrivateAttr

from sqlite_graph_memory import brain_ask as _sgm

GraphMode = Union[bool, Literal["auto"]]


class SQLiteGraphMemoryRetriever(BaseRetriever):
    """Associative recall over a folder of markdown notes.

    Vector search finds entry points, 1-hop ``[[wikilinks]]`` pull in linked notes,
    a cross-encoder reranks the union. The index is two files on disk; no server.

    Setup:
        .. code-block:: bash

            pip install "langchain-sqlite-graph-memory[lite]"
            sgm-index /path/to/vault --out ./index

    Instantiate:
        .. code-block:: python

            from langchain_sqlite_graph_memory import SQLiteGraphMemoryRetriever

            retriever = SQLiteGraphMemoryRetriever(index_dir="./index", k=8)

    Invoke:
        .. code-block:: python

            docs = retriever.invoke("why did we drop the graph database?")
            for d in docs:
                print(d.metadata["rank"], d.metadata["title"], d.metadata["via_graph"])
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    index_dir: Optional[str] = None
    """Directory with the `sgm-index` output. Default: env BRAIN_INDEX_DIR, then ./index."""
    k: int = Field(default=_sgm.TOPN, ge=1)
    """How many documents to return."""
    graph: GraphMode = "auto"
    """Wikilink expansion. True = always, False = never (plain vector + rerank),
    "auto" = on, except for name/tool lookups, where it measured as noise
    (sqlite_graph_memory.brain_ask.looks_like_entity)."""
    fetch_k: int = Field(default=_sgm.TOPK_RETRIEVE, ge=1)
    """Unique notes taken from vector search before expansion and rerank."""
    encoder: Any = None
    """Optional preloaded encoder with `.encode(...)`. Default: the SGM e5 model."""
    reranker: Any = None
    """Optional preloaded cross-encoder with `.predict(pairs)`. Default: the SGM reranker."""

    _emb: Any = PrivateAttr(default=None)
    _meta: Any = PrivateAttr(default=None)
    _by_base: Any = PrivateAttr(default=None)

    # ------------------------------------------------------------------ loading
    def _index_paths(self) -> tuple[Path, Path]:
        base = Path(self.index_dir or os.getenv("BRAIN_INDEX_DIR", "index"))
        return base / "_brain_e5.npy", base / "_brain_e5_meta.pkl"

    def _ensure_loaded(self) -> None:
        if self._meta is not None:
            return
        emb_path, meta_path = self._index_paths()
        if not emb_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                "No sqlite-graph-memory index at %s. Build one first: "
                "sgm-index <notes folder> (see https://github.com/tonydzi/sqlite-graph-memory)"
                % emb_path.parent
            )
        self._emb, self._meta = _sgm.load_index(emb_path, meta_path)
        self._by_base = _sgm.index_by_basename(self._meta)
        if self.encoder is None:
            self.encoder = _sgm.load_encoder()
        if self.reranker is None:
            self.reranker = _sgm.load_reranker()

    def _use_graph(self, query: str) -> bool:
        if self.graph == "auto":
            return not _sgm.looks_like_entity(query)
        return bool(self.graph)

    # ------------------------------------------------------------------ retrieval
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        k = int(kwargs.get("k", self.k))
        self._ensure_loaded()
        meta = self._meta
        sims = _sgm.query_sims(self.encoder, self._emb, query)
        base_order = _sgm.retrieve_candidates(sims, meta, topk=self.fetch_k)
        added: List[int] = []
        if base_order and self._use_graph(query):
            added = _sgm.expand_1hop(sims, meta, base_order, by_base=self._by_base)
        via_graph = set(added)
        ranked = _sgm.rerank_candidates(
            self.reranker, query, base_order + added, meta, topn=k
        )
        docs = []
        for rank, (i, score) in enumerate(ranked, 1):
            m = meta[i]
            docs.append(
                Document(
                    page_content=m.get("snippet", ""),
                    metadata={
                        "path": m.get("path"),
                        "title": m.get("title"),
                        "date": m.get("date"),
                        "score": float(score),
                        "rank": rank,
                        "via_graph": i in via_graph,
                        "source": m.get("path"),
                    },
                )
            )
        return docs
