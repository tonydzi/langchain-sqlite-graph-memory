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
  рельса: none, no LLM call. Embedding (and the optional cross-encoder) run locally.

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
    an optional cross-encoder reranks the union. The index is two files on disk; no server.

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
    """Wikilink expansion. "auto" (default) = exactly `sgm-ask --graph`: full 1-hop for topic
    queries; for name/tool lookups (looks_like_entity) only the `superseded_by` hop, so an
    outdated note still brings its replacement. True = full 1-hop always, False = none."""
    fetch_k: Optional[int] = Field(default=None, ge=1)
    """Unique notes taken from vector search before expansion and rerank. None follows the
    core: env BRAIN_TOPK_RETRIEVE, else 60."""
    ppr: Optional[bool] = None
    """Personalized PageRank fusion over [[wikilinks]]. None follows the core (env SGM_PPR,
    off by default); True/False force it."""
    encoder: Any = None
    """Optional preloaded encoder with `.encode(...)`. Default: the SGM e5 model."""
    rerank: Optional[bool] = None
    """Cross-encoder rerank. None (default) follows the core library, which since 0.3.7
    keeps the dense e5 order unless SGM_RERANK=1, because the mmarco reranker lowered
    recall on long queries. True forces the rerank, False forces dense order. Passing
    a `reranker` object with rerank=None turns it on."""
    reranker: Any = None
    """Optional preloaded cross-encoder with `.predict(pairs)`. Loaded only when rerank is on."""

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

    def _use_rerank(self) -> bool:
        if self.rerank is not None:
            return bool(self.rerank)
        if self.reranker is not None:
            return True
        return bool(_sgm.rerank_enabled(argv=[]))

    def _use_ppr(self) -> bool:
        if self.ppr is not None:
            return bool(self.ppr)
        return bool(_sgm.ppr_enabled())

    def _pool(self) -> int:
        if self.fetch_k is not None:
            return int(self.fetch_k)
        try:
            pool, _ = _sgm._parse_pool([])
        except SystemExit as exc:  # core rejects a bad BRAIN_TOPK_RETRIEVE this way
            raise ValueError("invalid BRAIN_TOPK_RETRIEVE: expected a positive integer") from exc
        return int(pool) if pool is not None else int(_sgm.TOPK_RETRIEVE)

    def _graph_added(self, sims: Any, base_order: List[int], query: str) -> List[int]:
        """Same branches as sgm-ask main(): graph off, full hop, or the entity gate."""
        if not base_order or self.graph is False:
            return []
        if self.graph == "auto" and _sgm.looks_like_entity(query):
            return _sgm.expand_1hop(
                sims, self._meta, base_order, by_base=self._by_base, only_superseded=True
            )
        return _sgm.expand_1hop(sims, self._meta, base_order, by_base=self._by_base)

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
        base_order = _sgm.retrieve_candidates(sims, meta, topk=self._pool())
        added = self._graph_added(sims, base_order, query)
        via_graph = set(added)
        ce = None
        if self._use_rerank():
            if self.reranker is None:  # lazy: the decision is re-read on every call
                self.reranker = _sgm.load_reranker()
            ce = self.reranker
        cand = base_order + added
        ppr_stems = _sgm.ppr_lane(meta, sims, base_order) if (base_order and self._use_ppr()) else []
        if ppr_stems:
            full = _sgm.rerank_candidates(ce, query, cand, meta, topn=None, sims=sims)
            ranked = _sgm.fuse_ppr(full, meta, ppr_stems, k)
        else:
            ranked = _sgm.rerank_candidates(ce, query, cand, meta, topn=k, sims=sims)
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
