import pytest
from langchain_core.documents import Document

from langchain_sqlite_graph_memory import SQLiteGraphMemoryRetriever
from tests.fakes import FakeEncoder, FakeReranker, build_vault_and_index


@pytest.fixture()
def index_dir(tmp_path):
    return str(build_vault_and_index(tmp_path))


def make(index_dir, **kw):
    return SQLiteGraphMemoryRetriever(
        index_dir=index_dir, encoder=FakeEncoder(), reranker=FakeReranker(), **kw
    )


def test_returns_documents_with_metadata(index_dir):
    docs = make(index_dir, k=2).invoke("deploy release checklist")
    assert docs and all(isinstance(d, Document) for d in docs)
    top = docs[0]
    assert top.metadata["title"] == "alpha"
    assert top.metadata["rank"] == 1
    assert top.metadata["path"].endswith("alpha.md")
    assert top.metadata["source"] == top.metadata["path"]
    assert "release checklist" in top.page_content


def test_wikilink_hop_brings_in_linked_note(index_dir):
    # fetch_k=1: vector search alone sees only `alpha`; `beta` shares no word with the
    # query and is reachable only through alpha's [[beta]] link.
    docs = make(index_dir, k=3, fetch_k=1, graph=True).invoke("deploy checklist")
    titles = [d.metadata["title"] for d in docs]
    assert titles == ["alpha", "beta"]
    assert docs[1].metadata["via_graph"] is True
    assert docs[0].metadata["via_graph"] is False


def test_graph_off_is_plain_vector_plus_rerank(index_dir):
    docs = make(index_dir, k=3, fetch_k=1, graph=False).invoke("deploy checklist")
    assert [d.metadata["title"] for d in docs] == ["alpha"]


def test_auto_skips_graph_for_entity_lookups(index_dir):
    # "Obsidian" is a known tool token -> looks_like_entity -> no expansion
    docs = make(index_dir, k=3, fetch_k=1).invoke("Obsidian")
    assert not any(d.metadata["via_graph"] for d in docs)
    docs = make(index_dir, k=3, fetch_k=1).invoke("deploy checklist")
    assert any(d.metadata["via_graph"] for d in docs)


def test_k_kwarg_overrides_constructor(index_dir):
    r = make(index_dir, k=4)
    assert len(r.invoke("release deploy")) == 4
    assert len(r.invoke("release deploy", k=1)) == 1


def test_missing_index_says_how_to_build_one(tmp_path):
    r = make(str(tmp_path / "nowhere"))
    with pytest.raises(FileNotFoundError, match="sgm-index"):
        r.invoke("anything")


def test_same_order_as_the_sgm_pipeline(index_dir):
    """The adapter must not re-implement retrieval: its order equals what the SGM
    functions produce when called directly, the way sgm-ask calls them."""
    from sqlite_graph_memory import brain_ask as sgm
    from pathlib import Path

    q = "release deploy version"
    emb, meta = sgm.load_index(Path(index_dir) / "_brain_e5.npy", Path(index_dir) / "_brain_e5_meta.pkl")
    sims = sgm.query_sims(FakeEncoder(), emb, q)
    base = sgm.retrieve_candidates(sims, meta)
    added = sgm.expand_1hop(sims, meta, base)
    want = [meta[i]["title"] for i, _ in sgm.rerank_candidates(FakeReranker(), q, base + added, meta, topn=3)]
    got = [d.metadata["title"] for d in make(index_dir, k=3, graph=True).invoke(q)]
    assert got == want


def _dense(index_dir, **kw):
    """Retriever with the core default: no reranker object, rerank=None."""
    return SQLiteGraphMemoryRetriever(index_dir=index_dir, encoder=FakeEncoder(), **kw)


def test_default_follows_core_dense_order_and_never_loads_reranker(index_dir, monkeypatch):
    """Core >=0.3.7 keeps the dense e5 order unless SGM_RERANK=1. The adapter must match
    sgm-ask's default, and must not pull the cross-encoder in at all."""
    from sqlite_graph_memory import brain_ask as sgm
    from pathlib import Path

    monkeypatch.delenv("SGM_RERANK", raising=False)
    monkeypatch.setattr(sgm, "load_reranker", lambda *a, **k: (_ for _ in ()).throw(AssertionError("reranker loaded")))
    q = "release deploy version"
    emb, meta = sgm.load_index(Path(index_dir) / "_brain_e5.npy", Path(index_dir) / "_brain_e5_meta.pkl")
    sims = sgm.query_sims(FakeEncoder(), emb, q)
    base = sgm.retrieve_candidates(sims, meta)
    added = sgm.expand_1hop(sims, meta, base)
    want = [meta[i]["title"] for i, _ in sgm.rerank_candidates(None, q, base + added, meta, topn=3, sims=sims)]
    docs = _dense(index_dir, k=3, graph=True).invoke(q)
    assert [d.metadata["title"] for d in docs] == want
    assert docs[0].metadata["score"] == pytest.approx(float(max(sims)), rel=1e-5)


def test_sgm_rerank_env_turns_rerank_on(index_dir, monkeypatch):
    from sqlite_graph_memory import brain_ask as sgm

    calls = []
    monkeypatch.setenv("SGM_RERANK", "1")
    monkeypatch.setattr(sgm, "load_reranker", lambda *a, **k: calls.append(1) or FakeReranker())
    _dense(index_dir, k=2).invoke("deploy release")
    assert calls == [1]


def test_rerank_false_ignores_a_passed_reranker(index_dir):
    class Boom:
        def predict(self, pairs):
            raise AssertionError("reranker used despite rerank=False")

    docs = SQLiteGraphMemoryRetriever(
        index_dir=index_dir, encoder=FakeEncoder(), reranker=Boom(), rerank=False, k=2
    ).invoke("deploy release")
    assert len(docs) == 2


async def _ainvoke(r, q):
    return await r.ainvoke(q)


def test_async_invoke(index_dir):
    import asyncio

    docs = asyncio.run(_ainvoke(make(index_dir, k=2), "deploy release"))
    assert len(docs) == 2
