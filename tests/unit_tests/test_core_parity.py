"""Parity with `sgm-ask --graph` beyond the default path: the entity gate's superseded_by
hop, the rerank-on path, BRAIN_TOPK_RETRIEVE, SGM_PPR and a rerank switched on late.
Each test pins one branch of sqlite_graph_memory.brain_ask.main()."""
import pickle
from pathlib import Path

import numpy as np
import pytest

from langchain_sqlite_graph_memory import SQLiteGraphMemoryRetriever
from sqlite_graph_memory import brain_ask as sgm
from tests.fakes import FakeEncoder, FakeReranker, _vec, build_vault_and_index


@pytest.fixture()
def index_dir(tmp_path):
    return str(build_vault_and_index(tmp_path))


@pytest.fixture()
def superseded_index(tmp_path):
    """`obsidianold` is the only vector hit for 'Obsidian' and names its replacement in
    frontmatter. `newsetup` shares no word with the query."""
    vault, index = tmp_path / "vault", tmp_path / "index"
    vault.mkdir()
    index.mkdir()
    notes = {
        "obsidianold": '---\nsuperseded_by: "[[newsetup]]"\n---\nObsidian setup, old way. See [[filler]].',
        # no word may hash into obsidian's bucket (44 of 64): "sync" did, and leaked a dense hit
        "newsetup": "Current vault layout and backup rules.",
        "filler": "Unrelated grocery list.",
    }
    meta, rows = [], []
    for stem, body in notes.items():
        p = vault / ("%s.md" % stem)
        p.write_text(body, encoding="utf-8")
        meta.append({"path": str(p), "title": stem, "snippet": body, "date": "2026-11-16"})
        rows.append(_vec("passage: " + body))
    np.save(index / "_brain_e5.npy", np.stack(rows))
    (index / "_brain_e5_meta.pkl").write_bytes(pickle.dumps(meta))
    return str(index)


def _r(index_dir, **kw):
    kw.setdefault("encoder", FakeEncoder())
    return SQLiteGraphMemoryRetriever(index_dir=index_dir, **kw)


def test_entity_query_still_follows_superseded_by(superseded_index, monkeypatch):
    monkeypatch.delenv("SGM_RERANK", raising=False)
    assert sgm.looks_like_entity("Obsidian")
    docs = _r(superseded_index, k=5, fetch_k=1).invoke("Obsidian")
    titles = [d.metadata["title"] for d in docs]
    assert titles[0] == "obsidianold"
    assert "newsetup" in titles, "gated query lost the superseded_by replacement"
    assert "filler" not in titles, "gated query must not do the general wikilink hop"
    assert [d.metadata["via_graph"] for d in docs if d.metadata["title"] == "newsetup"] == [True]


def test_graph_false_skips_even_superseded(superseded_index):
    docs = _r(superseded_index, k=5, fetch_k=1, graph=False).invoke("Obsidian")
    assert [d.metadata["title"] for d in docs] == ["obsidianold"]


class ReverseReranker:
    """Scores the reverse of the dense order, so rerank-on and dense disagree."""

    def __init__(self):
        self.calls = 0

    def predict(self, pairs):
        self.calls += 1
        return np.array([100.0 + i for i in range(len(pairs))], dtype="float32")


def test_rerank_on_path_uses_reranker_order_and_scores(index_dir, monkeypatch):
    monkeypatch.delenv("SGM_RERANK", raising=False)
    dense = [d.metadata["title"] for d in _r(index_dir, k=4, graph=False).invoke("deploy release")]
    rr = ReverseReranker()
    docs = _r(index_dir, k=4, graph=False, reranker=rr).invoke("deploy release")
    assert rr.calls == 1
    assert [d.metadata["title"] for d in docs] == list(reversed(dense))
    assert docs[0].metadata["score"] >= 100.0


def test_rerank_true_loads_core_reranker(index_dir, monkeypatch):
    monkeypatch.delenv("SGM_RERANK", raising=False)
    rr = ReverseReranker()
    monkeypatch.setattr(sgm, "load_reranker", lambda *a, **k: rr)
    _r(index_dir, k=2, rerank=True).invoke("deploy release")
    assert rr.calls == 1


def test_rerank_switched_on_after_first_call_is_honoured(index_dir, monkeypatch):
    monkeypatch.delenv("SGM_RERANK", raising=False)
    rr = ReverseReranker()
    monkeypatch.setattr(sgm, "load_reranker", lambda *a, **k: rr)
    r = _r(index_dir, k=2)
    r.invoke("deploy release")
    assert rr.calls == 0
    monkeypatch.setenv("SGM_RERANK", "1")
    r.invoke("deploy release")
    assert rr.calls == 1


def test_pool_follows_brain_topk_retrieve(index_dir, monkeypatch):
    monkeypatch.setenv("BRAIN_TOPK_RETRIEVE", "1")
    docs = _r(index_dir, k=4, graph=False).invoke("deploy release checklist")
    assert len(docs) == 1
    monkeypatch.setenv("BRAIN_TOPK_RETRIEVE", "zero")
    with pytest.raises(ValueError, match="BRAIN_TOPK_RETRIEVE"):
        _r(index_dir, k=4).invoke("deploy release")


def test_explicit_fetch_k_beats_env(index_dir, monkeypatch):
    monkeypatch.setenv("BRAIN_TOPK_RETRIEVE", "1")
    docs = _r(index_dir, k=4, graph=False, fetch_k=3).invoke("deploy release")
    assert len(docs) == 3


def test_sgm_ppr_env_runs_the_pagerank_lane(index_dir, monkeypatch):
    seen = {}

    def fake_lane(meta, sims, base_order):
        seen["lane"] = True
        return [Path(meta[i]["path"]).stem.lower() for i in base_order][::-1]

    def fake_fuse(full, meta, stems, topn):
        seen["fuse_topn"] = topn
        return list(full)[::-1][:topn]

    monkeypatch.setattr(sgm, "ppr_lane", fake_lane)
    monkeypatch.setattr(sgm, "fuse_ppr", fake_fuse)
    monkeypatch.setenv("SGM_PPR", "1")
    _r(index_dir, k=2, graph=False).invoke("deploy release")
    assert seen == {"lane": True, "fuse_topn": 2}
    seen.clear()
    monkeypatch.delenv("SGM_PPR")
    _r(index_dir, k=2, graph=False).invoke("deploy release")
    assert seen == {}
