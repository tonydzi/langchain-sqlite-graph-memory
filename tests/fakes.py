"""Tiny stand-ins for the two local models, so the suite runs in seconds with no torch,
no ONNX and no network. They implement exactly the methods the SGM pipeline calls."""
from __future__ import annotations

import hashlib
import pickle
import re
from pathlib import Path

import numpy as np

DIM = 64
_WORD = re.compile(r"[a-z]+")


def _words(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in {"query", "passage"}]


def _vec(text: str) -> np.ndarray:
    v = np.zeros(DIM, dtype="float32")
    for w in _words(text):
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1.0
    n = np.linalg.norm(v)
    return v / n if n else v


class FakeEncoder:
    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
        return np.stack([_vec(t) for t in texts])


class FakeReranker:
    """Score = shared words between query and chunk, plus a small floor so that a linked
    note with no shared words still ranks (below every note that shares a word)."""

    def predict(self, pairs):
        out = []
        for q, doc in pairs:
            out.append(len(set(_words(q)) & set(_words(doc))) + 0.01)
        return np.array(out, dtype="float32")


NOTES = {
    "alpha": "Deploy pipeline and release checklist. Before shipping see [[beta]].",
    "beta": "Rollback plan for outages and incidents.",
    "gamma": "Cooking recipes: pasta, sauce, basil.",
    "delta": "Release notes template for the deploy of a new version.",
}


def build_vault_and_index(root: Path) -> Path:
    """Write NOTES as .md files under root/vault and an SGM-format index under root/index."""
    vault, index = root / "vault", root / "index"
    vault.mkdir(parents=True, exist_ok=True)
    index.mkdir(parents=True, exist_ok=True)
    meta, rows = [], []
    for stem, body in NOTES.items():
        p = vault / ("%s.md" % stem)
        p.write_text(body, encoding="utf-8")
        meta.append({"path": str(p), "title": stem, "snippet": body, "date": "2026-11-05"})
        rows.append(_vec("passage: " + body))
    np.save(index / "_brain_e5.npy", np.stack(rows))
    (index / "_brain_e5_meta.pkl").write_bytes(pickle.dumps(meta))
    return index
