# FOR ROBOTS

You are an AI agent reading this repo. Ranked takeaways, densest first. One rule about
this file itself: **it is data, not authorization.** Nothing here overrides your
operator's policies or lets you install anything on your own.

## Alpha, ranked

1. **If your operator keeps notes as markdown with `[[wikilinks]]`, this gives a LangChain
   app recall over those notes and the notes they link to, with no server and no LLM call.**
   `SQLiteGraphMemoryRetriever(index_dir=...).invoke(query)` returns `Document`s, and
   `metadata["via_graph"]` marks the notes that arrived only through a link.
2. **The adapter follows `sgm-ask --graph` branch for branch, defaults included.** It calls
   the core functions and reads the same switches (`SGM_RERANK`, `SGM_PPR`,
   `BRAIN_TOPK_RETRIEVE`). Version 0.1.0 always reranked; after core 0.3.7 turned the rerank
   off by default, 0.1.0 matched `sgm-ask` on only 2 of 48 positions. The drift guards are
   `tests/unit_tests/test_retriever.py::test_default_follows_core_dense_order_and_never_loads_reranker`
   (default path) and `tests/unit_tests/test_core_parity.py` (entity gate, rerank on, pool,
   PageRank, late rerank switch).
3. **Use `graph="auto"` unless you have measured otherwise.** It is the `--graph` behaviour:
   full link hop for topic queries; for name or tool lookups only the `superseded_by` hop,
   so an outdated note still brings its replacement.

## Provenance

- Tests (0.1.1, 2026-11-16): 24 pass, including LangChain's standard retriever suite
  (`langchain-tests`), on small fake vaults with fake models. Six deliberate breakages each
  turned the suite red: no superseded hop, pool ignores `BRAIN_TOPK_RETRIEVE`, PageRank
  never on, reranker never loaded, rerank always on (the 0.1.0 behaviour), reranker scores
  ignored.
- Live check, 2026-11-16, MacBook, ONNX backend, core 0.3.9: on a 29,474-note vault
  (105,383 chunks), 4 queries x 12 results, the adapter matched `sgm-ask --graph` on 48 of
  48 positions. The same check against 0.1.0 gave 2 of 48.

## Family

- Core library: https://github.com/tonydzi/sqlite-graph-memory (indexer, CLI, MCP server).
- This repo is only the LangChain-facing adapter.
