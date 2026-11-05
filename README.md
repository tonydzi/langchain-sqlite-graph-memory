# langchain-sqlite-graph-memory

A LangChain retriever over your own markdown notes, with the links between them. It runs
[sqlite-graph-memory](https://github.com/tonydzi/sqlite-graph-memory): vector search finds
the entry points, the `[[wikilinks]]` you already wrote are the graph, and a cross-encoder
reranks what the 1-hop expansion pulled in. Everything runs on your machine: no server, no
graph database, no LLM call at retrieval time.

```bash
pip install "langchain-sqlite-graph-memory[lite]"   # ONNX models, starts in seconds
sgm-index /path/to/your/notes                      # builds ./index once
```

```python
from langchain_sqlite_graph_memory import SQLiteGraphMemoryRetriever

retriever = SQLiteGraphMemoryRetriever(index_dir="./index", k=8)

for doc in retriever.invoke("why did we drop the graph database?"):
    print(doc.metadata["rank"], doc.metadata["title"], doc.metadata["via_graph"])
```

Each result is a `Document`. `page_content` holds the matched chunk of the note. `metadata`
holds `path`, `title`, `date`, `score` (cross-encoder), `rank`, and `via_graph`. `via_graph`
is `True` when the note came in only through a wikilink from a top vector hit, so you can
tell associative recall from direct recall.

## Options

| Argument | Default | What it does |
|---|---|---|
| `index_dir` | `$BRAIN_INDEX_DIR`, else `./index` | Folder with the `sgm-index` output |
| `k` | 12 | Documents returned. `retriever.invoke(q, k=3)` overrides it per call |
| `graph` | `"auto"` | `True` always follows wikilinks, `False` never does. `"auto"` follows them except for name or tool lookups, where the hop measured as noise |
| `fetch_k` | 60 | Unique notes taken from vector search before expansion and rerank |
| `encoder`, `reranker` | SGM models | Pass your own preloaded models, or fakes in tests |

## In an agent

```python
from langchain_core.tools import create_retriever_tool

recall = create_retriever_tool(
    retriever, "recall_notes",
    "Search my notes and the notes they link to. Use for anything I wrote down before.",
)
```

## Scope, stated plainly

- **This adapter calls the core library and adds nothing to retrieval.** It calls the same
  functions `sgm-ask` calls (`query_sims`, `retrieve_candidates`, `expand_1hop`,
  `rerank_candidates`). A test pins this: the adapter's order must equal a direct call of
  those functions. On a real vault of 29,474 notes (105,383 chunks), the adapter returned the same 12 notes in the
  same order as `sgm-ask --graph`.
- **The index is two files on disk**, an embedding matrix (`.npy`) and note metadata
  (`.pkl`), written by `sgm-index`. Wikilinks are read from the notes at query time.
  SQLite stores the core library's session memory and A/B logs. The retriever needs only
  the two index files.
- Embeddings: `intfloat/multilingual-e5-base`. Reranker:
  `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`. Both work for many languages.
- Not implemented yet: a `VectorStore` (adding documents through LangChain) and a LangGraph
  long-term memory store. Indexing goes through `sgm-index`.

## Tests

```bash
pip install -e ".[test]"
pytest
```

The suite includes LangChain's standard retriever tests (`langchain-tests`). It runs with
fake models on a four-note vault: no download, no GPU, a few seconds.

## License

MIT
