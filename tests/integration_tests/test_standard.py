"""LangChain's own standard test suite for retrievers (langchain-tests), run against a
tiny on-disk index with fake models, so it needs no network and no GPU."""
import tempfile
from functools import cached_property
from pathlib import Path

from langchain_tests.integration_tests import RetrieversIntegrationTests

from langchain_sqlite_graph_memory import SQLiteGraphMemoryRetriever
from tests.fakes import FakeEncoder, FakeReranker, build_vault_and_index

_TMP = Path(tempfile.mkdtemp(prefix="lc-sgm-"))
_INDEX = str(build_vault_and_index(_TMP))


class TestSQLiteGraphMemoryRetrieverStandard(RetrieversIntegrationTests):
    @property
    def retriever_constructor(self) -> type[SQLiteGraphMemoryRetriever]:
        return SQLiteGraphMemoryRetriever

    @property
    def retriever_constructor_params(self) -> dict:
        return {"index_dir": _INDEX, "encoder": FakeEncoder(), "reranker": FakeReranker()}

    @property
    def retriever_query_example(self) -> str:
        return "deploy release checklist"
