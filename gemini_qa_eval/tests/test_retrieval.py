from gqe.config import Settings
from gqe.data import Doc, load_company_docs
from gqe.evaluation.retrieval_metrics import gold_doc_ids, retrieval_scores
from gqe.retrieval import BM25, Retriever, format_context

DOCS = [
    Doc("D1", "Alpha Battery", "Alpha Battery\nRole: Battery Cell\nLocation: Savannah"),
    Doc("D2", "Beta Seating", "Beta Seating\nRole: Interiors\nLocation: Macon"),
    Doc("D3", "Gamma Thermal", "Gamma Thermal\nRole: Thermal Management\nLocation: Macon"),
]


def test_bm25_ranks_the_matching_record_first():
    ranked = BM25(DOCS).search("battery cell supplier", k=3)
    assert DOCS[ranked[0][0]].doc_id == "D1"


def test_bm25_returns_nothing_for_unrelated_query():
    assert BM25(DOCS).search("zzzz unrelated tokens", k=3) == []


def test_retriever_returns_ranked_hits():
    hits = Retriever(DOCS, cfg=Settings()).search("thermal management", k=2)
    assert hits[0].doc.doc_id == "D3"
    assert [hit.rank for hit in hits] == list(range(1, len(hits) + 1))


def test_format_context_numbers_blocks_and_respects_budget():
    hits = Retriever(DOCS, cfg=Settings()).search("Macon", k=3)
    assert format_context(hits).startswith("[C1]")
    assert len(format_context(hits, max_chars=40)) <= 80


def test_retrieval_metrics_reward_early_hits():
    early = retrieval_scores(["D1", "D9"], {"D1"})
    late = retrieval_scores(["D9", "D1"], {"D1"})
    assert early["mrr"] == 1.0
    assert late["mrr"] == 0.5
    assert early["ndcg_at_k"] > late["ndcg_at_k"]


def test_gold_docs_come_from_the_golden_answer():
    assert gold_doc_ids("Alpha Battery is the supplier.", DOCS) == {"D1"}


def test_real_workbook_retrieval_finds_battery_companies():
    docs = load_company_docs()
    hits = Retriever(docs, cfg=Settings()).search("battery cell and battery pack companies in Georgia", k=5)
    titles = " ".join(hit.doc.title for hit in hits).lower()
    assert "hitachi" in titles or "honda" in titles
