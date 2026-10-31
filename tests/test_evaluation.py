import json

from app.evaluation.metrics import chunk_matches, mean, reciprocal_rank, retrieval_metrics
from app.evaluation.run_eval import DATASET, load_dataset

R = [
    {"document_id": "a", "text": "nothing here"},
    {"document_id": "b", "text": "The Salt   Levy was abolished"},
    {"document_id": "a", "text": "the assembly meets in spring"},
]


def test_chunk_matching_is_whitespace_and_case_insensitive():
    assert chunk_matches(R[1], {"document_id": "b", "contains": ["salt levy was"]})
    assert not chunk_matches(R[1], {"document_id": "a", "contains": ["salt levy"]})


def test_metrics_values():
    labels = [{"document_id": "b", "contains": ["salt levy"]},
              {"document_id": "a", "contains": ["assembly meets"]}]
    assert retrieval_metrics(R, labels, 1) == {"recall@1": 0.0, "precision@1": 0.0}
    m3 = retrieval_metrics(R, labels, 3)
    assert m3["recall@3"] == 1.0 and abs(m3["precision@3"] - 2 / 3) < 1e-9
    assert retrieval_metrics(R, [], 3) == {}
    assert reciprocal_rank(R, labels) == 0.5
    assert reciprocal_rank(R, [{"document_id": "z", "contains": []}]) == 0.0
    assert reciprocal_rank(R, []) is None
    assert mean([1.0, 0.0]) == 0.5 and mean([]) is None


def test_dataset_is_well_formed():
    items = load_dataset(DATASET)
    assert len(items) >= 20
    ids = [i["id"] for i in items]
    assert len(ids) == len(set(ids))
    types = {i["type"] for i in items}
    for required in ["factual", "multi_document", "timeline", "comparison", "constitutional", "disputed",
                     "insufficient", "misleading_premise", "recent_no_source"]:
        assert required in types, required
    for it in items:
        assert it["expected_behavior"] in {"answer", "abstain", "decline", "answer_or_abstain"}
        assert isinstance(it.get("relevant", []), list)
        json.dumps(it)
