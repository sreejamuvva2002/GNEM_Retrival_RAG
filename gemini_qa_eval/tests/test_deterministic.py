import math

from gqe.evaluation.deterministic import (
    citation_rate,
    entity_scores,
    find_entities,
    is_refusal,
    numeric_agreement,
    rouge_l,
    score_answer,
    token_f1,
)

ENTITIES = ["Hitachi Astemo Americas Inc.", "Honda Development & Manufacturing", "Kia Georgia", "Novelis Inc."]


def test_token_f1_identical_text_is_one():
    assert token_f1("battery pack supplier", "battery pack supplier")["f1"] == 1.0


def test_token_f1_disjoint_text_is_zero():
    assert token_f1("alpha beta", "gamma delta")["f1"] == 0.0


def test_rouge_l_rewards_ordered_overlap():
    ordered = rouge_l("a b c d", "a b c d")
    shuffled = rouge_l("d c b a", "a b c d")
    assert ordered == 1.0
    assert shuffled < ordered


def test_entity_matching_ignores_legal_suffixes():
    found = find_entities("We buy from Hitachi Astemo Americas and from Novelis.", ENTITIES)
    assert "hitachi astemo americas" in found
    assert "novelis" in found


def test_entity_scores_flag_invented_companies():
    scores = entity_scores("Kia Georgia and Novelis Inc. supply parts.", "Kia Georgia supplies parts.", ENTITIES)
    assert scores["entity_recall"] == 1.0
    assert scores["entities_hallucinated"] == 1
    assert scores["entity_precision"] == 0.5


def test_entity_scores_are_nan_without_reference_entities():
    scores = entity_scores("Kia Georgia.", "The market grew.", ENTITIES)
    assert math.isnan(scores["entity_recall"])


def test_numeric_agreement_catches_wrong_counts():
    assert numeric_agreement("There are 18 companies.", "There are 18 companies.")["numeric_recall"] == 1.0
    assert numeric_agreement("There are 12 companies.", "There are 18 companies.")["numeric_recall"] == 0.0


def test_refusal_detection():
    assert is_refusal("The records do not contain that information.")
    assert not is_refusal("There are 18 Tier 1/2 companies in Georgia.")


def test_citation_rate_counts_trailing_markers():
    assert citation_rate("Alpha supplies seats. [C1]") == 1.0
    assert citation_rate("Alpha supplies seats to the plant.") == 0.0


def test_score_answer_returns_every_headline_metric():
    scores = score_answer("Kia Georgia builds EVs. [C1]", "Kia Georgia builds EVs in West Point.", ENTITIES)
    for key in ("token_f1", "rouge_l", "entity_recall", "numeric_recall", "citation_rate", "is_refusal"):
        assert key in scores
