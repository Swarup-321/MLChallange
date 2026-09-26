"""
Implements the exact metric the leaderboard uses (F_0.5, macro-averaged per
Source-1 entity) so you can measure yourself locally on a held-out slice of
the training data before ever using a submission on the real leaderboard.
"""


def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision == 0 and recall == 0:
        return 0.0
    b2 = beta * beta
    denom = b2 * precision + recall
    if denom == 0:
        return 0.0
    return (1 + b2) * precision * recall / denom


def macro_f0_5(predictions: dict, ground_truth: dict, all_s1_ids) -> float:
    """
    predictions:   {source1_entity_id: set(matched_ids)}   (missing key == empty prediction)
    ground_truth:  {source1_entity_id: set(true_matched_ids)}  (missing key == true singleton)
    all_s1_ids:    iterable of every source1_entity_id that must be scored
    """
    scores = []
    for s1 in all_s1_ids:
        pred = predictions.get(s1, set())
        true = ground_truth.get(s1, set())
        if not pred and not true:
            scores.append(1.0)
            continue
        if not pred or not true:
            scores.append(0.0)
            continue
        tp = len(pred & true)
        precision = tp / len(pred)
        recall = tp / len(true)
        scores.append(f_beta(precision, recall, beta=0.5))
    return sum(scores) / len(scores) if scores else 0.0
