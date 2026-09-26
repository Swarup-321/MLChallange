"""
The matching model: given the similarity-feature row for one (S1, candidate)
pair, output a probability that the pair is a true match. We use a gradient
boosted tree (scikit-learn's HistGradientBoostingClassifier) rather than a
neural net / LLM: it works well on this kind of "a dozen numeric similarity
scores" tabular data, trains in seconds to minutes, needs no GPU, and is a
plain MIT-licensed scikit-learn model - trivially inside the size/license
rule.  Swap in LightGBM/XGBoost later if you want a bit more accuracy; the
rest of the pipeline does not care which library produced the model.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

from features import FEATURE_COLUMNS
from evaluate import macro_f0_5


def attach_labels(feature_df: pd.DataFrame, ground_truth: dict) -> pd.DataFrame:
    """ground_truth: {source1_entity_id: set(true_matched_ids)}

    Labels every (source1_entity_id, candidate_entity_id) row as 1 if it's a
    true match, else 0.

    IMPORTANT: this is done as a vectorized merge, NOT
    `feature_df.apply(is_match, axis=1)`. Row-wise `.apply(axis=1)` makes
    pandas construct one full boxed Series object PER ROW just to hand it to
    a Python function - on tens of millions of rows that is tens of millions
    of Series allocations, which is catastrophically slow (it can run for
    ages without finishing) and can also blow up at the end when pandas
    tries to infer a dtype for the resulting object array (this is what
    produced the `Unable to allocate ... data type complex128` crash - a
    symptom of that broken type inference, not a sign you need more RAM).

    Instead: turn ground_truth into a small DataFrame of ONLY the true
    (source1_entity_id, candidate_entity_id) pairs with label=1, and left-
    merge it onto feature_df. A merge is a single vectorized hash-join
    implemented in C - no per-row Python calls, no per-row Series objects -
    and its cost scales with the (much smaller) number of true matches, not
    with a Python-level loop over all 40M+ candidate pairs.
    """
    pos_pairs = [(s1, cid) for s1, ids in ground_truth.items() for cid in ids]
    if pos_pairs:
        pos_df = pd.DataFrame(pos_pairs, columns=["source1_entity_id", "candidate_entity_id"])
        pos_df["label"] = np.int8(1)
    else:
        pos_df = pd.DataFrame({
            "source1_entity_id": pd.Series(dtype=feature_df["source1_entity_id"].dtype),
            "candidate_entity_id": pd.Series(dtype=feature_df["candidate_entity_id"].dtype),
            "label": pd.Series(dtype="int8"),
        })

    merged = feature_df.merge(
        pos_df, on=["source1_entity_id", "candidate_entity_id"], how="left"
    )
    # Rows with no matching positive pair get NaN from the left merge -
    # those are the true negatives (candidate, but not a real match).
    merged["label"] = merged["label"].fillna(0).astype("int8")
    return merged


def split_s1_ids(all_s1_ids, test_size=0.2, seed=42):
    """Split at the Source-1-entity level (not the row/pair level) so every
    pair belonging to one S1 entity stays on the same side of the split -
    otherwise the same entity leaking into both train and validation would
    make validation look artificially better than it really is."""
    ids = sorted(all_s1_ids)
    train_ids, val_ids = train_test_split(ids, test_size=test_size, random_state=seed)
    return set(train_ids), set(val_ids)


def train_model(train_df: pd.DataFrame) -> HistGradientBoostingClassifier:
    # to_numpy(dtype=...) converts directly to float32 in one pass. Plain
    # `.values` on a DataFrame with mixed float32/int32 columns silently
    # upcasts the WHOLE array to float64 (numpy's mixed-dtype promotion
    # rule) - quietly doubling memory right at the two most memory-heavy
    # remaining steps (training and scoring), after features.py already
    # went to the trouble of downcasting these columns to float32/int32.
    X = train_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = train_df["label"].values
    model = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.08,
        max_iter=300,
        l2_regularization=1.0,
        class_weight="balanced",  # true matches are rare among all candidate pairs
        random_state=42,
    )
    model.fit(X, y)
    return model


def score_pairs(model, feature_df: pd.DataFrame) -> pd.DataFrame:
    """Returns just source1_entity_id, candidate_entity_id, score - which is
    all predictions_at_threshold()/macro_f0_5() ever read from this. The old
    version did feature_df.copy() and kept every FEATURE_COLUMNS column too,
    which on the full dataset means copying the largest part of the table
    (8 numeric columns) just to tack on one more - here we only copy the two
    (much smaller) id columns."""
    scores = model.predict_proba(feature_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32))[:, 1]
    out = feature_df[["source1_entity_id", "candidate_entity_id"]].copy()
    out["score"] = scores
    return out


def predictions_at_threshold(scored_df: pd.DataFrame, threshold: float) -> dict:
    keep = scored_df[scored_df["score"] >= threshold]
    preds = {}
    for s1, group in keep.groupby("source1_entity_id"):
        preds[s1] = set(group["candidate_entity_id"])
    return preds


def choose_best_threshold(scored_val_df: pd.DataFrame, ground_truth_val: dict, all_val_s1_ids,
                           thresholds=None) -> tuple:
    """Try a grid of thresholds, return (best_threshold, best_f0.5)."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.30, 0.96, 0.02), 2)
    best_t, best_score = 0.5, -1.0
    for t in thresholds:
        preds = predictions_at_threshold(scored_val_df, t)
        score = macro_f0_5(preds, ground_truth_val, all_val_s1_ids)
        if score > best_score:
            best_t, best_score = t, score
    return best_t, best_score
