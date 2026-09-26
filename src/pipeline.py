"""
End-to-end runner.

  python3 pipeline.py --data-dir dataset --output-dir output

What it does, in order:
  1. Load train_source1/2/3.tsv + train_ground_truth.tsv
  2. Blocking: generate candidate pairs for the TRAIN Source-1 entities
  3. Build similarity features for every (S1, candidate) pair
  4. Label each pair using the ground truth, split by S1 entity into
     train/validation, fit the classifier
  5. Pick the probability threshold that maximizes macro F_0.5 on validation
     (this also prints your expected leaderboard score - read it!)
  6. Repeat blocking + features on the TEST Source-1/2/3 files, score with
     the trained model, apply the chosen threshold
  7. Write output/matching_results.tsv and output/candidate_pairs.tsv
"""
import argparse
import gc
import os
import sys
import time
from contextlib import contextmanager

import pandas as pd


@contextmanager
def timed(label: str):
    start = time.time()
    yield
    print(f"  [{label} took {time.time() - start:.1f}s]")

from blocking import generate_candidates
from features import build_record_lookup, build_feature_table
from modeling import attach_labels, split_s1_ids, train_model, score_pairs, \
    predictions_at_threshold, choose_best_threshold
from evaluate import macro_f0_5

REQUIRED_COLS = ["entity_id", "business_name", "business_address", "country"]


def load_source(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    missing = set(REQUIRED_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing expected columns: {missing}")
    return df


def load_ground_truth(path: str) -> dict:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    gt = {}
    for row in df.itertuples(index=False):
        ids = {x for x in row.matched_entity_ids.split(",") if x}
        gt[row.source1_entity_id] = ids
    return gt


def write_id_list_file(path: str, mapping: dict, all_s1_ids, id_col: str, list_col: str):
    """mapping: {source1_entity_id: iterable(ids)}. Every id in all_s1_ids gets
    exactly one row, in sorted order, even if mapping has nothing for it."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{id_col}\t{list_col}\n")
        for s1 in sorted(all_s1_ids):
            ids = sorted(set(mapping.get(s1, ())))
            f.write(f"{s1}\t{','.join(ids)}\n")


def run(data_dir: str, output_dir: str, candidate_cap_note=True):
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    print("== Loading data ==")
    train_s1 = load_source(os.path.join(train_dir, "train_source1.tsv"))
    train_s2 = load_source(os.path.join(train_dir, "train_source2.tsv"))
    train_s3 = load_source(os.path.join(train_dir, "train_source3.tsv"))
    ground_truth = load_ground_truth(os.path.join(train_dir, "train_ground_truth.tsv"))
    print(f"  train: {len(train_s1)} S1, {len(train_s2)} S2, {len(train_s3)} S3 records")

    print("== Blocking (train) ==")
    with timed("blocking (train)"):
        train_candidates = generate_candidates(train_s1, train_s2, train_s3)
    n_pairs = train_candidates["candidate_entity_ids"].apply(len).sum()
    print(f"  {n_pairs} candidate pairs across {len(train_candidates)} S1 entities "
          f"({n_pairs / max(len(train_candidates), 1):.1f} candidates/entity on average)")

    print("== Features (train) ==")
    with timed("feature building (train)"):
        s1_lookup = build_record_lookup(train_s1)
        s23_lookup = build_record_lookup(pd.concat([train_s2, train_s3], ignore_index=True))
        # train_s1/s2/s3 are now fully redundant with the lookups above (same
        # info, in a lighter normalized form) and nothing below reads them
        # again. Free them HERE rather than waiting for run() to return - this
        # one function stays alive through model training and the (larger)
        # test phase, so anything still referenced by it stays resident in
        # memory the whole time regardless of whether it's still needed.
        del train_s1, train_s2, train_s3
        gc.collect()
        feature_df = build_feature_table(train_candidates, s1_lookup, s23_lookup, n_workers=1)
        # Likewise: once the feature table is built, the lookups themselves
        # are done being useful for the train phase.
        del s1_lookup, s23_lookup
        gc.collect()
    feature_df = attach_labels(feature_df, ground_truth)
    print(f"  {len(feature_df)} labeled pairs, {feature_df['label'].sum()} positive")

    print("== Train / validation split (by S1 entity) ==")
    all_train_s1_ids = set(train_candidates["source1_entity_id"])
    train_ids, val_ids = split_s1_ids(all_train_s1_ids, test_size=0.2, seed=42)
    train_rows = feature_df[feature_df["source1_entity_id"].isin(train_ids)]
    val_rows = feature_df[feature_df["source1_entity_id"].isin(val_ids)]
    # Boolean-indexing a DataFrame COPIES the matching rows - it doesn't hand
    # back a view. So train_rows + val_rows together already hold roughly one
    # full extra copy of feature_df's data; keeping feature_df itself around
    # too (nothing below reads it directly again) would make that two extra
    # copies of one of the largest tables in this whole pipeline for nothing.
    del feature_df
    gc.collect()

    print("== Training model ==")
    with timed("model training"):
        model = train_model(train_rows)
    del train_rows
    gc.collect()

    print("== Choosing threshold on validation ==")
    val_scored = score_pairs(model, val_rows)
    del val_rows  # score_pairs already extracted what it needs into val_scored
    gc.collect()
    gt_val = {k: v for k, v in ground_truth.items() if k in val_ids}
    best_t, best_f = choose_best_threshold(val_scored, gt_val, val_ids)
    print(f"  best threshold = {best_t}  ->  validation macro F_0.5 = {best_f:.4f}")

    # Sanity check: what's the ceiling given our blocking recall? If a
    # Source-1 entity's true matches never even made it into the candidate
    # set, no model threshold can recover them.
    #
    # IMPORTANT: build a plain dict lookup ONCE instead of repeatedly
    # filtering train_candidates with a boolean mask inside the loop. The
    # original `train_candidates.loc[train_candidates.source1_entity_id == s1, ...]`
    # pattern re-scans the ENTIRE (2M+ row) DataFrame on every single
    # iteration - with ~400K+ validation entities that's on the order of
    # hundreds of billions of comparisons, and it would never finish in any
    # practical amount of time (this is an O(n^2) algorithmic bug, not a
    # memory one - it wouldn't crash, it would just look "stuck" forever).
    # A dict built once from the same data turns each lookup into O(1).
    candidate_lookup = dict(zip(train_candidates["source1_entity_id"],
                                 train_candidates["candidate_entity_ids"]))
    recoverable = 0
    total_with_matches = 0
    for s1, true_ids in gt_val.items():
        if not true_ids:
            continue
        total_with_matches += 1
        cands = set(candidate_lookup.get(s1, ()))
        if true_ids <= cands:
            recoverable += 1
    if total_with_matches:
        print(f"  blocking recall ceiling on validation: {recoverable}/{total_with_matches} "
              f"non-singleton entities have ALL true matches inside the candidate set "
              f"({100 * recoverable / total_with_matches:.1f}%)")

    # train_candidates (43M+ rows on the full dataset) isn't read again after
    # the recall-ceiling loop above - free it (and the lookup dict built from
    # it) before loading the (larger) test set on top of it.
    del train_candidates, candidate_lookup, val_scored, gt_val
    gc.collect()

    print("== Loading test data ==")
    test_s1 = load_source(os.path.join(test_dir, "test_source1.tsv"))
    test_s2 = load_source(os.path.join(test_dir, "test_source2.tsv"))
    test_s3 = load_source(os.path.join(test_dir, "test_source3.tsv"))
    print(f"  test: {len(test_s1)} S1, {len(test_s2)} S2, {len(test_s3)} S3 records")
    # Grab this now, while test_s1 is cheap to read, so test_s1 doesn't have
    # to stay alive just for this one line further down.
    all_test_s1_ids = set(test_s1["entity_id"])

    print("== Blocking (test) ==")
    with timed("blocking (test)"):
        test_candidates = generate_candidates(test_s1, test_s2, test_s3)

    print("== Features + scoring (test) ==")
    with timed("feature building + scoring (test)"):
        t_s1_lookup = build_record_lookup(test_s1)
        t_s23_lookup = build_record_lookup(pd.concat([test_s2, test_s3], ignore_index=True))
        # Raw test dataframes are now redundant with t_s1_lookup/t_s23_lookup
        # (already captured all_test_s1_ids above) - free them immediately.
        del test_s1, test_s2, test_s3
        gc.collect()
        test_feature_df = build_feature_table(test_candidates, t_s1_lookup, t_s23_lookup, n_workers=1)
        del t_s1_lookup, t_s23_lookup
        gc.collect()
        test_scored = score_pairs(model, test_feature_df) if len(test_feature_df) else test_feature_df
        # score_pairs only needs the id columns + the new score column, so it
        # no longer copies the (larger) numeric feature columns - see
        # modeling.py. test_feature_df itself is done being useful now.
        del test_feature_df
        gc.collect()
    test_predictions = predictions_at_threshold(test_scored, best_t) if len(test_scored) else {}
    del test_scored
    gc.collect()

    candidate_map = {
        row.source1_entity_id: row.candidate_entity_ids
        for row in test_candidates.itertuples(index=False)
    }
    del test_candidates
    gc.collect()

    print("== Writing outputs ==")
    write_id_list_file(os.path.join(output_dir, "matching_results.tsv"),
                        test_predictions, all_test_s1_ids,
                        "source1_entity_id", "matched_entity_ids")
    write_id_list_file(os.path.join(output_dir, "candidate_pairs.tsv"),
                        candidate_map, all_test_s1_ids,
                        "source1_entity_id", "candidate_entity_ids")
    print(f"  wrote {output_dir}/matching_results.tsv and {output_dir}/candidate_pairs.tsv")
    print("\nNow run utils/validate_submission.py against these before uploading.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="dataset",
                         help="Folder containing train/ and test/ subfolders")
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    run(args.data_dir, args.output_dir)
