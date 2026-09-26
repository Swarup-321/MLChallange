# Business Entity Resolution — Pipeline

## What this does
1. **Blocking (`src/blocking.py`)** — for each Source-1 business, finds a short list of
   plausible Source-2/3 candidates using shared name words + shared postal code, instead
   of comparing against every record.
2. **Features (`src/features.py`)** — turns each (S1, candidate) pair into ~8 similarity
   numbers (word overlap, character similarity, postal/country match, etc).
3. **Model (`src/modeling.py`)** — a gradient-boosted tree (scikit-learn
   `HistGradientBoostingClassifier`) learns from `train_ground_truth.tsv` which pairs are
   real matches, and we pick the score cutoff that maximizes the challenge's own metric
   (macro F_0.5) on a held-out slice of the training data.
4. **`src/pipeline.py`** ties it together: trains + validates on the train set, then
   applies the same blocking → features → model to the test set and writes both required
   output files.

## Setup
```bash
cd business_entity_resolution
pip install -r requirements.txt
```

## Run
Point `--data-dir` at the folder that contains your `train/` and `test/` subfolders
(the `dataset/` folder from the challenge):

```bash
cd src
python3 pipeline.py --data-dir /path/to/dataset --output-dir /path/to/output
```

This prints, in order: candidate-set size, blocking **recall ceiling** (the % of
validation entities whose true matches actually made it into the candidate set — this is
the hard upper bound on your score, so check it's high before worrying about the model),
the chosen threshold, and your estimated macro F_0.5. Then it writes:
- `output/matching_results.tsv`
- `output/candidate_pairs.tsv`

Then validate before uploading:
```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir /path/to/dataset/test
```

## Known limitations / what to improve next
- **Blocking** only uses name-token overlap and postal code. If two true-match records
  share literally no words and no postal code (e.g. `"Sharma Sons"` vs `"S&S Traders"`,
  a DBA/trade name with no overlap), they'll never become candidates — check your
  blocking recall ceiling printout; if it's not high, add another blocking key
  (e.g. phonetic encoding like Soundex/Metaphone on the first name token, or n-gram
  blocking for very short names).
- **String similarity** uses `difflib.SequenceMatcher` (Python stdlib, no install needed).
  Swapping in `rapidfuzz` (`pip install rapidfuzz`) gives the same kind of score, faster,
  if you're running on the full multi-million-row dataset and it matters.
- **Threshold** is a single global cutoff. If precision/recall needs differ a lot by
  country, consider a per-country threshold instead — same `choose_best_threshold`
  function, just call it once per country slice.
- The classifier trains fine as long as your candidate set contains a healthy mix of
  true matches AND true non-matches (negatives). If your blocking is *too* precise
  (never proposes a wrong candidate), the model has nothing to learn from — that's a
  good problem to have, but double check on a bigger sample than a handful of rows.
