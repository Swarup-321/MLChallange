"""
For every (Source-1, candidate) pair, turn the two raw records into a row of
numbers a classifier can learn from: "how similar are these names/addresses,
on a few different scales". We deliberately use several different similarity
measures (word-overlap AND character-level AND length-based) because typos,
word reordering, and abbreviations each break a different measure - having
all of them lets the model figure out which signals to trust.

PERFORMANCE NOTE: with a huge dataset, this is the step that costs the most
time, because it runs a real computation for every candidate pair (tens of
millions of them, easily). Four things make it fast and light enough to run
on the full data:
  1. rapidfuzz instead of Python's built-in difflib for character similarity
     - same idea, but implemented in C, roughly 10-50x faster per call.
     Falls back to difflib automatically if rapidfuzz isn't installed, so
     this still runs with zero extra setup, just slower.
  2. Splitting the candidate pairs into chunks and computing features for
     each chunk on a separate CPU core at once (multiprocessing), instead of
     one core doing all of it. On an 8-core machine that's roughly an 8x
     speedup for free; on a 32-core cloud box, roughly 32x.
  3. MEMORY (per-worker data): each worker is only ever given the small
     slice of the lookup its OWN chunk actually needs (a couple thousand S1
     entities plus their candidates - tens of thousands of records), not the
     entire multi-million-record lookup. Sending the whole lookup to every
     worker (via Pool's `initializer`/`initargs`, or trying to rely on OS
     fork) means N workers effectively hold N copies of it in RAM - and on
     Windows (multiprocessing's "spawn" start method, no fork available at
     all), each worker also has to unpickle its own full copy from scratch,
     which is exactly what runs a machine out of memory before any real work
     happens. Slicing per chunk keeps this small and correct on every OS the
     same way, with no platform-specific branching. We also only store the
     normalized strings per record (not precomputed token sets) - token sets
     are cheap to rebuild from an already-normalized string on demand, but
     expensive to keep resident for every one of potentially tens of
     millions of records at once.
  4. MEMORY (result assembly): each chunk is turned into a small, TYPED
     pandas DataFrame (a few numeric columns, one chunk_size rows) as soon as
     it's computed, and chunks are combined with pd.concat. We deliberately
     do NOT accumulate one giant Python list of per-pair dicts and hand that
     to pd.DataFrame(...) at the end - pandas builds a list-of-dicts
     DataFrame by first materializing a full object-dtype (N_rows x N_cols)
     array (see pandas.core.internals.construction._list_of_dict_to_arrays),
     which has to exist AT THE SAME TIME as the (already memory-heavy) list
     of dicts it was built from. On ~43.5M rows that intermediate object
     array alone is several GiB, on top of the dict list, and is what causes
     `numpy._core._exceptions._ArrayMemoryError` at the final
     `pd.DataFrame(results)` call. Building small typed DataFrames per chunk
     (float32 for the continuous similarity scores, int32 for the
     count/flag columns) and concatenating them avoids ever holding an
     untyped, boxed copy of the full result table.
"""
import sys
from multiprocessing import Pool, cpu_count

import pandas as pd

from text_normalize import normalize_name, normalize_text, extract_postal_code

try:
    from rapidfuzz.fuzz import ratio as _rapidfuzz_ratio
    def _char_ratio(a: str, b: str) -> float:
        if not a and not b:
            return 1.0
        return _rapidfuzz_ratio(a, b) / 100.0
except ImportError:  # falls back to stdlib, no install required, just slower
    from difflib import SequenceMatcher
    def _char_ratio(a: str, b: str) -> float:
        if not a and not b:
            return 1.0
        return SequenceMatcher(None, a, b).ratio()


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# Each record is stored as a small tuple, not a dict: (name_norm, addr_norm,
# postal, country). A tuple has far less per-object overhead than a dict with
# string keys, and we build ~12.5M of these on the full test set, so the
# saving is real. Token sets are NOT precomputed/stored here - see
# `_name_tokens_of` / `_addr_tokens_of` below, which rebuild them cheaply
# from the normalized string only when a pair actually needs them.
_NAME_NORM, _ADDR_NORM, _POSTAL, _COUNTRY = range(4)


def build_record_lookup(df: pd.DataFrame) -> dict:
    """entity_id -> (name_norm, addr_norm, postal, country) tuple, so we only
    normalize each record once no matter how many candidate pairs it's in."""
    lookup = {}
    for row in df.itertuples(index=False):
        # sys.intern: country has only a handful of distinct values (US,
        # India, France, ...) across millions of rows, so interning lets all
        # rows sharing a value point at ONE string object instead of millions
        # of duplicate copies. Cheap to do, meaningful at this scale.
        country = sys.intern(row.country or "")
        lookup[row.entity_id] = (
            normalize_name(row.business_name),
            normalize_text(row.business_address),
            extract_postal_code(row.business_address),
            country,
        )
    return lookup


def _name_tokens_of(name_norm: str) -> set:
    return set(name_norm.split())


def _addr_tokens_of(addr_norm: str) -> set:
    return set(addr_norm.split())


def pair_features(s1_id: str, cand_id: str, lookup: dict) -> dict:
    a, b = lookup[s1_id], lookup[cand_id]
    a_name_tokens = _name_tokens_of(a[_NAME_NORM])
    b_name_tokens = _name_tokens_of(b[_NAME_NORM])
    a_addr_tokens = _addr_tokens_of(a[_ADDR_NORM])
    b_addr_tokens = _addr_tokens_of(b[_ADDR_NORM])
    return {
        "source1_entity_id": s1_id,
        "candidate_entity_id": cand_id,
        "name_jaccard": _jaccard(a_name_tokens, b_name_tokens),
        "name_char_ratio": _char_ratio(a[_NAME_NORM], b[_NAME_NORM]),
        "addr_jaccard": _jaccard(a_addr_tokens, b_addr_tokens),
        "addr_char_ratio": _char_ratio(a[_ADDR_NORM], b[_ADDR_NORM]),
        "common_name_tokens": len(a_name_tokens & b_name_tokens),
        "name_len_diff": abs(len(a[_NAME_NORM]) - len(b[_NAME_NORM])),
        "postal_match": int(bool(a[_POSTAL]) and a[_POSTAL] == b[_POSTAL]),
        "country_match": int(a[_COUNTRY] == b[_COUNTRY]),
    }


FEATURE_COLUMNS = [
    "name_jaccard", "name_char_ratio", "addr_jaccard", "addr_char_ratio",
    "common_name_tokens", "name_len_diff", "postal_match", "country_match",
]

# dtype each feature column is downcast to when a chunk's DataFrame is built.
# Keeping these as float32/int32 (instead of pandas' default float64/int64,
# and FAR instead of leaving them as boxed Python objects) is what keeps the
# final concatenated table's memory footprint small - on 43.5M rows the
# difference between float64 and float32 alone is ~1.4 GB across these 4
# columns.
_FLOAT_COLS = ("name_jaccard", "name_char_ratio", "addr_jaccard", "addr_char_ratio")
_INT_COLS = ("common_name_tokens", "name_len_diff", "postal_match", "country_match")


def _process_chunk(task) -> pd.DataFrame:
    """task: (chunk_rows, sub_lookup).
    chunk_rows: list of (source1_entity_id, [candidate_entity_ids]).
    sub_lookup: ONLY the records this chunk's rows actually reference - not
    the full multi-million-record lookup. This is what keeps the data sent
    to (and held by) each worker small, on every OS.

    Returns a small, typed per-chunk DataFrame rather than a list of dicts -
    see the module docstring (point 4) for why that matters for the final
    memory footprint once all chunks are combined."""
    chunk_rows, sub_lookup = task

    s1_col, cand_col = [], []
    feat_cols = {c: [] for c in FEATURE_COLUMNS}

    for s1_id, cand_ids in chunk_rows:
        for cand_id in cand_ids:
            feats = pair_features(s1_id, cand_id, sub_lookup)
            s1_col.append(s1_id)
            cand_col.append(cand_id)
            for c in FEATURE_COLUMNS:
                feat_cols[c].append(feats[c])

    data = {"source1_entity_id": s1_col, "candidate_entity_id": cand_col}
    data.update(feat_cols)
    df = pd.DataFrame(data)

    # Downcast right away, inside the (small) per-chunk DataFrame, so the
    # eventual pd.concat never has to upcast/copy a mix of dtypes.
    for c in _FLOAT_COLS:
        df[c] = df[c].astype("float32")
    for c in _INT_COLS:
        df[c] = df[c].astype("int32")

    return df


def _iter_tasks(chunks, s1_lookup, s23_lookup):
    """Yield (chunk_rows, sub_lookup) lazily, one chunk at a time, so we never
    hold more than one chunk's worth of sliced lookup data in memory in the
    parent while handing work out - and each worker only ever receives the
    slice for the chunk it's currently running, not the whole dataset.

    Deliberately takes s1_lookup and s23_lookup SEPARATELY rather than one
    merged dict: S1 ids and S2-/S3- ids never overlap, so a merged dict adds
    an entire extra dict's worth of memory (one more hash table sized for
    every record in the dataset) for zero benefit over just checking both
    dicts here."""
    for chunk_rows in chunks:
        needed_ids = set()
        for s1_id, cand_ids in chunk_rows:
            needed_ids.add(s1_id)
            needed_ids.update(cand_ids)
        sub_lookup = {}
        for eid in needed_ids:
            rec = s1_lookup.get(eid)
            if rec is None:
                rec = s23_lookup.get(eid)
            if rec is not None:
                sub_lookup[eid] = rec
        yield chunk_rows, sub_lookup


def build_feature_table(candidate_pairs: pd.DataFrame, s1_lookup: dict, s23_lookup: dict,
                         n_workers: int = None, chunk_size: int = 2000) -> pd.DataFrame:
    """
    candidate_pairs: columns source1_entity_id, candidate_entity_ids (list)
    Returns one row per (s1, candidate) pair with FEATURE_COLUMNS filled in.

    n_workers: how many CPU cores to use. Defaults to all of them. Set to 1
    to disable multiprocessing entirely (e.g. for debugging, or on a
    platform where forking processes is flaky, like some notebooks).

    Builds one small typed DataFrame per chunk and pd.concat's them, instead
    of collecting a single Python list of ~40M+ per-pair dicts and handing
    that whole list to pd.DataFrame(...) at the end. The list-of-dicts path
    is what was causing
    `numpy._core._exceptions._ArrayMemoryError: Unable to allocate 3.25 GiB
    for an array with shape (43575385, 10) and data type object` - pandas
    has to build a full untyped (object-dtype) array from the dict list
    before it can split it into typed columns, and that array has to exist
    alongside the dict list it came from. Concatenating small typed
    per-chunk frames never creates that giant boxed intermediate.
    """
    all_rows = list(zip(candidate_pairs["source1_entity_id"], candidate_pairs["candidate_entity_ids"]))
    if not all_rows:
        empty = pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id"] + FEATURE_COLUMNS)
        for c in _FLOAT_COLS:
            empty[c] = empty[c].astype("float32")
        for c in _INT_COLS:
            empty[c] = empty[c].astype("int32")
        return empty

    # Leave one core free by default so the machine (VS Code, browser, OS UI)
    # stays responsive while this runs - pass n_workers explicitly to override.
    n_workers = n_workers or max(1, min(cpu_count() - 1, 32))
    chunks = [all_rows[i:i + chunk_size] for i in range(0, len(all_rows), chunk_size)]

    if n_workers <= 1 or len(chunks) <= 1:
        chunk_dfs = [_process_chunk((chunk_rows, sub_lookup))
                     for chunk_rows, sub_lookup in _iter_tasks(chunks, s1_lookup, s23_lookup)]
    else:
        # No initializer/initargs, no full-lookup broadcast: each task
        # carries only its own small slice, built lazily by _iter_tasks so
        # we're not materializing every slice up front either. No merged
        # dict either - see _iter_tasks. imap (not imap_unordered) keeps
        # chunk_dfs in submission order, which costs nothing here and makes
        # output reproducible/debuggable; switch back to imap_unordered if
        # you want results as soon as any worker finishes instead.
        with Pool(processes=n_workers) as pool:
            chunk_dfs = list(pool.imap(
                _process_chunk, _iter_tasks(chunks, s1_lookup, s23_lookup)))

    # copy=False: concat can then just stitch the underlying per-chunk
    # column arrays together instead of deep-copying every value again.
    result = pd.concat(chunk_dfs, ignore_index=True, copy=False)
    del chunk_dfs
    return result
