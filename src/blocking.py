"""
Candidate generation ("blocking").

The idea in plain terms: instead of comparing every Source-1 business against
every Source-2/3 business (which is billions of comparisons), we only compare
two records if they share something distinctive - a meaningful word in the
name, or the same postal code. We build a lookup table (a "dictionary") that
maps each distinctive word/code to the list of entity_ids that contain it.
Then, for one Source-1 business, we just look up its words in that table and
collect whoever else shares them. That is the whole trick.

Two safety valves keep the candidate sets small (which the challenge directly
rewards) AND keep this fast at millions of records:
  1. We drop words that are too common (e.g. "restaurant", "the", "services")
     because a word shared by thousands of businesses is useless for
     narrowing anything down - AND because looking one up means touching
     every one of those thousands of ids, which is what makes blocking slow
     at scale. Keeping these buckets small is what keeps runtime reasonable,
     not just candidate-set size.
  2. We cap the number of candidates kept per Source-1 entity, ranked by how
     many blocking keys two records have in common (more shared signal =
     more likely to be a real match).

PERFORMANCE NOTE: this prints progress as it goes (every PROGRESS_EVERY
records) specifically so a multi-million-row run never looks frozen - if you
don't see a new line every so often, something is actually wrong, not just
slow.
"""
import time
from collections import defaultdict
import pandas as pd

from text_normalize import name_tokens, extract_postal_code

# Words/postal-codes that appear in more than this fraction of a country's
# records are considered too generic to block on. Kept deliberately tight -
# at millions of records, a bucket of even a few thousand ids gets touched
# by every S1 record that shares that word, which is what makes blocking
# slow; small buckets are what keeps this fast, not just accurate.
MAX_TOKEN_DOC_FREQ_RATIO = 0.003
MAX_TOKEN_DOC_FREQ_ABS = 500
MIN_TOKEN_LEN = 3  # skip 1-2 letter tokens, too generic to be useful
MAX_CANDIDATES_PER_S1 = 50  # hard cap: keeps candidate_pairs.tsv small

PROGRESS_EVERY = 200_000  # print a progress line every N records processed


def _build_index(df: pd.DataFrame, id_col: str = "entity_id"):
    """
    Returns:
      token_index:   {country: {token: set(entity_id)}}
      postal_index:  {country: {postal_code: set(entity_id)}}
    """
    token_index = defaultdict(lambda: defaultdict(set))
    postal_index = defaultdict(lambda: defaultdict(set))

    start = time.time()
    n = len(df)
    for i, row in enumerate(df.itertuples(index=False)):
        eid = getattr(row, id_col)
        country = getattr(row, "country") or ""
        toks = name_tokens(getattr(row, "business_name"))
        toks = {t for t in toks if len(t) >= MIN_TOKEN_LEN}
        for t in toks:
            token_index[country][t].add(eid)

        postal = extract_postal_code(getattr(row, "business_address"))
        if postal:
            postal_index[country][postal].add(eid)

        if (i + 1) % PROGRESS_EVERY == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate if rate > 0 else float("inf")
            print(f"    indexed {i + 1:,}/{n:,} records "
                  f"({rate:,.0f}/s, ~{eta:.0f}s left)")

    return token_index, postal_index


def _prune_frequent_tokens(token_index):
    """Remove tokens that show up in too many records within a country -
    they add candidates without adding discriminating power, and are the
    main thing that makes blocking slow at scale if left unpruned."""
    for country, table in token_index.items():
        n = sum(len(v) for v in table.values()) or 1
        cap = min(MAX_TOKEN_DOC_FREQ_ABS, max(50, int(n * MAX_TOKEN_DOC_FREQ_RATIO)))
        for tok in list(table.keys()):
            if len(table[tok]) > cap:
                del table[tok]


def _prune_frequent_postals(postal_index):
    """Same idea for postal codes - mostly guards against placeholder/blank
    codes like '00000' that would otherwise create one giant bucket."""
    for country, table in postal_index.items():
        n = sum(len(v) for v in table.values()) or 1
        cap = min(MAX_TOKEN_DOC_FREQ_ABS, max(200, int(n * MAX_TOKEN_DOC_FREQ_RATIO)))
        for code in list(table.keys()):
            if len(table[code]) > cap:
                del table[code]


def generate_candidates(source1: pd.DataFrame, source2: pd.DataFrame,
                         source3: pd.DataFrame) -> pd.DataFrame:
    """
    source1/2/3: DataFrames with columns entity_id, business_name,
                 business_address, country.

    Returns a DataFrame with columns:
      source1_entity_id, candidate_entity_ids (python list of str)
    """
    combined = pd.concat([source2, source3], ignore_index=True)

    print(f"  building index over {len(combined):,} Source-2/3 records...")
    token_index, postal_index = _build_index(combined)
    _prune_frequent_tokens(token_index)
    _prune_frequent_postals(postal_index)

    print(f"  matching {len(source1):,} Source-1 records against the index...")
    rows = []
    start = time.time()
    n = len(source1)
    for i, row in enumerate(source1.itertuples(index=False)):
        s1_id = row.entity_id
        country = row.country or ""
        s1_tokens = {t for t in name_tokens(row.business_name) if len(t) >= MIN_TOKEN_LEN}
        s1_postal = extract_postal_code(row.business_address)

        # score = number of blocking keys (tokens + postal) shared with the
        # S1 record; used only to rank candidates when there are more than
        # MAX_CANDIDATES_PER_S1 of them.
        scores = defaultdict(int)
        for t in s1_tokens:
            for cand_id in token_index.get(country, {}).get(t, ()):
                scores[cand_id] += 1
        if s1_postal:
            for cand_id in postal_index.get(country, {}).get(s1_postal, ()):
                scores[cand_id] += 2  # postal match is a stronger signal than one word

        if scores:
            ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            candidate_ids = [cid for cid, _ in ranked[:MAX_CANDIDATES_PER_S1]]
        else:
            candidate_ids = []

        rows.append({"source1_entity_id": s1_id, "candidate_entity_ids": candidate_ids})

        if (i + 1) % PROGRESS_EVERY == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate if rate > 0 else float("inf")
            print(f"    matched {i + 1:,}/{n:,} S1 records "
                  f"({rate:,.0f}/s, ~{eta:.0f}s left)")

    return pd.DataFrame(rows)
