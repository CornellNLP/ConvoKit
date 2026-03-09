# Written with LLM support to identify as many text variations as possible to enable reliable text-matching (no unique ID available in csv).

import json, pandas as pd, re, html, unicodedata
from collections import defaultdict, Counter

JSON_IN = "conversations_text_attr.json"  # located in ./merged-data (created in mergeJSON.csv)
CSV_IN = "<path>unintended-offense-tweets/scored_replies/averaged_scores_with_context.csv"  # sourced from https://github.com/IDEA-NTHU-Taiwan/unintended-offense-tweets. Also located in ./original-data
JSON_OUT = "conversations_text_attr_offense.json"  # located in ./merged-data

# ====== Regex ======
url_re = re.compile(r"https?://\S+")
mention_rt_re = re.compile(r"^(rt\s+@[\w_]+:\s*)", flags=re.IGNORECASE)
mention_re = re.compile(r"@[\w_]+")
hashtag_re = re.compile(r"#")
punct_re = re.compile(r"[^\w\s]")
ws_re = re.compile(r"\s+")
sep_re = re.compile(r"\s*\[\s*sep\s*\]\s*", flags=re.IGNORECASE)
long_repeat = re.compile(r"(.)\1{2,}")  # collapse runs of >=3 to 2

# emoji-name & tiny stopword sets to stabilize keys
EMOJI_NAME_STOP = {
    "face",
    "with",
    "grinning",
    "smiling",
    "crying",
    "loudly",
    "tears",
    "joy",
    "sweat",
    "rolling",
    "eyes",
    "red",
    "blue",
    "green",
    "heart",
    "button",
    "symbol",
    "sparkles",
    "star",
    "skull",
    "fire",
    "hundred",
    "thumbs",
    "up",
    "down",
    "folded",
    "hands",
    "laughing",
    "sob",
    "zzz",
    "smirk",
    "kiss",
    "wink",
    "relieved",
    "astonished",
    "weary",
    "tired",
    "yawning",
    "pleading",
    "scream",
    "grimacing",
    "pensive",
}
STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "to",
    "of",
    "for",
    "in",
    "on",
    "at",
    "is",
    "it",
    "this",
    "that",
    "you",
    "your",
    "i",
    "me",
    "my",
    "we",
    "our",
    "they",
    "them",
    "their",
    "be",
    "am",
    "are",
    "was",
    "were",
    "as",
    "so",
}


def normalize_for_match(s: str) -> str:
    if s is None:
        return ""
    s = html.unescape(str(s))
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u200d", "")
    s = s.encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = url_re.sub(" ", s)
    s = mention_rt_re.sub("", s)
    s = mention_re.sub(" ", s)
    s = hashtag_re.sub("", s)
    s = punct_re.sub(" ", s)
    s = ws_re.sub(" ", s).strip()
    s = long_repeat.sub(r"\1\1", s)  # coooool -> coool
    return s


def toks(s: str):
    return [t for t in s.split() if t]


def prefix_key(s: str, n=12) -> str:
    tt = toks(s)
    return " ".join(tt[:n])


def suffix_key(s: str, n=12) -> str:
    tt = toks(s)
    return " ".join(tt[-n:]) if tt else ""


def scrub_key(s: str, n=18) -> str:
    out = []
    for t in toks(s):
        if t in STOPWORDS or t in EMOJI_NAME_STOP:
            continue
        out.append(t)
        if len(out) >= n:
            break
    return " ".join(out)


def token_set(s: str):
    return {t for t in toks(s) if t not in STOPWORDS and t not in EMOJI_NAME_STOP}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / float(len(a | b))


def char_ngrams(s: str, n=4):
    s = f" {s} "  # mild padding helps edges
    return {s[i : i + n] for i in range(max(0, len(s) - n + 1))}


# ====== load JSON conversations ======
with open(JSON_IN, "r", encoding="utf-8") as f:
    A = json.load(f)

buckets = ["context_tweets", "target_tweet", "follow-up_tweet", "cue_tweets"]

rows = []
for ci, conv in enumerate(A):
    for b in buckets:
        ts = conv.get(b) or []
        if isinstance(ts, dict):
            ts = [ts]
            conv[b] = ts
        for ti, tw in enumerate(ts):
            text = (tw.get("full_text") or tw.get("text") or "").strip()
            n = normalize_for_match(text)
            rows.append(
                {
                    "ci": ci,
                    "bucket": b,
                    "ti": ti,
                    "norm_text": n,
                    "pref12": prefix_key(n, 12),
                    "suf12": suffix_key(n, 12),
                    "pref20": prefix_key(n, 20),
                    "suf20": suffix_key(n, 20),
                    "scrub": scrub_key(n, 18),
                }
            )

df_json = pd.DataFrame(rows)

# Split JSON rows into all + target-only views
json_all_maps = {
    k: df_json[["ci", "bucket", "ti", k]].dropna().drop_duplicates()
    for k in ["norm_text", "pref12", "suf12", "pref20", "suf20", "scrub"]
}

df_json_target = df_json[df_json["bucket"] == "target_tweet"].copy()
json_target_maps = {
    k: df_json_target[["ci", "bucket", "ti", k]].dropna().drop_duplicates()
    for k in ["norm_text", "pref12", "suf12", "pref20", "suf20", "scrub"]
}

# Inverted token index for fuzzy tier (JSON universe)
json_tokens = [token_set(s) for s in df_json["norm_text"].tolist()]
df_counts = Counter(t for ts in json_tokens for t in ts)
RARE_MAX_DF = 800  # a bit wider than before for recall
inv_tok = defaultdict(set)
for idx, ts in enumerate(json_tokens):
    for t in ts:
        if df_counts[t] <= RARE_MAX_DF:
            inv_tok[t].add(idx)

# Character 4-gram index (JSON)
json_4g = [char_ngrams(s, 4) for s in df_json["norm_text"].tolist()]
gram_counts = Counter(g for gs in json_4g for g in gs)
RARE_GRAM_MAX_DF = 1200
inv_gram = defaultdict(set)
for idx, gs in enumerate(json_4g):
    for g in gs:
        if gram_counts[g] <= RARE_GRAM_MAX_DF:
            inv_gram[g].add(idx)

# ====== load CSV ======
csv = pd.read_csv(CSV_IN, encoding="utf-8-sig")
assert {"tweet", "offensiveness", "confidence"}.issubset(csv.columns), csv.columns
csv = csv.reset_index(drop=True).rename_axis("rid").reset_index()


# Segment helpers
def seg_after_first(s: str) -> str:
    return sep_re.split(s, 1)[-1].strip()


def seg_before_first(s: str) -> str:
    parts = sep_re.split(s, 1)
    return parts[0].strip() if parts else s.strip()


def seg_after_last(s: str) -> str:
    parts = sep_re.split(s)
    return parts[-1].strip() if parts else s.strip()


raw = csv["tweet"].astype(str)
segments = pd.concat(
    [
        pd.DataFrame(
            {"rid": csv["rid"], "seg_name": "after_first", "text": raw.map(seg_after_first)}
        ),
        pd.DataFrame(
            {"rid": csv["rid"], "seg_name": "after_last", "text": raw.map(seg_after_last)}
        ),
        pd.DataFrame(
            {"rid": csv["rid"], "seg_name": "before_first", "text": raw.map(seg_before_first)}
        ),
    ],
    ignore_index=True,
)

segments["offensiveness"] = segments["rid"].map(csv.set_index("rid")["offensiveness"])
segments["confidence"] = segments["rid"].map(csv.set_index("rid")["confidence"])

segments["norm_text"] = segments["text"].map(normalize_for_match)
segments["pref12"] = segments["norm_text"].map(lambda s: prefix_key(s, 12))
segments["suf12"] = segments["norm_text"].map(lambda s: suffix_key(s, 12))
segments["pref20"] = segments["norm_text"].map(lambda s: prefix_key(s, 20))
segments["suf20"] = segments["norm_text"].map(lambda s: suffix_key(s, 20))
segments["scrub"] = segments["norm_text"].map(lambda s: scrub_key(s, 18))
segments["tokset"] = segments["norm_text"].map(token_set)
segments["len_tokens"] = segments["norm_text"].map(lambda s: len(toks(s)))
segments["grams4"] = segments["norm_text"].map(lambda s: char_ngrams(s, 4))

# ====== multi-tier matching ======
# Tier A: exact+fallbacks against all buckets (as before)
priority_order = [
    ("norm_text", 100),
    ("pref12", 80),
    ("suf12", 70),
    ("pref20", 60),
    ("suf20", 50),
    ("scrub", 40),
]


def run_tier(seg_df, maps, base_bias=0):
    cand = []
    for key, base_score in priority_order:
        hit = seg_df.merge(maps[key], on=key, how="left")
        hit = hit[pd.notna(hit["ci"])].copy()
        if hit.empty:
            continue
        hit["bucket_bonus"] = hit["bucket"].eq("target_tweet").astype(int) * 5
        seg_bonus_map = {"after_first": 4, "after_last": 2, "before_first": 1}
        hit["seg_bonus"] = hit["seg_name"].map(seg_bonus_map).fillna(0)
        hit["tier_score"] = base_bias + base_score + hit["bucket_bonus"] + hit["seg_bonus"]
        hit = hit.sort_values(["rid", "tier_score"], ascending=[True, False]).drop_duplicates(
            subset=["rid", "ci", "bucket", "ti"], keep="first"
        )
        cand.append(hit[["rid", "ci", "bucket", "ti", "tier_score"]])
    if cand:
        return pd.concat(cand, ignore_index=True)
    else:
        return pd.DataFrame(columns=["rid", "ci", "bucket", "ti", "tier_score"])


# Tier A1: all buckets
cand = run_tier(segments, json_all_maps, base_bias=0)


# Tier A2: target-only maps for remaining (slightly looser)
def remaining_rids(c):
    return (
        set(segments["rid"]) - set(c["rid"])
        if c is not None and not c.empty
        else set(segments["rid"])
    )


rem = remaining_rids(cand)
if rem:
    cand2 = run_tier(
        segments[segments["rid"].isin(rem)], json_target_maps, base_bias=5
    )  # small bias to prefer these
    cand = pd.concat([cand, cand2], ignore_index=True) if not cand2.empty else cand

# ====== Tier B: short-text special (≤6 tokens): prefer exact subset matches on tokens ======
rem = remaining_rids(cand)
if rem:
    short_seg = segments[(segments["rid"].isin(rem)) & (segments["len_tokens"] <= 6)].copy()
    st_rows = []
    for _, row in short_seg.iterrows():
        qset = row["tokset"]
        if not qset:
            continue
        idxs = set()
        for t in qset:
            idxs |= inv_tok.get(t, set())
        best = None
        best_score = -1
        for idx in idxs:
            jrow = df_json.iloc[idx]
            jset = token_set(jrow["norm_text"])
            # require qset ⊆ jset or high overlap for very short queries
            subset_ok = qset.issubset(jset) or jaccard(qset, jset) >= 0.9
            if not subset_ok:
                continue
            score = 35 + (5 if jrow["bucket"] == "target_tweet" else 0) + len(qset) * 0.2
            if score > best_score:
                best_score = score
                best = (row["rid"], int(jrow["ci"]), jrow["bucket"], int(jrow["ti"]), score)
        if best:
            st_rows.append(best)
    if st_rows:
        st_df = pd.DataFrame(st_rows, columns=["rid", "ci", "bucket", "ti", "tier_score"])
        cand = pd.concat([cand, st_df], ignore_index=True) if not cand.empty else st_df

# ====== Tier C: character 4-gram fuzzy for remaining ======
rem = remaining_rids(cand)
if rem:
    seg_left = segments[segments["rid"].isin(rem)].copy()
    seg_left["seg_rank"] = (
        seg_left["seg_name"].map({"after_first": 3, "after_last": 2, "before_first": 1}).fillna(0)
    )

    FUZZY_JACCARD_TOK = 0.78  # a hair looser
    FUZZY_JACCARD_CHAR = 0.80
    MIN_SUBSTR_LEN = 22
    FUZZY_SCORE = 30

    fuzzy_rows = []
    for rid, grp in seg_left.groupby("rid"):
        grp = grp.sort_values("seg_rank", ascending=False)
        picked = grp.iloc[0]
        q_text = picked["norm_text"]
        q_tokens = picked["tokset"]
        q_grams = picked["grams4"]

        cand_idxs = set()
        for t in q_tokens:
            cand_idxs |= inv_tok.get(t, set())
        for g in q_grams:
            cand_idxs |= inv_gram.get(g, set())
        if not cand_idxs:
            continue

        best = None
        best_score = -1
        for idx in cand_idxs:
            jrow = df_json.iloc[idx]
            c_text = jrow["norm_text"]
            c_tokens = token_set(c_text)
            c_grams = json_4g[idx]

            # substring fast-path
            substr_ok = (len(q_text) >= MIN_SUBSTR_LEN and q_text in c_text) or (
                len(c_text) >= MIN_SUBSTR_LEN and c_text in q_text
            )

            tok_sim = jaccard(q_tokens, c_tokens)
            char_sim = jaccard(q_grams, c_grams)

            if substr_ok or (tok_sim >= FUZZY_JACCARD_TOK and char_sim >= FUZZY_JACCARD_CHAR):
                score = FUZZY_SCORE + (3 if substr_ok else 0) + tok_sim + char_sim
                if jrow["bucket"] == "target_tweet":
                    score += 5
                if score > best_score:
                    best_score = score
                    best = (rid, int(jrow["ci"]), jrow["bucket"], int(jrow["ti"]), score)
        if best:
            fuzzy_rows.append(best)

    if fuzzy_rows:
        fuzzy_df = pd.DataFrame(fuzzy_rows, columns=["rid", "ci", "bucket", "ti", "tier_score"])
        cand = pd.concat([cand, fuzzy_df], ignore_index=True) if not cand.empty else fuzzy_df


# ====== choose exactly one conversation per CSV row ======
def choose_one(df):
    if df is None or df.empty:
        return pd.DataFrame(columns=["rid", "ci"])
    t = df.copy()
    t["is_target"] = t["bucket"].eq("target_tweet").astype(int)
    chosen = t.sort_values(
        ["rid", "tier_score", "is_target", "ti"], ascending=[True, False, False, True]
    ).drop_duplicates(subset=["rid"], keep="first")
    return chosen[["rid", "ci"]]


chosen_ci = choose_one(cand)
csv_with_ci = csv.merge(chosen_ci, on="rid", how="left")
print(f"Identified conversation for {csv_with_ci['ci'].notna().sum()} / {len(csv)} CSV rows")

# ====== apply scores ONLY to target_tweet ======
apply_count = 0
for _, r in csv_with_ci.iterrows():
    if pd.isna(r["ci"]):
        continue
    ci = int(r["ci"])
    off = r["offensiveness"]
    conf = r["confidence"]

    tgt = A[ci].get("target_tweet") or []
    tt_list = [tgt] if isinstance(tgt, dict) else tgt
    if not tt_list:
        continue

    tt = tt_list[0]  # apply to first target tweet
    if pd.notna(off):
        tt["offensiveness"] = float(off)
    if pd.notna(conf):
        tt["confidence"] = float(conf)
    apply_count += 1

print(f"Applied scores to {apply_count} target_tweets")

with open(JSON_OUT, "w", encoding="utf-8") as out:
    json.dump(A, out, ensure_ascii=False, indent=2)
print(f"Wrote: {JSON_OUT}")
