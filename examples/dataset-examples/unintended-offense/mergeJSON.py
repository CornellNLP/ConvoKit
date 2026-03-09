# Merging the Two JSON Files

import json, re, html, unicodedata

# from collections import defaultdict

# regex for text normalization (the correct regex syntax for each type was searched and found using LLM support)
url_re = re.compile(r"https?://\S+")
mention_rt_re = re.compile(r"^(rt\s+@[\w_]+:\s*)", flags=re.IGNORECASE)
mention_re = re.compile(r"@[\w_]+")
hashtag_symbol_re = re.compile(r"#")
punct_re = re.compile(r"[^\w\s]")
whitespace_re = re.compile(r"\s+")


# normalize using regex above
def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = html.unescape(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = url_re.sub(" ", s)
    s = mention_rt_re.sub("", s)
    s = mention_re.sub(" ", s)
    s = hashtag_symbol_re.sub("", s)
    s = punct_re.sub(" ", s)
    s = whitespace_re.sub(" ", s).strip()
    return s


# bring in both JSON files (sourced from https://github.com/IDEA-NTHU-Taiwan/unintended-offense-tweets. Also located in ./original-data)
with open(r"<path>/conversations_text_only.json") as f:
    A = json.load(f)
with open(r"<path>/conversations_with_attr.json") as f:
    B = json.load(f)

b_by_convo = {}
for thread in B:  # for each thread in the json file with attributes
    if not isinstance(thread, list):  # if not already in list
        thread = [thread]  # add to list
    convo_id = None  # set convo id to none to fall back on as fail safe
    mapping = {}  # create dict
    for tweet in thread:
        convo_id = convo_id or tweet.get("conversation_id")  # get convo id
        txt = tweet.get("full_text") or tweet.get("text") or ""  # get Tweet text
        key = (
            normalize_text(txt),
            str(tweet.get("author_id", "")),
        )  # call normalize_text() to apply normalization to text
        mapping.setdefault(key, tweet)  # set if not already in dict
    if convo_id is None:
        continue
    b_by_convo[convo_id] = mapping  # save mapping

fields_to_copy = [
    "id",
    "created_at",
    "source",
    "public_metrics",
    "reply_settings",
]  # want to copy Tweet-level metadata


def copy_fields(dst: dict, src: dict, fields: list, into_key: str = "_meta"):
    meta = dst.setdefault(into_key, {})  # if _meta doesn't exist yet, create as empty dict
    for key in fields:  # for every field...
        if key in src:  # if exists in dict
            meta[key] = src[key]  # copy into meta dict


type_tweets = ["context_tweets", "target_tweet", "follow-up_tweet", "cue_tweets"]

# initialize matches to zero
no_convo_match = 0
no_tweet_match = 0
matched = 0

for convo in A:  # for each convo in the json file with types, but no attributes
    convo_id = convo.get("conversation_id")  # get convo id from JSON
    if convo_id not in b_by_convo:  # if there is no matching convo id in the other JSON file
        no_convo_match += 1  # increment the no match metric
        for typetweet in type_tweets:  # for each type of Tweet (context, target, etc)...
            for tweet in convo.get(typetweet, []) or []:  # for each tweet, get the type of tweet
                tweet["_merge_note"] = (
                    "no_conversation_match_in_B"  # add a note to inform that there was no match
                )
        continue

    index = b_by_convo[convo_id]  # look up prev mapping and assign to index
    for typetweet in type_tweets:
        tweets = convo.get(typetweet, [])  # get the type of tweet
        if not isinstance(tweets, list):  # make sure Tweet is list
            tweets = [tweets]
            convo[typetweet] = tweets
        for tweet in tweets:
            txt = tweet.get("full_text") or tweet.get("text") or ""  # extract text as txt
            key = (
                normalize_text(txt),
                str(tweet.get("author_id", "")),
            )  # set key to normalized text and author
            src = index.get(key)  # find source Tweet
            if src:  # if match...
                copy_fields(tweet, src, fields_to_copy, into_key="_meta")  # copy meta data in
                tweet["_merge_note"] = (
                    "matched_on_text_author"  # merge note should indicate that it was matched on author
                )
                matched += 1  # increment matched
            else:
                tweet["_merge_note"] = "no_tweet_match_in_B"  # otherwise indicate not a match
                no_tweet_match += 1  # and increment no match count

print(
    f"Matched: {matched}, conversations without B: {no_convo_match}, tweets without match: {no_tweet_match}"
)  # summary

# export merged json (located in ./merged-data)
with open("conversations_text_attr.json", "w", encoding="utf-8") as out:
    json.dump(A, out, ensure_ascii=False, indent=2)
