from collections import defaultdict
import json

with open("supreme-raw/supreme.conversations.txt", "r") as f:
    cur_convo_id = 0
    reply_to = None
    utts = []
    speakers = {}
    advocates = defaultdict(dict)
    for line in f:
        (
            case_id,
            utt_id,
            after_previous,
            speaker,
            is_justice,
            justice_vote,
            presentation_side,
            utterance,
        ) = line.strip().split(" +++$+++ ")
        if after_previous == "FALSE":
            cur_convo_id += 1
            reply_to = None
        speaker_type = "J" if is_justice == "JUSTICE" else "A"
        side = 1 if presentation_side == "PETITIONER" else 0
        utts.append(
            {
                "id": utt_id,
                "conversation_id": str(cur_convo_id),
                "text": utterance,
                "meta": {"case_id": case_id, "speaker_type": speaker_type, "side": side},
                "reply_to": reply_to,
                "speaker": speaker,
            }
        )
        reply_to = utt_id

        role = "justice" if speaker_type == "J" else "advocate"
        speakers[speaker] = {"name": speaker, "type": speaker_type, "role": role}
        if speaker_type == "A":
            advocates[case_id][speaker] = {"side": side, "role": presentation_side}

with open("supreme-raw/supreme.outcome.txt", "r") as f:
    outcomes = {}
    for line in f:
        case_id, winning_side = line.strip().split(" +++$+++ ")
        winning_side = 1 if winning_side == "PETITIONER" else 0
        outcomes[case_id] = winning_side

with open("supreme-raw/supreme.votes.txt", "r") as f:
    votes = {}
    for line in f:
        votes_case = {}
        toks = line.strip().split(" +++$+++ ")
        case_id = toks[0]
        for tok in toks[1:]:
            justice, side = tok.split("::")
            justice = "JUSTICE " + justice
            votes_case[justice] = {"PETITIONER": 1, "RESPONDENT": 0, "NA": None}[side]
        votes[case_id] = votes_case

# make convos
convos = defaultdict(dict)
for utt in utts:
    convo_id = utt["conversation_id"]
    case_id = utt["meta"]["case_id"]
    convos[convo_id]["case_id"] = case_id
    convos[convo_id]["win_side"] = outcomes[case_id] if case_id in outcomes else None
    convos[convo_id]["votes_side"] = votes[case_id] if case_id in votes else None
    convos[convo_id]["advocates"] = advocates[case_id]

with open("supreme/conversations.json", "w") as f:
    json.dump(convos, f)

with open("supreme/speakers.json", "w") as f:
    json.dump(speakers, f)

with open("supreme/utterances.jsonl", "w") as f:
    for utt in utts:
        f.write("{}\n".format(json.dumps(utt)))

with open("supreme/corpus.json", "w") as f:
    json.dump({"name": "supreme"}, f)

with open("supreme/index.json", "w") as f:
    f.write("""
{"utterances-index": {"case_id": "<class 'str'>",
"speaker_type": "<class 'str'>",
"side": "<class 'NoneType'>"},
"speakers-index": {"name": "<class 'str'>",
"type": "<class 'str'>", "role": "<class 'str'>"},
"conversations-index": {"case_id": "<class 'str'>",
"advocates": "<class 'dict'>"},
"overall-index": {"name": "<class 'str'>",
"year": "<class 'int'>"}, "version": 1}
""")
