from datasets import Dataset


def default_speaker_prefixes(roles):
    number_of_roles = len(roles)
    speakers = ["Speaker " + chr(65 + (i % 26)) + ": " for i in range(number_of_roles)]
    return speakers


def format_conversations(convos):
    formatted_convos = []
    for convo in convos:
        utts = [utt for utt in convo.iter_utterances()]
        roles = list({utt.meta["role"] for utt in utts})
        spk_prefixes = default_speaker_prefixes(roles)
        role_to_prefix = {roles[i]: spk_prefixes[i] for i in range(len(roles))}
        formatted_utts = []
        for utt in utts:
            utt_text = role_to_prefix[utt.meta["role"]] + utt.text
            formatted_utts.append(utt_text)
        formatted_convo = "\n\n".join(formatted_utts)
        formatted_convos.append(formatted_convo)
    return formatted_convos


def get_chunk_dataset(tokenizer, convos, max_tokens=4096, overlap_tokens=100):
    chunks = []
    for convo in convos:
        convo_chunks = chunk_text_with_overlap(
            tokenizer, text, max_tokens=max_tokens, overlap_tokens=overlap_tokens
        )
        chunks += convo_chunks

    data_dict = {"text": chunks}
    dataset = Dataset.from_dict(data_dict)
    return dataset


def chunk_text_with_overlap(tokenizer, text, max_tokens=4096, overlap_tokens=100):
    tokens = tokenizer.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        overlap_end = max(start + max_tokens - overlap_tokens, start)
        chunk = tokens[start:overlap_end]
        chunks.append(tokenizer.decode(chunk))
        start = overlap_end
    return chunks
