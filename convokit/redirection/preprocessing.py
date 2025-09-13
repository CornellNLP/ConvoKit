try:
    from datasets import Dataset

    DATASETS_AVAILABLE = True
except (ModuleNotFoundError, ImportError) as e:
    raise ModuleNotFoundError(
        "datasets is not currently installed. Run 'pip install convokit[llm]' if you would like to use the redirection preprocessing functionality."
    ) from e


def default_speaker_prefixes(roles):
    """
    Gemerates speaker prefixes for speaker roles.

    :param roles: Roles to generate prefixes for.

    :return: List of speaker prefixes
    """
    number_of_roles = len(roles)
    speakers = ["Speaker " + chr(65 + (i % 26)) + ": " for i in range(number_of_roles)]
    return speakers


def format_conversations(convos):
    """
    Format the conversations used for fine-tuning and inference.

    :param convos: List of conversations to format

    :return: Formatted conversations
    """
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


def get_chunk_dataset(tokenizer, convos, max_tokens=512, overlap_tokens=50):
    """
    Generate a chunked dataset for training given max sequence length
    and overlap length.

    :param tokenizer: Tokenizer of model
    :param convos: List of conversations to generate dataset
    :param max_tokens: Max sequence length
    :param overlap_tokens: Number of overlap tokens for chunks

    :return: Chunk dataset
    """
    chunks = []
    for convo in convos:
        convo_chunks = chunk_text_with_overlap(
            tokenizer,
            convo,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )
        chunks += convo_chunks

    data_dict = {"text": chunks}
    dataset = Dataset.from_dict(data_dict)
    return dataset


def chunk_text_with_overlap(tokenizer, text, max_tokens=512, overlap_tokens=50):
    """
    Split conversation into chunks for training.

    :param tokenizer: Tokenizer of model
    :param text: Text to chunk
    :param max_tokens: Max sequence length
    :param overlap_tokens: Number of overlap tokens for chunks

    :return: Chunk of texts
    """
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
