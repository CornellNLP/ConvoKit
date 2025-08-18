from collections import namedtuple

try:
    from datasets import Dataset

    DATASETS_AVAILABLE = True
except (ModuleNotFoundError, ImportError) as e:
    raise ModuleNotFoundError(
        "datasets is not currently installed. Run 'pip install convokit[llm]' if you would like to use the utterance simulator functionality."
    ) from e

ContextTuple = namedtuple(
    "ContextTuple", ["context", "current_utterance", "future_context", "conversation_id"]
)


def default_prompt_fn(
    context_tuple,
    tokenizer,
    stage="train",
):
    """
    Default prompt function to convert context tuple into prompt.
    """
    prompt = []
    context = context_tuple.context
    prompt += [
        {
            "role": "system",
            "content": "You are a member of the subreddit r/changemyview. Your task is to output the next message in the conversation.",
        }
    ]
    prompt += [{"role": "user", "content": utt.text} for utt in context]

    if stage in ["train", "val"]:
        next_utt = context_tuple.future_context[0]
        prompt += [{"role": "assistant", "content": next_utt.text}]
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False)

    else:
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

    return prompt


def contexts_to_dataset(
    contexts,
    tokenizer,
    prompt_fn,
    stage="train",
):
    """
    Converts Iterator of contexts into dataset.
    """
    data = {"text": [], "utt_ids": []}
    for context in contexts:
        text = prompt_fn(
            context_tuple=context,
            tokenizer=tokenizer,
            stage=stage,
        )
        utt_id = context.current_utterance.id

        data["text"].append(text)
        data["utt_ids"].append(utt_id)

    dataset = Dataset.from_dict(data)
    return dataset
