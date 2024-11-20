from .preprocessing import default_speaker_prefixes


def default_previous_context_selector(convo):
    actual_contexts = {}
    reference_contexts = {}
    utts = [utt for utt in convo.iter_utterances()]
    roles = list({utt.meta["role"] for utt in utts})
    assert len(roles) == 2
    spk_prefixes = default_speaker_prefixes(roles)
    role_to_prefix = {roles[i]: spk_prefixes[i] for i in range(len(roles))}
    role_1 = roles[0]
    role_2 = roles[1]
    prev_spk = None
    prev_1, prev_2, cur_1, cur_2 = None, None, None, None
    for i, utt in enumerate(utts):
        utt_text = utt.text
        cur_spk = utt.meta["role"]
        if prev_spk is not None and cur_spk != prev_spk:
            if role_2 in cur_spk:
                prev_1 = cur_1
            else:
                prev_2 = cur_2

        if prev_1 and prev_2 is not None:
            if role_2 in cur_spk:
                prev = prev_1
                prev_prev = prev_2
            else:
                prev = prev_2
                prev_prev = prev_1

            prev_prev_text, prev_prev_role = prev_prev
            prev_text, prev_role = prev

            prev_prev_data = role_to_prefix[prev_prev_role] + prev_prev_text
            prev_data = role_to_prefix[prev_role] + prev_text
            cur_data = role_to_prefix[cur_spk] + utt_text

            actual_contexts[utt.id] = [prev_data, cur_data]
            reference_contexts[utt.id] = [prev_data, prev_prev_data]

        if role_1 in cur_spk:
            cur_1 = (utt_text, cur_spk)
        if role_2 in cur_spk:
            cur_2 = (utt_text, cur_spk)

        prev_spk = cur_spk

    return actual_contexts, reference_contexts


def default_future_context_selector(convo):
    future_contexts = {}
    cur_1 = None
    cur_2 = None
    utts = [utt for utt in convo.iter_utterances()]
    roles = list({utt.meta["role"] for utt in utts})
    assert len(roles) == 2
    spk_prefixes = default_speaker_prefixes(roles)
    role_to_prefix = {roles[i]: spk_prefixes[i] for i in range(len(roles))}
    role_1 = roles[0]
    role_2 = roles[1]
    n = len(utts)
    for i in range(n - 1, -1, -1):
        utt = utts[i]
        utt_text = utt.text
        cur_spk = utt.meta["role"]
        if role_2 in cur_spk:
            cur_2 = (utt_text, cur_spk)
            if cur_1 is not None:
                future_text, future_role = cur_1
                future_data = role_to_prefix[future_role] + future_text
                future_contexts[utt.id] = [future_data]
        else:
            cur_1 = (utt_text, cur_spk)
            if cur_2 is not None:
                future_text, future_role = cur_2
                future_data = role_to_prefix[future_role] + future_text
                future_contexts[utt.id] = [future_data]
    return future_contexts
