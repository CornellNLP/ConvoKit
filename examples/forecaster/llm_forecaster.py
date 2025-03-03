import sys, os
sys.path.insert(0, "/home/sqt2/CALMpaper/ConvoKit")
import argparse
def parseargs(arglist):
    """ This is my version of an argument parser.
        Parameters:
            arglist: the command line list of args
        Returns:
            the result of parsing with the system parser
    """
    parser = argparse.ArgumentParser()
    for onearg in arglist:
        if len(onearg) == 5:
            parser.add_argument(onearg[0], onearg[1], help=onearg[2], type=onearg[3], default=onearg[4])
        else:
            parser.add_argument(onearg[0], onearg[1], help=onearg[2], type=onearg[3])
    args = parser.parse_args()
    return args


from convokit import LLMCGAModel
from convokit import download, Corpus, Utterance, Forecaster
import json, os
from functools import partial
from utils import *
args = parseargs([['-model', '--model_name_or_path', 'model_name', str],
                     ['-corpus_name', '--corpus_name', 'corpus_name', str],
                     ['-finetune', '--do_finetune', 'finetune_before_evaluate', bool],
                     ['-threshold', '--do_tune_threshold', 'tune_threshold_before_evaluate', bool],
                     ['-lr', '--learning_rate', 'learning_rate', float, 1e-4],
                     ['-bs', '--per_device_batch_size', 'number_of_samples_on_each_GPU', int, 1],
                     ['-graddient_step', '--gradient_accumulation_steps', 'accumulation_steps_for_finetuning', int, 32],
                     ['-epoch', '--num_train_epochs', 'num_train_epochs', int, 2],
                     ['-output', '--output_dir', 'output_directory', str],
                     ])
print(args)
# CPU mode (noting that it will be slower)
DEVICE = "cuda"

# corpus_name = "cga-wikiconv"
# corpus_name = "cga-cmv-legacy"
# corpus_name = "cga-cmv-large"
label_metadata = "has_removed_comment" if 'cmv' in args.corpus_name else 'conversation_has_personal_attack'
YOUR_MODEL_DIRECTORY = "/reef/sqt2/DevTest-newConvokit/YOUR_DATA_DIRECTORY"
YOUR_DATA_DIRECTORY = "/reef/sqt2/DevTest-newConvokit/YOUR_DATA_DIRECTORY"
YOUR_SAVING_DIRECTORY = "/reef/sqt2/DevTest-newConvokit/YOUR_SAVING_DIRECTORY"
if args.corpus_name == "cga-wikiconv":
    corpus = Corpus(filename=download("conversations-gone-awry-corpus", data_dir=YOUR_DATA_DIRECTORY))
else:
    if args.corpus_name == "cga-cmv-legacy":
        corpus = Corpus(filename=download("conversations-gone-awry-cmv-corpus", data_dir=YOUR_DATA_DIRECTORY))
    elif args.corpus_name == "cga-cmv-large":
        cmv_dir = "/reef/lyt5/cga_cmv_no_deleted"
        corpus = Corpus(cmv_dir)

    all_new_utterances = []
    for convo in corpus.iter_conversations():
        last_utterance_id = convo.get_chronological_utterance_list()[-1].id
        random_speaker = convo.get_chronological_speaker_list()[0]
        convo_id = convo.id
        dummy_id = convo.id + "_dummy_reply"
        new_utterance = Utterance(id=dummy_id,
                                speaker=random_speaker,
                                conversation_id=convo_id,
                                reply_to=last_utterance_id,
                                text="This#is#a#dummy#reply.",
                                timestamp=1672516520,
                                )
        all_new_utterances.append(new_utterance)
    corpus.add_utterances(all_new_utterances)

def generic_fit_selector(context_tuple, split):
    """
    We use this generic function for both training and validation data.
    In both cases, its job is to select only those contexts for which the
    FUTURE context is empty. This is in accordance with how CRAFT Model was
    originally trained on CGA-CMV, taking the last context from each
    conversation ("last" defined as being up to and including the chronologically
    last utterance as recorded in the corpus)
    """
    convo = context_tuple.current_utterance.get_conversation()
    convo_length = len(convo.get_chronological_utterance_list())

    matches_split = (context_tuple.current_utterance.get_conversation().meta["split"] == split)
    is_end = (len(context_tuple.context) == convo_length-1)
    return (matches_split and is_end)

def transform_selector(context_tuple, split):
    """
    For transform we only need to check that the conversation is in the test split
    """
    convo = context_tuple.current_utterance.get_conversation()
    convo_length = len(convo.get_chronological_utterance_list())

    matches_split = (context_tuple.current_utterance.get_conversation().meta["split"] == split)
    is_end = (len(context_tuple.context) == convo_length)

    return (matches_split and not is_end)


config_dict = {
    "output_dir": args.output_dir,
    "gradient_accumulation_steps": args.gradient_accumulation_steps,
    "per_device_batch_size": args.per_device_batch_size,
    "num_train_epochs": args.num_train_epochs,
    "learning_rate": args.learning_rate,
    "random_seed": 1,
    "do_finetune": args.do_finetune,
    "do_tune_threshold": args.do_tune_threshold,
    "device": DEVICE
}
gemma_model = LLMCGAModel(args.model_name_or_path, config=config_dict)
gemma_forecaster = Forecaster(gemma_model, label_metadata)
gemma_forecaster.fit(corpus,
                     context_selector=partial(generic_fit_selector, split="train"),
                     val_context_selector=partial(transform_selector, split="val"))

corpus = gemma_forecaster.transform(corpus, partial(transform_selector, split="test"))
_, metrics = gemma_forecaster.summarize(corpus, lambda c: c.meta['split'] == "test")
result_file = os.path.join(config_dict['output_dir'], "test_result.json")
with open(result_file, 'w') as outfile:
    json_object = json.dumps(metrics, indent=4)
    outfile.write(json_object)