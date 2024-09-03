try:
    import torch
except (ModuleNotFoundError, ImportError) as e:
    raise ModuleNotFoundError(
        "torch is not currently installed. Run 'pip install convokit[craft]' if you would like to use the CRAFT model."
    )

import pandas as pd
from convokit.forecaster.CRAFT.data import loadPrecomputedVoc, processContext, batchIterator
from convokit import download, warn
from convokit.convokitConfig import ConvoKitConfig
from .CRAFT.model import EncoderRNN, ContextEncoderRNN, SingleTargetClf
from .CRAFT.runners import Predictor, trainIters, evaluateDataset
from .forecasterModel import ForecasterModel
import numpy as np
import torch.nn.functional as F
from torch import optim, nn
from typing import Dict, Union
import os

# parameters baked into the model design (because the provided models were saved with these parameters);
# these cannot be changed by the user
HIDDEN_SIZE = 500
ENCODER_N_LAYERS = 2
CONTEXT_ENCODER_N_LAYERS = 2
DECODER_N_LAYERS = 2
MAX_LENGTH = 80

# config dict contains parameters that could, in theory, be adjusted without causing things to crash
DEFAULT_CONFIG = {
    "dropout": 0.1,
    "batch_size": 64,
    "clip": 50.0,
    "learning_rate": 1e-5,
    "print_every": 10,
    "finetune_epochs": 30,
    "validation_size": 0.2,
}

MODEL_FILENAME_MAP = {
    "craft-wiki-pretrained": "craft_pretrained.tar",
    "craft-wiki-finetuned": "craft_full.tar",
    "craft-cmv-pretrained": "craft_pretrained.tar",
    "craft-cmv-finetuned": "craft_full.tar",
}

DECISION_THRESHOLDS = {
    "craft-wiki-pretrained": 0.570617,
    "craft-wiki-finetuned": 0.570617,
    "craft-cmv-pretrained": 0.548580,
    "craft-cmv-finetuned": 0.548580,
}

# To understand the separation of concerns for the CRAFT files:
# CRAFT/model.py contains the pytorch modules that comprise the CRAFT neural network
# CRAFT/data.py contains utility methods for manipulating the data for it to be passed to the CRAFT model
# CRAFT/runners.py adapts the scripts for training and inference of a CRAFT model


class CRAFTModel(ForecasterModel):
    """
    A ConvoKit Forecaster-adherent reimplementation of the CRAFT conversational forecasting model from
    the paper "Trouble on the Horizon: Forecasting the Derailment of Online Conversations as they Develop"
    (Chang and Danescu-Niculescu-Mizil, 2019).

    Usage note: CRAFT is a neural network model; full end-to-end training of neural networks is considered
    outside the scope of ConvoKit, so the ConvoKit CRAFTModel must be initialized with existing weights.
    ConvoKit provides weights for the CGA-WIKI and CGA-CMV corpora. If you just want to run a fully-trained
    CRAFT model on those corpora (i.e., only transform, no fit), you can use the finetuned weights
    (craft-wiki-finetuned and craft-cmv-finetuned, respectively). If you want to take a pretrained model and
    finetune it on your own data (i.e., both fit and transform), you can use the pretrained weights
    (craft-wiki-pretrained and craft-cmv-pretrained, respectively), which provide trained versions of the
    underlying utterance and conversation encoder layers but leave the classification layers at their
    random initializations so that they can be fitted to your data.

    :param initial_weights: Specifies where to find the saved model to be loaded to initialize CRAFT. To use ConvoKit's provided models, use "craft-wiki-pretrained" for the model pretrained on Wikipedia data, or "craft-wiki-finetuned" for the model already fine-tuned on CGA-WIKI. Replace "wiki" with "cmv" for the Reddit CMV equivalents. Alternatively, if you have a custom model you want to use, you can pass in the full path to the saved PyTorch checkpoint file.
    :param vocab_index2word: File containing the mapping from vocabulary index to raw string tokens. If you are using a provided model, you MUST leave this as the default value of "auto" (other values will be ignored and overridden to "auto"). Conversely, if using a custom model, you CANNOT leave this as "auto" and you must provide a full path to the vocabulary file that you made for your custom model.
    :param vocab_word2index: File containing the mapping from raw string tokens to vocabulary index. If you are using a provided model, you MUST leave this as the default value of "auto" (other values will be ignored and overridden to "auto"). Conversely, if using a custom model, you CANNOT leave this as "auto" and you must provide a full path to the vocabulary file that you made for your custom model.
    :param decision_threshold: Output probability beyond which a forecast should be considered "positive"/"True". Highly recommended to leave this at auto, which will use published values for the provided models, or 0.5 for custom models.
    :param torch_device: "cpu" or "cuda" (for GPUs). If you have access to a GPU it is strongly recommended to set this to "cuda"; the default is "cpu" only for compatibility with non-GPU setups.
    :param config: Allows overwriting of CRAFT hyperparameters. Strongly recommended to keep this at default unless you know what you're doing!
    """

    def __init__(
        self,
        initial_weights: str,
        vocab_index2word: str = "auto",
        vocab_word2index: str = "auto",
        decision_threshold: Union[float, str] = "auto",
        torch_device: str = "cpu",
        config: dict = DEFAULT_CONFIG,
    ):
        super().__init__()

        # load the initial weights and store this as the current model
        if initial_weights in MODEL_FILENAME_MAP:
            # load ConvoKitConfig in order to look up the model storage path
            convokitconfig = ConvoKitConfig()
            download_dir = os.path.expanduser(convokitconfig.model_directory)
            # download the model and its supporting vocabulary objects
            base_path = download(initial_weights, data_dir=download_dir)
            model_path = os.path.join(base_path, MODEL_FILENAME_MAP[initial_weights])
            # load the vocab, ensuring that we use the download ones
            if vocab_index2word != "auto" or vocab_word2index != "auto":
                warn(
                    f"CRAFTModel was initialized using a ConvoKit-provided model {initial_weights} but a custom vocabulary was specified. This is an unsupported configuration; the custom vocabulary will be ignored and the model-provided vocabulary will be loaded."
                )
            self._voc = loadPrecomputedVoc(
                initial_weights,
                os.path.join(base_path, "word2index.json"),
                os.path.join(base_path, "index2word.json"),
            )
        else:
            # assume that initial_weights is a true path to a local model
            model_path = initial_weights
            # we don't know the vocab for local models, so the user must manually supply one
            if vocab_index2word == "auto" or vocab_word2index == "auto":
                raise ValueError(
                    "CRAFTModel was initialized using a path to a custom model; a custom vocabulary also must be specified for this use case ('auto' is not supported)!"
                )
            self._voc = loadPrecomputedVoc(
                os.path.basename(initial_weights), vocab_word2index, vocab_index2word
            )
        self._model = torch.load(model_path, map_location=torch.device(torch_device))

        # either take the decision threshold as given or use a predetermined one (default 0.5 if none can be found)
        if type(decision_threshold) == float:
            self._decision_threshold = decision_threshold
        else:
            if decision_threshold != "auto":
                raise TypeError("CRAFTModel: decision_threshold must be either a float or 'auto'")
            self._decision_threshold = DECISION_THRESHOLDS.get(initial_weights, 0.5)

        self._device = torch.device(torch_device)
        self._config = config

    def _context_to_craft_data(self, contexts):
        """
        Convert context utterances to a list of token-lists using the model's vocabulary object,
        maintaining the original temporal ordering
        """
        pairs = []
        for context in contexts:
            convo = context.current_utterance.get_conversation()
            label = self.labeler(convo)
            processed_context = processContext(self._voc, context, label)
            utt = processed_context[-1]["tokens"][: (MAX_LENGTH - 1)]
            context_utts = [u["tokens"][: (MAX_LENGTH - 1)] for u in processed_context]
            pairs.append((context_utts, utt, label, context.current_utterance.id))
        return pairs

    def _init_craft(self):
        """
        Initialize the CRAFT layers using the currently saved checkpoints
        (these will either be the initial_weights, or what got saved after fit())
        """
        print("Loading saved parameters...")
        encoder_sd = self._model["en"]
        context_sd = self._model["ctx"]
        try:
            attack_clf_sd = self._model["atk_clf"]
        except KeyError:
            # this happens if we're loading from a non-finetuned initial weights; the classifier layer still needs training
            attack_clf_sd = None
        embedding_sd = self._model["embedding"]
        self._voc.__dict__ = self._model["voc_dict"]

        print("Building encoders, decoder, and classifier...")
        # Initialize word embeddings
        embedding = nn.Embedding(self._voc.num_words, HIDDEN_SIZE)
        embedding.load_state_dict(embedding_sd)
        # Initialize utterance and context encoders
        encoder = EncoderRNN(HIDDEN_SIZE, embedding, ENCODER_N_LAYERS, self._config["dropout"])
        context_encoder = ContextEncoderRNN(
            HIDDEN_SIZE, CONTEXT_ENCODER_N_LAYERS, self._config["dropout"]
        )
        encoder.load_state_dict(encoder_sd)
        context_encoder.load_state_dict(context_sd)
        # Initialize classifier
        attack_clf = SingleTargetClf(HIDDEN_SIZE, self._config["dropout"])
        if attack_clf_sd is not None:
            attack_clf.load_state_dict(attack_clf_sd)
        # Use appropriate device
        encoder = encoder.to(self._device)
        context_encoder = context_encoder.to(self._device)
        attack_clf = attack_clf.to(self._device)
        print("Models built and ready to go!")

        return embedding, encoder, context_encoder, attack_clf

    def fit(self, contexts, val_contexts=None):
        """
        Fine-tune the CRAFT model, and save the best model according to validation performance.

        :param contexts: an iterator over context tuples, provided by the Forecaster framework
        :param val_contexts: an iterator over context tuples to be used only for validation. IMPORTANT: this is marked Optional only for compatibility with the generic Forecaster API; CRAFT actually REQUIRES a validation set so leaving this parameter at None will raise an error!
        """
        # convert the input contexts into CRAFT's data format
        train_pairs = self._context_to_craft_data(contexts)
        print("Processed", len(train_pairs), "context tuples for model training")
        # val_contexts is made Optional to conform to the Forecaster spec, but in reality CRAFT requires a validation set
        if val_contexts is None:
            raise ValueError("CRAFTModel requires a validation set!")
        val_pairs = self._context_to_craft_data(val_contexts)
        print("Processed", len(val_pairs), "context tuples for model validation")

        # initialize the CRAFT model with whatever weights we currently have saved
        embedding, encoder, context_encoder, attack_clf = self._init_craft()

        # Compute the number of training iterations we will need in order to achieve the number of epochs specified in the settings at the start of the notebook
        n_iter_per_epoch = len(train_pairs) // self._config["batch_size"] + int(
            len(train_pairs) % self._config["batch_size"] == 1
        )
        n_iteration = n_iter_per_epoch * self._config["finetune_epochs"]

        # Put dropout layers in train mode
        encoder.train()
        context_encoder.train()
        attack_clf.train()

        # Initialize optimizers
        print("Building optimizers...")
        encoder_optimizer = optim.Adam(encoder.parameters(), lr=self._config["learning_rate"])
        context_encoder_optimizer = optim.Adam(
            context_encoder.parameters(), lr=self._config["learning_rate"]
        )
        attack_clf_optimizer = optim.Adam(attack_clf.parameters(), lr=self._config["learning_rate"])

        # Run training iterations, validating after every epoch
        print("Starting Training!")
        print("Will train for {} iterations".format(n_iteration))
        best_model = trainIters(
            self._voc,
            train_pairs,
            val_pairs,
            encoder,
            context_encoder,
            attack_clf,
            encoder_optimizer,
            context_encoder_optimizer,
            attack_clf_optimizer,
            embedding,
            n_iteration,
            self._config["batch_size"],
            self._config["print_every"],
            n_iter_per_epoch,
            self._config["clip"],
            self._device,
            MAX_LENGTH,
            batchIterator,
        )

        # save the resulting checkpoints so we can load them later during transform
        self._model = best_model

    def transform(self, contexts, forecast_attribute_name, forecast_prob_attribute_name):
        """
        Run a fine-tuned CRAFT model on the provided data

        :param contexts: context tuples from the Forecaster framework
        :param forecast_attribute_name: Forecaster will use this to look up the table column containing your model's discretized predictions (see output specification below)
        :param forecast_prob_attribute_name: Forecaster will use this to look up the table column containing your model's raw forecast probabilities (see output specification below)

        :return: a Pandas DataFrame, with one row for each context, indexed by the ID of that context's current utterance. Contains two columns, one with raw probabilities named according to forecast_prob_attribute_name, and one with discretized (binary) forecasts named according to forecast_attribute_name
        """
        # convert the input contexts into CRAFT's data format
        test_pairs = self._context_to_craft_data(contexts)
        print("Processed", len(test_pairs), "context tuples for model evaluation")

        # initialize the CRAFT model with whatever weights we currently have saved
        embedding, encoder, context_encoder, attack_clf = self._init_craft()

        # Set dropout layers to eval mode
        encoder.eval()
        context_encoder.eval()
        attack_clf.eval()

        # Initialize the pipeline
        predictor = Predictor(encoder, context_encoder, attack_clf)

        # Run the pipeline!
        forecasts_df = evaluateDataset(
            test_pairs,
            encoder,
            context_encoder,
            predictor,
            self._voc,
            self._config["batch_size"],
            self._device,
            MAX_LENGTH,
            batchIterator,
            self._decision_threshold,
            forecast_attribute_name,
            forecast_prob_attribute_name,
        )

        return forecasts_df
