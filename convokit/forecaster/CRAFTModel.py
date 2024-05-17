try:
    import torch
except (ModuleNotFoundError, ImportError) as e:
    raise ModuleNotFoundError(
        "torch is not currently installed. Run 'pip install convokit[craft]' if you would like to use the CRAFT model."
    )

import pandas as pd
from convokit.forecaster.CRAFT.data import loadPrecomputedVoc, batchIterator
from .CRAFT.model import initialize_model, makeContextEncoderInput, Predictor
from .forecasterModel import ForecasterModel
import numpy as np
import torch.nn.functional as F
from torch import optim
from typing import Dict, Union
import os

DEFAULT_CONFIG = {
    "hidden_size": 500,
    "encoder_n_layers": 2,
    "context_encoder_n_layers": 2,
    "decoder_n_layers": 2,
    "dropout": 0.1,
    "batch_size": 64,
    "clip": 50.0,
    "learning_rate": 1e-5,
    "print_every": 10,
    "train_epochs": 30,
    "validation_size": 0.2,
    "max_length": 80
}

DECISION_THRESHOLDS = {
    "craft-wiki-pretrained": 0.570617,
    "craft-wiki-finetuned": 0.570617,
    "craft-cmv-pretrained": 0.548580,
    "craft-cmv-finetuned": 0.548580
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
    """

    def __init__(
        self,
        initial_weights: str,
        decision_threshold: Union[float, str] = "auto",
        config: dict = DEFAULT_CONFIG
    ):
        super().__init__()