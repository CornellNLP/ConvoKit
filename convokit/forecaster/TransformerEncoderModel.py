import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)

import os
import pandas as pd
from tqdm import tqdm
from .forecasterModel import ForecasterModel
from .TransformerForecasterConfig import TransformerForecasterConfig
from itertools import tee

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEFAULT_CONFIG = TransformerForecasterConfig(
    output_dir="TransformerEncoderModel",
    gradient_accumulation_steps=1,
    per_device_batch_size=4,
    num_train_epochs=1,
    learning_rate=6.7e-6,
    random_seed=1,
    context_mode="normal",
    device="cuda",
)


class TransformerEncoderModel(ForecasterModel):
    """
    A ConvoKit Forecaster-adherent implementation of conversational forecasting model based on Transformer Encoder Model (e.g. BERT, RoBERTa, SpanBERT, DeBERTa).
    This class is first used in the paper "Conversations Gone Awry, But Then? Evaluating Conversational Forecasting Models"
    (Tran et al., 2025).

    :param model_name_or_path: The name or local path of the pretrained transformer model to load.
    :param config: (Optional) TransformerForecasterConfig object containing parameters for training and evaluation.
    """

    def __init__(self, model_name_or_path, config=DEFAULT_CONFIG, decision_policy=None):
        super().__init__(decision_policy=decision_policy)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            model_max_length=512,
            truncation_side="left",
            padding_side="right",
        )
        model_config = AutoConfig.from_pretrained(
            model_name_or_path, num_labels=2, problem_type="single_label_classification"
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, ignore_mismatched_sizes=True, config=model_config
        ).to(config.device)
        if not os.path.exists(config.output_dir):
            os.makedirs(config.output_dir)
        self.config = config
        return

    @property
    def best_threshold(self):
        if hasattr(self.decision_policy, "threshold"):
            return self.decision_policy.threshold
        return None

    @best_threshold.setter
    def best_threshold(self, value):
        if hasattr(self.decision_policy, "threshold"):
            self.decision_policy.threshold = float(value)

    def get_checkpoints(self):
        return [cp for cp in os.listdir(self.config.output_dir) if "checkpoint-" in cp]

    def load_checkpoint(self, checkpoint_name):
        full_model_path = os.path.join(self.config.output_dir, checkpoint_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(full_model_path).to(
            self.config.device
        )

    def finalize_best_checkpoint_selection(
        self, best_checkpoint, best_config, val_contexts=None, score_fn=None
    ):
        super().finalize_best_checkpoint_selection(
            best_checkpoint, best_config, val_contexts=val_contexts, score_fn=score_fn
        )
        if val_contexts is None or self.best_threshold is None:
            return
        val_dataset = self._context_to_bert_data(val_contexts)
        val_dataset.set_format("torch")
        eval_forecasts_df = self._score_dataset(val_dataset, threshold=self.best_threshold)
        eval_prediction_file = os.path.join(self.config.output_dir, "val_predictions.csv")
        eval_forecasts_df.to_csv(eval_prediction_file)

    def _context_mode(self, context):
        """
        Select the utterances to include in the input context based on the configured context mode.

        This method determines whether to include the full dialogue context or only
        the current utterance, depending on the value of `self.config.context_mode`.

        Supported modes:
        - "normal": Use the full dialogue context (i.e., all utterances leading up to the current one).
        - "no-context": Use only the current utterance.

        :param context: A context tuple containing `context.context` (prior utterances)
            and `context.current_utterance`.

        :return: A list of utterance objects to be used for tokenization.

        :raises ValueError: If `self.config.context_mode` is not one of the supported values.
        """
        if self.config.context_mode == "normal":
            context_utts = context.context
        elif self.config.context_mode == "no-context":
            context_utts = [context.current_utterance]
        else:
            raise ValueError(
                f"Context mode {self.config.context_mode} is not defined. Valid value must be either 'normal' or 'no-context'."
            )
        return context_utts

    def _tokenize(self, context):
        """
        Tokenize a list of utterances into model-ready input using the class tokenizer.

        This method joins the utterances in the given context using the tokenizer's
        separator token (e.g., `[SEP]`), then tokenizes the resulting. It applies
        padding and truncation to ensure the sequence fits within the model's maximum
        input length.

        :param context: A list of Utterance objects.

        :return: A dictionary containing:
            - 'input_ids': the token IDs for the input sequence
            - 'attention_mask': the attention mask corresponding to the input
        """
        tokenized_context = self.tokenizer.encode_plus(
            text=f" {self.tokenizer.sep_token} ".join([u.text for u in context]),
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=512,
        )
        return tokenized_context

    def _context_to_bert_data(self, contexts):
        """
        Convert context tuples into a HuggingFace Dataset formatted for BERT-family models.

        This method processes each context tuple by:
        - Extracting the full conversation history associated with the current utterance
        - Generating a label for the conversation using the provided `self.labeler`
        - Formatting the context according to the model’s context mode
        - Tokenizing the resulting text input
        - Collecting input IDs, attention masks, labels, and utterance IDs

        The result is packaged into a `datasets.Dataset` object suitable for training
        or evaluation with a Transformer-based classification model.

        :param contexts: An iterable of context tuples, each containing a current utterance
            and its conversation history.

        :return: A HuggingFace `Dataset` with fields:
            - 'input_ids': tokenized input sequences
            - 'attention_mask': corresponding attention masks
            - 'labels': ground-truth binary labels
            - 'id': IDs of the current utterances
        """
        pairs = {"id": [], "input_ids": [], "attention_mask": [], "labels": []}
        for context in contexts:
            convo = context.current_utterance.get_conversation()
            label = self.labeler(convo)

            context_utts = self._context_mode(context)
            tokenized_context = self._tokenize(context_utts)
            pairs["input_ids"].append(tokenized_context["input_ids"])
            pairs["attention_mask"].append(tokenized_context["attention_mask"])
            pairs["labels"].append(label)
            pairs["id"].append(context.current_utterance.id)
        return Dataset.from_dict(pairs)

    @torch.inference_mode
    @torch.no_grad
    def _score_dataset(
        self,
        dataset,
        model=None,
        threshold=None,
        forecast_prob_attribute_name="forecast_prob",
        forecast_attribute_name="forecast",
    ):
        """
        Generate predictions using the model on the given dataset and return them in a Pandas DataFrame.

        :param dataset: A torch-formatted iterable (e.g., HuggingFace Dataset) where each item contains
            'input_ids', 'attention_mask', and 'id'.
        :param model: (Optional) A PyTorch model for inference. If not provided, `self.model` is used.
        :param threshold: (float) Threshold to convert raw probabilities into binary predictions.
        :param forecast_prob_attribute_name: (Optional) Column name for raw forecast probabilities in the output DataFrame.
            Defaults to "forecast_prob" if not specified.
        :param forecast_attribute_name: (Optional) Column name for binary predictions in the output DataFrame.
            Defaults to "forecast" if not specified.

        :return: A Pandas DataFrame indexed by utterance ID. Contains two columns:
            - One with raw probabilities (named `forecast_prob_attribute_name`)
            - One with binary predictions (named `forecast_attribute_name`)
        """
        if not model:
            model = self.model.to(self.config.device)
        if threshold is None:
            threshold = self.best_threshold if self.best_threshold is not None else 0.5
        utt_ids = []
        preds = []
        scores = []
        for data in tqdm(dataset):
            input_ids = data["input_ids"].to(self.config.device, dtype=torch.long).reshape([1, -1])
            attention_mask = (
                data["attention_mask"].to(self.config.device, dtype=torch.long).reshape([1, -1])
            )
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = F.softmax(outputs.logits, dim=-1)
            utt_ids.append(data["id"])
            raw_score = probs[0, 1].item()
            preds.append(int(raw_score > threshold))
            scores.append(raw_score)

        return pd.DataFrame(
            {forecast_attribute_name: preds, forecast_prob_attribute_name: scores}, index=utt_ids
        )

    @torch.inference_mode
    @torch.no_grad
    def score(self, context) -> float:
        self.model.eval()
        context_utts = self._context_mode(context)
        tokenized_context = self._tokenize(context_utts)
        input_ids = (
            torch.tensor(tokenized_context["input_ids"], dtype=torch.long)
            .to(self.config.device)
            .reshape([1, -1])
        )
        attention_mask = (
            torch.tensor(tokenized_context["attention_mask"], dtype=torch.long)
            .to(self.config.device)
            .reshape([1, -1])
        )
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        probs = F.softmax(outputs.logits, dim=-1)
        return probs[0, 1].item()

    def fit_belief_estimator(self, contexts, val_contexts=None):
        """
        Fine-tune the TransformerEncoder model, and save the best model according to validation performance.

        This method transforms the input contexts into model-compatible format,
        configures training parameters, and trains the model using HuggingFace's
        Trainer API. It also tunes a decision threshold using a separate
        held-out validation set.

        :param contexts: an iterator over context tuples, provided by the Forecaster framework
        :param val_contexts: optional validation contexts (not used by this stage).
        """
        train_pairs = self._context_to_bert_data(contexts)
        dataset = DatasetDict({"train": train_pairs})
        dataset.set_format("torch")

        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.per_device_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            num_train_epochs=self.config.num_train_epochs,
            learning_rate=self.config.learning_rate,
            logging_strategy="epoch",
            weight_decay=0.01,
            eval_strategy="no",
            save_strategy="epoch",
            prediction_loss_only=False,
            seed=self.config.random_seed,
        )
        trainer = Trainer(model=self.model, args=training_args, train_dataset=dataset["train"])
        trainer.train()
        return

    def fit(self, contexts, val_contexts=None):
        val_contexts_belief_estimator, val_contexts_decision_policy = tee(val_contexts, 2)
        self.fit_belief_estimator(contexts, val_contexts_belief_estimator)
        self.fit_decision_policy(contexts, val_contexts_decision_policy, score_fn=self.score)
        return

    def _predict(self, context, threshold=None):
        utt_score = self.score(context)
        # keep threshold override for backward compatibility.
        if threshold is not None:
            utt_pred = int(utt_score > threshold)
        else:
            result = self.decision_policy.decide(context, self.score)
            if len(result) == 2:
                utt_score, utt_pred = result
            elif len(result) == 3:
                utt_score, utt_pred, _ = result
            else:
                raise ValueError(
                    "decision_policy.decide() must return (utt_score, utt_pred) "
                    "or (utt_score, utt_pred, metadata_dict)"
                )
        return utt_score, utt_pred

    def transform(self, contexts, forecast_attribute_name, forecast_prob_attribute_name):
        """
        Generate forecasts using the fine-tuned TransformerEncoder model on the provided contexts, and save the predictions to the output directory specified in the configuration.

        :param contexts: context tuples from the Forecaster framework
        :param forecast_attribute_name: Forecaster will use this to look up the table column containing your model's discretized predictions (see output specification below)
        :param forecast_prob_attribute_name: Forecaster will use this to look up the table column containing your model's raw forecast probabilities (see output specification below)

        :return: a Pandas DataFrame, with one row for each context, indexed by the ID of that context's current utterance. Contains two columns, one with raw probabilities named according to forecast_prob_attribute_name, and one with discretized (binary) forecasts named according to forecast_attribute_name
        """
        utt_ids = []
        preds = []
        scores = []
        for context in tqdm(contexts):
            utt_score, utt_pred = self._predict(context)
            utt_ids.append(context.current_utterance.id)
            preds.append(utt_pred)
            scores.append(utt_score)
        forecasts_df = pd.DataFrame(
            {forecast_attribute_name: preds, forecast_prob_attribute_name: scores}, index=utt_ids
        )

        prediction_file = os.path.join(self.config.output_dir, "test_predictions.csv")
        forecasts_df.to_csv(prediction_file)

        return forecasts_df
