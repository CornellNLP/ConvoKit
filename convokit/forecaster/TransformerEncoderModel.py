try:
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

    TRANSFORMERS_AVAILABLE = True
except (ModuleNotFoundError, ImportError) as e:
    raise ModuleNotFoundError(
        "torch, transformers, or datasets is not currently installed. Run 'pip install convokit[llm]' if you would like to use the TransformerEncoderModel."
    ) from e

import os
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from sklearn.metrics import roc_curve
from .forecasterModel import ForecasterModel
from .TransformerForecasterConfig import TransformerForecasterConfig
import shutil


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

    def __init__(self, model_name_or_path, config=DEFAULT_CONFIG):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            model_max_length=512,
            truncation_side="left",
            padding_side="right",
        )
        self.best_threshold = 0.5
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
    def _predict(
        self,
        dataset,
        model=None,
        threshold=0.5,
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

    def _tune_threshold(self, val_dataset, val_contexts):
        """
        Tune the decision threshold and select the best model checkpoint based on validation accuracy.

        This method evaluates all model checkpoints in the configured output directory using a
        held-out validation set.

        The selected model, threshold, and associated metadata are stored in:
        - `self.model`: the best-performing fine-tuned model
        - `self.best_threshold`: the optimal decision threshold
        - `dev_config.json`: file containing best checkpoint metadata
        - `val_predictions.csv`: CSV file with forecast outputs on the validation set

        Additionally, all non-optimal model checkpoints are removed to save disk space, and the
        tokenizer is saved to the directory of the best checkpoint.

        :param val_dataset: A HuggingFace-compatible dataset containing features for validation.
        :param val_contexts: An iterable of context tuples corresponding to the validation set.
                            Used to map utterance IDs to conversation IDs and extract ground-truth labels.

        :return: A dictionary containing the best checkpoint path, best threshold, and best validation accuracy.
        """
        checkpoints = [cp for cp in os.listdir(self.config.output_dir) if "checkpoint-" in cp]
        best_val_accuracy = 0
        val_convo_ids = set()
        utt2convo = {}
        val_labels_dict = {}
        for context in val_contexts:
            convo_id = context.conversation_id
            utt_id = context.current_utterance.id
            label = self.labeler(context.current_utterance.get_conversation())
            utt2convo[utt_id] = convo_id
            val_labels_dict[convo_id] = label
            val_convo_ids.add(convo_id)
        val_convo_ids = list(val_convo_ids)
        for cp in checkpoints:
            full_model_path = os.path.join(self.config.output_dir, cp)
            finetuned_model = AutoModelForSequenceClassification.from_pretrained(
                full_model_path
            ).to(self.config.device)
            val_scores = self._predict(val_dataset, model=finetuned_model)
            # for each CONVERSATION, whether or not it triggers will be effectively determined by what the highest score it ever got was
            highest_convo_scores = {convo_id: -1 for convo_id in val_convo_ids}
            for utt_id in val_scores.index:
                convo_id = utt2convo[utt_id]
                utt_score = val_scores.loc[utt_id].forecast_prob
                if utt_score > highest_convo_scores[convo_id]:
                    highest_convo_scores[convo_id] = utt_score

            val_labels = np.asarray([int(val_labels_dict[c]) for c in val_convo_ids])
            val_scores = np.asarray([highest_convo_scores[c] for c in val_convo_ids])
            # use scikit learn to find candidate threshold cutoffs
            _, _, thresholds = roc_curve(val_labels, val_scores)

            def acc_with_threshold(y_true, y_score, thresh):
                y_pred = (y_score > thresh).astype(int)
                return (y_pred == y_true).mean()

            accs = [acc_with_threshold(val_labels, val_scores, t) for t in thresholds]
            best_acc_idx = np.argmax(accs)

            print("Accuracy:", cp, accs[best_acc_idx])
            if accs[best_acc_idx] > best_val_accuracy:
                best_checkpoint = cp
                best_val_accuracy = accs[best_acc_idx]
                self.best_threshold = thresholds[best_acc_idx]
                self.model = finetuned_model

        eval_forecasts_df = self._predict(val_dataset, threshold=self.best_threshold)
        eval_prediction_file = os.path.join(self.config.output_dir, "val_predictions.csv")
        eval_forecasts_df.to_csv(eval_prediction_file)

        # Save the best config
        best_config = {}
        best_config["best_checkpoint"] = best_checkpoint
        best_config["best_threshold"] = self.best_threshold
        best_config["best_val_accuracy"] = best_val_accuracy
        config_file = os.path.join(self.config.output_dir, "dev_config.json")
        with open(config_file, "w") as outfile:
            json_object = json.dumps(best_config, indent=4)
            outfile.write(json_object)

        # Clean other checkpoints to save disk space.
        for root, _, _ in os.walk(self.config.output_dir):
            if ("checkpoint" in root) and (best_checkpoint not in root):
                print("Deleting:", root)
                shutil.rmtree(root)
        # Save the tokenizer.
        self.tokenizer.save_pretrained(
            os.path.join(self.config.output_dir, best_config["best_checkpoint"])
        )
        return best_config

    def fit(self, contexts, val_contexts):
        """
        Fine-tune the TransformerEncoder model, and save the best model according to validation performance.

        This method transforms the input contexts into model-compatible format,
        configures training parameters, and trains the model using HuggingFace's
        Trainer API. It also tunes a decision threshold using a separate
        held-out validation set.

        :param contexts: an iterator over context tuples, provided by the Forecaster framework
        :param val_contexts: an iterator over context tuples to be used only for validation.
        """
        val_contexts = list(val_contexts)
        train_pairs = self._context_to_bert_data(contexts)
        val_for_tuning_pairs = self._context_to_bert_data(val_contexts)
        dataset = DatasetDict({"train": train_pairs, "val_for_tuning": val_for_tuning_pairs})
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
        _ = self._tune_threshold(dataset["val_for_tuning"], val_contexts)
        return

    def transform(self, contexts, forecast_attribute_name, forecast_prob_attribute_name):
        """
        Generate forecasts using the fine-tuned TransformerEncoder model on the provided contexts, and save the predictions to the output directory specified in the configuration.

        :param contexts: context tuples from the Forecaster framework
        :param forecast_attribute_name: Forecaster will use this to look up the table column containing your model's discretized predictions (see output specification below)
        :param forecast_prob_attribute_name: Forecaster will use this to look up the table column containing your model's raw forecast probabilities (see output specification below)

        :return: a Pandas DataFrame, with one row for each context, indexed by the ID of that context's current utterance. Contains two columns, one with raw probabilities named according to forecast_prob_attribute_name, and one with discretized (binary) forecasts named according to forecast_attribute_name
        """
        test_pairs = self._context_to_bert_data(contexts)
        dataset = DatasetDict({"test": test_pairs})
        dataset.set_format("torch")
        forecasts_df = self._predict(
            dataset["test"],
            threshold=self.best_threshold,
            forecast_attribute_name=forecast_attribute_name,
            forecast_prob_attribute_name=forecast_prob_attribute_name,
        )

        prediction_file = os.path.join(self.config.output_dir, "test_predictions.csv")
        forecasts_df.to_csv(prediction_file)

        return forecasts_df
