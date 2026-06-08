from itertools import tee, islice
import unsloth
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
import torch
import torch.nn.functional as F
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from collections import defaultdict

import os
from tqdm import tqdm
import pandas as pd
from .forecasterModel import ForecasterModel
from .TransformerForecasterConfig import TransformerForecasterConfig


def _get_template_map(model_name_or_path):
    """
    Map a model name or path to its corresponding prompt template family.

    :param model_name_or_path: Full model name or path.
    :return: Template name corresponding to the model family.
    :raises ValueError: If the model is not recognized.
    """
    TEMPLATE_PATTERNS = [
        ("gemma-2", "gemma2"),
        ("gemma-3", "gemma3"),
        ("mistral", "mistral"),
        ("zephyr", "zephyr"),
        ("phi-4", "phi-4"),
        ("llama-3", "llama3"),
    ]

    for pattern, template in TEMPLATE_PATTERNS:
        if pattern in model_name_or_path.lower():
            return template

    raise ValueError(f"Model '{model_name_or_path}' is not supported.")


DEFAULT_CONFIG = TransformerForecasterConfig(
    output_dir="TransformerDecoderModel",
    gradient_accumulation_steps=32,
    per_device_batch_size=2,
    num_train_epochs=1,
    learning_rate=1e-4,
    random_seed=1,
    context_mode="normal",
    device="cuda",
)


class TransformerDecoderModel(ForecasterModel):
    """
    A ConvoKit Forecaster-adherent implementation of conversational forecasting model based on Transformer Decoder Model (e.g. LlaMA, Gemma, GPT).
    This class is first used in the paper "Conversations Gone Awry, But Then? Evaluating Conversational Forecasting Models"
    (Tran et al., 2025).
    Supported model families include: Gemma2, Gemma3, Mistral, Zephyr, Phi-4, and LLaMA 3.

    :param model_name_or_path: The name or local path of the pretrained transformer model to load.
    :param config: (Optional) TransformerForecasterConfig object containing parameters for training and evaluation.
    :param system_msg: (Optional) Custom system-level message guiding the forecaster's behavior. If not provided, a default prompt tailored for CGA (Conversation Gone Awry) moderation tasks is used.
    :param question_msg: (Optional) Custom question prompt posed to the transformer model. If not provided, defaults to a standard CGA question asking about potential conversation derailment.
    """

    def __init__(
        self,
        model_name_or_path,
        config=DEFAULT_CONFIG,
        system_msg=None,
        question_msg=None,
        decision_policy=None,
    ):
        super().__init__(decision_policy=decision_policy)
        self.max_seq_length = 4_096 * 2
        self.model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name_or_path,
            max_seq_length=self.max_seq_length,
            load_in_4bit=True,
        )

        self.tokenizer = get_chat_template(
            tokenizer,
            chat_template=_get_template_map(self.model.config.name_or_path),
            mapping={"role": "from", "content": "value", "user": "human", "assistant": "model"},
        )
        # Custom prompt
        if system_msg and question_msg:
            self.system_msg = system_msg
            self.question_msg = question_msg
        # Default Prompt for CGA tasks
        if system_msg == question_msg == None:
            self.system_msg = (
                "Here is an ongoing conversation and you are the moderator. "
                "Observe the conversational and speaker dynamics to see if the conversation will derail into a personal attack. "
                "Be careful, not all sensitive topics lead to a personal attack."
            )
            self.question_msg = (
                "Will the above conversation derail into a personal attack now or at any point in the future? "
                "Strictly start your answer with Yes or No, otherwise the answer is invalid."
            )

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
        checkpoints = [cp for cp in os.listdir(self.config.output_dir) if "checkpoint-" in cp]
        if len(checkpoints) == 0:
            return ["zero-shot"]
        return checkpoints

    def load_checkpoint(self, checkpoint_name):
        if checkpoint_name == "zero-shot":
            return
        full_model_path = os.path.join(self.config.output_dir, checkpoint_name)
        self.model, _ = FastLanguageModel.from_pretrained(
            model_name=full_model_path,
            max_seq_length=self.max_seq_length,
            load_in_4bit=True,
        )

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

    def _tokenize(
        self,
        context_utts,
        label=None,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ):
        """
        Format and tokenize a sequence of utterances into model-ready input using a chat-style prompt.

        :param context_utts: A list of utterance objects to include in the prompt. Each utterance
            must have `.speaker_.id` and `.text` attributes.
        :param label: (Optional) A binary label indicating the target response ("Yes" or "No").
            If provided, it will be included in the final message under the "model" role.
        :param tokenize: (bool) Whether to tokenize the final message using the tokenizer.
            Defaults to True.
        :param add_generation_prompt: (bool) Whether to append a generation prompt at the end
            for decoder-style models. Defaults to True.
        :param return_tensors: Format in which to return tokenized tensors (e.g., `'pt'` for PyTorch).
            Passed to the tokenizer.

        :return: Tokenized input returned by `tokenizer.apply_chat_template`, ready for model input.
        """
        messages = [self.system_msg]
        for idx, utt in enumerate(context_utts):
            messages.append(f"[utt-{idx + 1}] {utt.speaker_.id}: {utt.text}")
        messages.append(self.question_msg)

        # Truncation
        human_message = "\n\n".join(messages)
        tokenized_message = self.tokenizer(human_message)["input_ids"]
        if len(tokenized_message) > self.max_seq_length - 100:
            human_message = self.tokenizer.decode(tokenized_message[-self.max_seq_length + 100 :])
        final_message = [{"type": "text", "from": "human", "value": human_message}]

        if label != None:
            text_label = "Yes" if label else "No"
            final_message.append({"type": "text", "from": "model", "value": text_label})

        tokenized_context = self.tokenizer.apply_chat_template(
            final_message,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            return_tensors=return_tensors,
        )
        return tokenized_context

    def _context_to_llm_data(self, contexts):
        """
        Convert context tuples into a HuggingFace Dataset formatted for LLM-style training.

        This method processes each context tuple by:
        - Extracting the full conversation associated with the current utterance
        - Generating a binary label using `self.labeler`
        - Formatting the context into a structured prompt using `_tokenize` (without actual tokenization)
        - Collecting the resulting prompt text into a list of training samples

        The output is a list of dictionaries with a single "text" field, suitable for training
        large language models (LLMs) in a text-to-text setting.

        :param contexts: An iterable of context tuples, each with a current utterance and
            conversation history.

        :return: A HuggingFace `Dataset` object containing one entry per context with a "text" field.
        """
        dataset = []
        for context in contexts:
            convo = context.current_utterance.get_conversation()
            label = self.labeler(convo)
            context_utts = self._context_mode(context)
            inputs = self._tokenize(
                context_utts,
                label=label,
                tokenize=False,
                add_generation_prompt=False,
                return_tensors=None,
            )
            dataset.append({"text": inputs})
        print(f"There are {len(dataset)} samples")
        return Dataset.from_list(dataset)

    def fit_belief_estimator(self, train_contexts, val_contexts=None):
        """
        Fine-tune the TransformerDecoder model using LoRA and save the best model based on validation performance.

        This method applies Low-Rank Adaptation (LoRA) to the decoder model, converts the
        training contexts into text-based input for LLM fine-tuning, and trains the model
        using HuggingFace's `SFTTrainer`.

        :param contexts: an iterator over context tuples, provided by the Forecaster framework
        :param val_contexts: an iterator over context tuples to be used only for validation.
        """
        # LORA
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=64,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=128,
            lora_dropout=0,  # supports any, but = 0 is optimized
            bias="none",  # supports any, but = "none" is optimized
            use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
            random_state=0,
            use_rslora=False,  # rank stabilized LoRA (True for new_cmv3/new_cmv4, False for new_cmv/new_cmv2)
            loftq_config=None,  # and LoftQ
        )
        # Processing Data
        train_dataset = self._context_to_llm_data(train_contexts)
        print(train_dataset)

        # Training
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            max_seq_length=self.max_seq_length,
            args=SFTConfig(
                dataset_text_field="text",
                per_device_train_batch_size=self.config.per_device_batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                warmup_steps=10,
                num_train_epochs=self.config.num_train_epochs,
                logging_strategy="epoch",
                save_strategy="epoch",
                learning_rate=self.config.learning_rate,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                optim="adamw_8bit",
                optim_target_modules=["attn", "mlp"],
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=0,
                output_dir=self.config.output_dir,
                report_to="none",
            ),
        )
        trainer.train()
        return

    def score(self, context) -> float:
        FastLanguageModel.for_inference(self.model)
        context_utts = self._context_mode(context)
        inputs = self._tokenize(context_utts).to(self.config.device)
        model_response = self.model.generate(
            input_ids=inputs,
            streamer=None,
            max_new_tokens=1,
            pad_token_id=self.tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )
        scores = model_response["scores"][0][0]

        yes_id = self.tokenizer.convert_tokens_to_ids("Yes")
        no_id = self.tokenizer.convert_tokens_to_ids("No")
        yes_logit = scores[yes_id].item()
        no_logit = scores[no_id].item()
        utt_score = F.softmax(torch.tensor([yes_logit, no_logit], dtype=torch.float32), dim=0)[
            0
        ].item()
        return utt_score

    def _predict(self, context, threshold=None):
        """
        Run inference on a single context using the fine-tuned TransformerDecoder model.

        This method prepares the input from the given context, generates a single-token
        prediction (either "Yes" or "No"), and computes the softmax probability for "Yes".
        The output is a confidence score and a binary prediction based on the given or
        default threshold.

        :param context: A context tuple containing the current utterance and conversation history.
        :param threshold: (Optional) A float threshold for converting the predicted probability into a binary label.
            If not provided, `self.best_threshold` is used.

        :return: A tuple (`utt_score`, `utt_pred`), where:
            - `utt_score` is the softmax probability assigned to "Yes"
            - `utt_pred` is the binary prediction (1 if `utt_score > threshold`, else 0)
        """
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

    def fit(self, contexts, val_contexts=None):
        val_contexts_belief_estimator, val_contexts_decision_policy = tee(val_contexts, 2)
        self.fit_belief_estimator(contexts, val_contexts_belief_estimator)
        self.fit_decision_policy(contexts, val_contexts_decision_policy, score_fn=self.score)
        return

    def transform(
        self, contexts, forecast_attribute_name, forecast_prob_attribute_name, verbose=False
    ):
        """
        Generate forecasts using the fine-tuned TransformerDecoder model on the provided contexts, and save the predictions to the output directory specified in the configuration.

        :param contexts: context tuples from the Forecaster framework
        :param forecast_attribute_name: Forecaster will use this to look up the table column containing your model's discretized predictions (see output specification below)
        :param forecast_prob_attribute_name: Forecaster will use this to look up the table column containing your model's raw forecast probabilities (see output specification below)
        :param verbose: if True, print verbose transform logging during the transformation

        :return: a Pandas DataFrame, with one row for each context, indexed by the ID of that context's current utterance. Contains two columns, one with raw probabilities named according to forecast_prob_attribute_name, and one with discretized (binary) forecasts named according to forecast_attribute_name
        """
        FastLanguageModel.for_inference(self.model)
        utt_ids = []
        preds = []
        scores = []
        metadatas = defaultdict(list)
        # verbose-only diagnostics: running conversation-level metrics and an
        # incremental predictions.csv dump. skipped entirely in the default path.
        report_every_n = 250
        prediction_file = os.path.join(self.config.output_dir, "predictions.csv")
        next_flush_start = 0
        csv_header_written = False
        convo_forecasts = {}
        convo_labels = {}
        if verbose and os.path.exists(prediction_file):
            os.remove(prediction_file)

        def _compute_conversation_metrics():
            common_convo_ids = [cid for cid in convo_forecasts if cid in convo_labels]
            if len(common_convo_ids) == 0:
                return None
            tp = 0
            fp = 0
            tn = 0
            fn = 0
            for convo_id in common_convo_ids:
                pred = int(convo_forecasts[convo_id] > 0)
                label = int(convo_labels[convo_id])
                if label == 1 and pred == 1:
                    tp += 1
                elif label == 0 and pred == 1:
                    fp += 1
                elif label == 0 and pred == 0:
                    tn += 1
                elif label == 1 and pred == 0:
                    fn += 1
            n = len(common_convo_ids)
            acc = (tp + tn) / n if n > 0 else 0.0
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
            return {"n": n, "acc": acc, "p": p, "r": r, "fpr": fpr, "f1": f1}

        # for safety/flexibility we can accept either only score and pred or also the metadata
        progress = tqdm(contexts)
        for idx, context in enumerate(progress, start=1):
            result = self.decision_policy.decide(context, self.score)

            if len(result) == 2:
                utt_score, utt_pred = result
                utt_metadata = {}
                # no metadata
            elif len(result) == 3:
                utt_score, utt_pred, utt_metadata = result
                # coerce None metadata to {} so policies that return (score, pred, None)
                # don't crash downstream utt_metadata.items() / .get() calls.
                if utt_metadata is None:
                    utt_metadata = {}
            else:
                raise ValueError(
                    "decision_policy.decide() must return (utt_score, utt_pred) "
                    "or (utt_score, utt_pred, metadata_dict)"
                )
            utt_ids.append(context.current_utterance.id)
            preds.append(utt_pred)
            scores.append(utt_score)
            current_idx = len(preds) - 1
            existing_metadata_keys = list(metadatas.keys())
            for key in existing_metadata_keys:
                metadatas[key].append(utt_metadata.get(key, None))
            for key, value in utt_metadata.items():
                if key not in metadatas:
                    metadatas[key] = [None] * current_idx
                    metadatas[key].append(value)

            if not verbose:
                continue

            convo_id = getattr(context, "conversation_id", None)
            try:
                convo = context.current_utterance.get_conversation()
                if convo_id is None and convo is not None:
                    convo_id = convo.id
                if convo_id is not None:
                    if convo_id in convo_forecasts:
                        convo_forecasts[convo_id] = max(convo_forecasts[convo_id], int(utt_pred))
                    else:
                        convo_forecasts[convo_id] = int(utt_pred)
                    if convo_id not in convo_labels:
                        convo_labels[convo_id] = int(self.labeler(convo))
            except Exception:
                pass

            if idx % report_every_n == 0:
                batch_cols = {
                    forecast_attribute_name: preds[next_flush_start:idx],
                    forecast_prob_attribute_name: scores[next_flush_start:idx],
                }
                for key, series in metadatas.items():
                    batch_cols[key] = series[next_flush_start:idx]
                batch_df = pd.DataFrame(batch_cols, index=utt_ids[next_flush_start:idx])
                batch_df.to_csv(
                    prediction_file,
                    mode="a" if csv_header_written else "w",
                    header=not csv_header_written,
                )
                csv_header_written = True
                next_flush_start = idx

                running_metrics = _compute_conversation_metrics()
                if running_metrics is not None:
                    tqdm.write(
                        f"[info] transform metrics running: "
                        f"processed_contexts={idx}, conversations={running_metrics['n']}, "
                        f"acc={running_metrics['acc']:.4f}, p={running_metrics['p']:.4f}, "
                        f"r={running_metrics['r']:.4f}, fpr={running_metrics['fpr']:.4f}, "
                        f"f1={running_metrics['f1']:.4f}"
                    )
                else:
                    tqdm.write(
                        f"[info] transform metrics running: "
                        f"processed_contexts={idx}, conversations=0"
                    )
        total_processed = len(preds)
        if verbose and total_processed > next_flush_start:
            batch_cols = {
                forecast_attribute_name: preds[next_flush_start:total_processed],
                forecast_prob_attribute_name: scores[next_flush_start:total_processed],
            }
            for key, series in metadatas.items():
                batch_cols[key] = series[next_flush_start:total_processed]
            batch_df = pd.DataFrame(batch_cols, index=utt_ids[next_flush_start:total_processed])
            batch_df.to_csv(
                prediction_file,
                mode="a" if csv_header_written else "w",
                header=not csv_header_written,
            )
            csv_header_written = True
        cols = {
            forecast_attribute_name: preds,
            forecast_prob_attribute_name: scores,
        }
        if verbose:
            final_metrics = _compute_conversation_metrics()
            if final_metrics is not None:
                tqdm.write(
                    f"[info] final transform metrics: "
                    f"processed_contexts={len(preds)}, conversations={final_metrics['n']}, "
                    f"acc={final_metrics['acc']:.4f}, p={final_metrics['p']:.4f}, "
                    f"r={final_metrics['r']:.4f}, fpr={final_metrics['fpr']:.4f}, "
                    f"f1={final_metrics['f1']:.4f}"
                )
        for key, series in metadatas.items():
            assert len(series) == len(
                preds
            ), "Metadata series length must match number of predictions"
            cols[key] = series  # each series same length as preds
        forecasts_df = pd.DataFrame(cols, index=utt_ids)
        return forecasts_df
