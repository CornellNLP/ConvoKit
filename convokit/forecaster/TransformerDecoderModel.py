import unsloth
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
import torch
import torch.nn.functional as F
import json
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from .forecasterModel import ForecasterModel
from .ForecasterTrainingArgument import ForecasterTrainingArgument
import shutil

def get_templet_map(model_name_or_path):
    TEMPLATE_MAP = {
        "google/gemma-2-2b-it":"gemma2",
        "google/gemma-2-9b-it":"gemma2",
        "google/gemma-3-12b-it":"gemma3",
        "mistralai/Mistral-7B-Instruct-v0.3":"mistral",
        "HuggingFaceH4/zephyr-7b-beta":"zephyr",
        "microsoft/phi-4":"phi-4",
        "meta-llama/Llama-3.1-8B-Instruct":"llama3",
        "meta-llama/Llama-3.2-3B-Instruct":"llama3",
    }
    for model in TEMPLATE_MAP:
        if model in model_name_or_path:
            return TEMPLATE_MAP[model]
    raise ValueError(
            f"Model {model_name_or_path} is not supported."
        )

DEFAULT_CONFIG = ForecasterTrainingArgument(
    output_dir= "TransformerDecoderModel",
    gradient_accumulation_steps= 32,
    per_device_batch_size= 2,
    num_train_epochs= 1,
    learning_rate= 1e-4,
    random_seed= 1,
    context_mode= "normal",
    device= "cuda"
)

class TransformerDecoderModel(ForecasterModel):
    """
    A ConvoKit Forecaster-adherent implementation of conversational forecasting model based on Transformer Decoder Model (e.g. LlaMA, Gemma, GPT).
    This class is first used in the paper "Conversations Gone Awry, But Then? Evaluating Conversational Forecasting Models"
    (Tran et al., 2025).

    :param model_name_or_path: The name or local path of the pretrained transformer model to load.
    :param config (object, optional): ForecasterTrainingArgument object containing parameters for training and evaluation.
    :param system_msg (str, optional): Custom system-level message guiding the forecaster's behavior. If not provided, a default prompt tailored for CGA (Conversation Gone Awry) moderation tasks is used.
    :param question_msg (str, optional): Custom question prompt posed to the transformer model. If not provided, defaults to a standard CGA question asking about potential conversation derailment.
    """

    def __init__(
        self,
        model_name_or_path,
        config = DEFAULT_CONFIG,
        system_msg = None,
        question_msg = None,
    ):
        self.max_seq_length = 4_096 * 2
        self.model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name_or_path,
            max_seq_length=self.max_seq_length,
            load_in_4bit=True,
        )

        self.tokenizer = get_chat_template(
            tokenizer,
            chat_template=get_templet_map(model_name_or_path),                 #TO-DO: Define this
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
        self.best_threshold = 0.5

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

    def _tokenize(self, context_utts,
                  label=None,
                  tokenize=True,
                  add_generation_prompt=True,
                  return_tensors='pt'):
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
            messages.append(
                    f"[utt-{idx + 1}] {utt.speaker_.id}: {utt.text}"
            )
        messages.append(self.question_msg)

        # Truncation
        human_message = "\n\n".join(messages)
        tokenized_message = self.tokenizer(human_message)['input_ids']
        if len(tokenized_message) > self.max_seq_length-100:
            human_message = self.tokenizer.decode(tokenized_message[-self.max_seq_length+100:])
        final_message = [{"type": "text", "from": "human", "value":human_message}]

        if label != None:
            text_label = "Yes" if label else "No"
            final_message.append({"type": "text", "from": "model", "value": text_label})

        tokenized_context = self.tokenizer.apply_chat_template(
        final_message,
        tokenize=tokenize,
        add_generation_prompt=add_generation_prompt,
        return_tensors=return_tensors
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
            inputs = self._tokenize(context_utts,
                                    label=label,
                                    tokenize=False,
                                    add_generation_prompt=False,
                                    return_tensors=None)
            dataset.append({"text": inputs})
        print(f"There are {len(dataset)} samples")
        return Dataset.from_list(dataset)

    def fit(self, train_contexts, val_contexts):
        """
        Fine-tune the TransformerDecoder model using LoRA and save the best model based on validation performance.

        This method applies Low-Rank Adaptation (LoRA) to the decoder model, converts the
        training contexts into text-based input for LLM fine-tuning, and trains the model
        using HuggingFace's `SFTTrainer`. After training, it tunes a decision threshold on
        a held-out validation set to optimize binary forecast classification.

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
            args=SFTConfig(
                        dataset_text_field="text",
                        max_seq_length=self.max_seq_length,
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
                        )
                        )
        trainer.train()
        _ = self._tune_threshold(self, val_contexts)
        return

    def _tune_threshold(self, val_contexts):
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
        checkpoints = [cp for cp in os.listdir(self.config.output_dir) if 'checkpoint-' in cp]
        if checkpoints == []:
            checkpoints.append("zero-shot")
        best_val_accuracy = 0
        val_convo_ids = set()
        utt2convo = {}
        val_labels_dict = {}
        val_contexts = list(val_contexts)
        for context in val_contexts:
            convo_id = context.conversation_id
            utt_id = context.current_utterance.id
            label = self.labeler(context.current_utterance.get_conversation())
            utt2convo[utt_id] = convo_id
            val_labels_dict[convo_id] = label
            val_convo_ids.add(convo_id)
        val_convo_ids = list(val_convo_ids)
        for cp in checkpoints:
            if cp != "zero-shot":
                full_model_path = os.path.join(self.config.output_dir, cp)
                self.model, _ = FastLanguageModel.from_pretrained(
                        model_name=full_model_path,
                        max_seq_length=self.max_seq_length,
                        load_in_4bit=True,
                        )
            FastLanguageModel.for_inference(self.model)
            utt2score = {}
            for context in tqdm(val_contexts):
                utt_score, _ = self._predict(context)
                utt_id = context.current_utterance.id
                utt2score[utt_id] = utt_score
            # for each CONVERSATION, whether or not it triggers will be effectively determined by what the highest score it ever got was
            highest_convo_scores = {convo_id: -1 for convo_id in val_convo_ids}

            for utt_id in utt2convo:
                convo_id = utt2convo[utt_id]
                utt_score = utt2score[utt_id]
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

        # Save the best config
        best_config = {}
        best_config["best_checkpoint"] = best_checkpoint
        best_config["best_threshold"] = self.best_threshold
        best_config["best_val_accuracy"] = best_val_accuracy
        config_file = os.path.join(self.config.output_dir, "dev_config.json")
        with open(config_file, "w") as outfile:
            json_object = json.dumps(best_config, indent=4)
            outfile.write(json_object)
        # Load best model
        best_model_path = os.path.join(self.config.output_dir, best_checkpoint)
        self.model, _ = FastLanguageModel.from_pretrained(
                model_name=best_model_path,
                max_seq_length=self.max_seq_length,
                load_in_4bit=True,
                )

        # Clean other checkpoints to save disk space.
        for root, _, _ in os.walk(self.config.output_dir):
            if ("checkpoint" in root) and (best_checkpoint not in root):
                print("Deleting:", root)
                shutil.rmtree(root)
        # Save the tokenizer.
        self.tokenizer.save_pretrained(os.path.join(self.config.output_dir, best_config['best_checkpoint']))
        return best_config

    def _predict(self,
                context,
                threshold=None):
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
        # Enabling inference with different checkpoints to _tune_best_val_accuracy
        if not threshold:
            threshold = self.best_threshold
        FastLanguageModel.for_inference(self.model)
        context_utts = self._context_mode(context)
        inputs = self._tokenize(context_utts).to(self.config.device)
        model_response = self.model.generate(
                            input_ids=inputs,
                            streamer=None,
                            max_new_tokens=1,
                            pad_token_id=self.tokenizer.eos_token_id,
                            output_scores = True,
                            return_dict_in_generate=True
                        )
        scores = model_response['scores'][0][0]

        yes_id = self.tokenizer.convert_tokens_to_ids('Yes')
        no_id = self.tokenizer.convert_tokens_to_ids('No')
        yes_logit = scores[yes_id].item()
        no_logit = scores[no_id].item()
        utt_score = F.softmax(torch.tensor([yes_logit,no_logit], dtype=torch.float32), dim=0)[0].item()
        utt_pred = int(utt_score > threshold)
        return utt_score, utt_pred

    def transform(self, contexts, forecast_attribute_name, forecast_prob_attribute_name):
        """
        Generate forecasts using the fine-tuned TransformerDecoder model on the provided contexts, and save the predictions to the output directory specified in the configuration.

        :param contexts: context tuples from the Forecaster framework
        :param forecast_attribute_name: Forecaster will use this to look up the table column containing your model's discretized predictions (see output specification below)
        :param forecast_prob_attribute_name: Forecaster will use this to look up the table column containing your model's raw forecast probabilities (see output specification below)

        :return: a Pandas DataFrame, with one row for each context, indexed by the ID of that context's current utterance. Contains two columns, one with raw probabilities named according to forecast_prob_attribute_name, and one with discretized (binary) forecasts named according to forecast_attribute_name
        """
        FastLanguageModel.for_inference(self.model)
        utt_ids = []
        preds = []
        scores = []
        for context in tqdm(contexts):
            utt_score, utt_pred = self._predict(context)

            utt_ids.append(context.current_utterance.id)
            preds.append(utt_pred)
            scores.append(utt_score)
        forecasts_df = pd.DataFrame({forecast_attribute_name: preds,
                                    forecast_prob_attribute_name: scores},
                                    index=utt_ids)
        prediction_file = os.path.join(self.config.output_dir, "test_predictions.csv")
        forecasts_df.to_csv(prediction_file)
        return forecasts_df