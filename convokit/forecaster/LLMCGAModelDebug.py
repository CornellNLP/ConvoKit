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
import shutil

TEMPLATE_MAP = {
    "google/gemma-2-2b-it":"gemma2",
    "google/gemma-2-9b-it":"gemma2",
    "unsloth/gemma-3-12b-it":"gemma3",
    "mistralai/Mistral-7B-Instruct-v0.3":"mistral",
    "HuggingFaceH4/zephyr-7b-beta":"zephyr",
    "microsoft/phi-4":"phi-4",
    "meta-llama/Llama-3.1-8B-Instruct":"llama3",
}
DEFAULT_CONFIG = {
    "output_dir": "LLMCGAModel",
    "per_device_batch_size": 2,
    "gradient_accumulation_steps":32,
    "num_train_epochs": 1,
    "learning_rate": 1e-4,
    "random_seed": 1,
    "do_finetune": False,
    "do_tune_threshold": True,
    "device": "cuda"
}
class LLMCGAModelDebug(ForecasterModel):
    def __init__(
        self,
        model_name_or_path,
        config = DEFAULT_CONFIG
    ):
        if model_name_or_path not in TEMPLATE_MAP:
            raise ValueError(
                    f"Model {model_name_or_path} is not supported."
                )
        self.max_seq_length = 4_096 * 2
        self.model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name_or_path,
            max_seq_length=self.max_seq_length,
            load_in_4bit=True,
        )

        self.tokenizer = get_chat_template(
            tokenizer,
            chat_template=TEMPLATE_MAP[model_name_or_path],                 #TO-DO: Define this
            mapping={"role": "from", "content": "value", "user": "human", "assistant": "model"},
        )

        # Prompt for CGA tasks
        self.system_msg = "Here is an ongoing conversation and you are the moderator. Observe the conversational and speaker dynamics to see if the conversation will derail into a personal attack. Be careful, not all sensitive topics lead to a personal attack."
        self.question_msg = "Will the above conversation derail into a personal attack now or at any point in the future? Strictly start your answer with Yes or No, otherwise the answer is invalid."
        self.best_threshold = 0.5

        if not os.path.exists(config['output_dir']):
            os.makedirs(config['output_dir'])
        self.config = config

        return

    def _tokenize(self, context_utts,
                  label=None,
                  tokenize=True,
                  add_generation_prompt=True,
                  return_tensors='pt'):

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
        final_message = [{"from": "human", "value":human_message}]

        if label != None:
            text_label = "Yes" if label else "No"
            final_message.append({"from": "model", "value": text_label})

        tokenized_context = self.tokenizer.apply_chat_template(
        final_message,
        tokenize=tokenize,
        add_generation_prompt=add_generation_prompt,
        return_tensors=return_tensors
        )
        return tokenized_context

    def _context_to_llm_data(self, contexts):
        dataset = []
        for context in contexts:
            convo = context.current_utterance.get_conversation()
            label = self.labeler(convo)
            if ("context_mode" not in self.config) or self.config["context_mode"] == "normal":
                context_utts = context.context
            elif self.config["context_mode"] == "no-context":
                context_utts = [context.current_utterance]
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
        Description: Train the conversational forecasting model on the given data
        Parameters:
        contexts: an iterator over context tuples, as defined by the above data format
        val_contexts: an optional second iterator over context tuples to be used as a separate held-out validation set.
                        The generator for this must be the same as test generator
        """
        if (not self.config['do_finetune']) and (not self.config['do_tune_threshold']):
            return
        if (self.config['do_finetune']) and (not self.config['do_tune_threshold']):
            raise ValueError(
                    f"When do_finetune is True, do_tune_threshold must also be True."
                )
        if self.config['do_finetune']:
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
                            per_device_train_batch_size=self.config['per_device_batch_size'],
                            gradient_accumulation_steps=self.config['gradient_accumulation_steps'],
                            warmup_steps=10,
                            num_train_epochs=self.config["num_train_epochs"],
                            logging_strategy="epoch",
                            save_strategy="epoch",
                            learning_rate=self.config["learning_rate"],
                            fp16=not is_bfloat16_supported(),
                            bf16=is_bfloat16_supported(),
                            optim="adamw_8bit",
                            optim_target_modules=["attn", "mlp"],
                            weight_decay=0.01,
                            lr_scheduler_type="linear",
                            seed=0,
                            output_dir=self.config["output_dir"],
                            report_to="none",
                            )
                            )
            trainer.train()

        if self.config['do_tune_threshold']:
            best_config = self._tune_best_val_accuracy(val_contexts)
        if self.config['do_finetune']:
            # Save the tokenizer.
            self.tokenizer.save_pretrained(os.path.join(self.config["output_dir"], best_config['best_checkpoint']))
        return

    def _tune_best_val_accuracy(self, val_contexts):
        """
        Save the tuned model to self.best_threshold and self.model
        """
        checkpoints = [cp for cp in os.listdir(self.config["output_dir"]) if 'checkpoint-' in cp]
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
            if cp == "zero-shot":
                print("Performing Zero-shot Inference.")
            else:
                full_model_path = os.path.join(self.config["output_dir"], cp)
                self.model, _ = FastLanguageModel.from_pretrained(
                        model_name=full_model_path,
                        max_seq_length=self.max_seq_length,
                        load_in_4bit=True,
                        )
                self.model.to(self.config["device"])
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
                self.model = model

        # Save the best config
        best_config = {}
        best_config["best_checkpoint"] = best_checkpoint
        best_config["best_threshold"] = self.best_threshold
        best_config["best_val_accuracy"] = best_val_accuracy
        config_file = os.path.join(self.config["output_dir"], "dev_config.json")
        with open(config_file, "w") as outfile:
            json_object = json.dumps(best_config, indent=4)
            outfile.write(json_object)

        # Clean other checkpoints to save disk space.
        for root, _, _ in os.walk(self.config["output_dir"]):
            if ("checkpoint" in root) and (best_checkpoint not in root):
                print("Deleting:", root)
                shutil.rmtree(root)
        return best_config

    def _predict(self,
                context,
                threshold=None):
        # Enabling inference with different checkpoints to _tune_best_val_accuracy
        if not threshold:
            threshold = self.best_threshold
        FastLanguageModel.for_inference(self.model)
        if ("context_mode" not in self.config) or self.config["context_mode"] == "normal":
            context_utts = context.context
        elif self.config["context_mode"] == "no-context":
            context_utts = [context.current_utterance]
        inputs = self._tokenize(context_utts).to(self.config['device'])
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
            prediction_file = os.path.join(self.config["output_dir"], "test_predictions.csv")
            forecasts_df.to_csv(prediction_file)
        return forecasts_df