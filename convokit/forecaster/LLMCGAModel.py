import json
import os
from tqdm import tqdm
import pandas as pd
from transformers import TrainingArguments
from datasets import Dataset
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from .forecasterModel import ForecasterModel

TEMPLATE_MAP = {
    "google/gemma-2-2b-it":"gemma2",
    "google/gemma-2-27b-it":"gemma2",
}
DEFAULT_CONFIG = {
    "output_dir": "LLMCGAModel", 
    "per_device_batch_size": 2,
    "gradient_accumulation_steps":32,
    "num_train_epochs": 2, 
    "learning_rate": 6.7e-6,
    "random_seed": 1,
    "device": "cuda"
}
class LLMCGAModel(ForecasterModel):
    def __init__(
        self,
        model_name_or_path,
        config = DEFAULT_CONFIG
    ):
        if model_name_or_path not in TEMPLATE_MAP:
            raise ValueError(
                    f"Model {model_name_or_path} is not supported."
                )
        self.max_seq_length = 4_096 * 4
        dtype = None
        load_in_4bit = True
        self.model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name_or_path,
            max_seq_length=self.max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            # token=os.environ["HF_TOKEN"],         #Question, Is this important?
        )

        self.tokenizer = get_chat_template(
            tokenizer,
            chat_template=TEMPLATE_MAP[model_name_or_path],                 #TO-DO: Define this
            mapping={"role": "from", "content": "value", "user": "human", "assistant": "model"},
        )

        # Prompt for CGA tasks
        self.system_msg = "Here is an ongoing conversation and you are the moderator. Observe the conversational and speaker dynamics to see if the conversation will derail into a personal attack. Be careful, not all sensitive topics lead to a personal attack."
        self.question_msg = "Will the above conversation derail into a personal attack now or at any point in the future? Answer in one word with 'Yes' or 'No', then explain your reasoning."
        
        if not os.path.exists(config['output_dir']):
            os.makedirs(config['output_dir'])
        self.config = config
        return
    
    def _tokenize(self, context, label=None, return_tensors='pt'):
        # convo = context.current_utterance.get_conversation()
        # label = self.labeler(convo)
        context_utts = context.context
        messages = [self.system_msg]
        for idx, utt in enumerate(context_utts):
            messages.append(
                    f"[utt-{idx + 1}] {utt.speaker_.id}: {utt.text}"
            )
        messages.append(self.question_msg)
        final_message = [{"from": "human", "value":"\n\n".join(messages)}]
        if label != None:
            label = (
                    f'Yes. The next comment, utt-{len(context_utts) + 1}, was removed by a moderator as it included a personal attack."'
                    if label
                    else "No. The conversation remained civil throughout."
                )
            messages.append({"from": "model", "value": label})
        #TO-DO: apply padding.
        tokenized_context = self.tokenizer.apply_chat_template(
        final_message, tokenize=True, add_generation_prompt=True, return_tensors=return_tensors
        )
        return tokenized_context
    
    def fit(self, train_contexts, val_contexts):
        """
        Description: Train the conversational forecasting model on the given data
        Parameters:
        contexts: an iterator over context tuples, as defined by the above data format
        val_contexts: an optional second iterator over context tuples to be used as a separate held-out validation set. 
                        The generator for this must be the same as test generator
        """
        # Processing Data
        train_dataset, val_dataset = [], []
        for context in train_contexts:
            inputs = self._tokenize(context)
            train_dataset.append({"input": inputs})
        for context in val_contexts:
            inputs = self._tokenize(context)
            val_dataset.append({"input": inputs})
        train_dataset = Dataset.from_list(train_dataset)
        val_dataset = Dataset.from_list(val_dataset)
        # LORA
        model = FastLanguageModel.get_peft_model(
                    model,
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
        
        # Training
        trainer = SFTTrainer(
            model=model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            args=TrainingArguments(
                        per_device_train_batch_size=self.config['per_device_batch_size'],
                        per_device_eval_batch_size=self.config['per_device_batch_size'],
                        metric_for_best_model="eval_loss",
                        greater_is_better=False,
                        load_best_model_at_end=True,
                        gradient_accumulation_steps=self.config['gradient_accumulation_steps'],  # 32 for new_cmv, 16 for new_cmv2/new_cmv3, 6 for new_cmv4
                        warmup_steps=10,
                        num_train_epochs=self.config["num_train_epochs"],
                        logging_strategy="epoch",
                        eval_strategy="no",
                        learning_rate=self.config["learning_rate"],  # 1e-4 for new_cmv, 1e-5 for new_cmv2/new_cmv3/new_cmv4
                        fp16=not is_bfloat16_supported(),
                        bf16=is_bfloat16_supported(),
                        optim="adamw_8bit",  # "adamw_8bit" for new_cmv/new_cmv2, "galore_adamw" for new_cmv3/new_cmv4
                        optim_target_modules=["attn", "mlp"],
                        weight_decay=0.01,
                        lr_scheduler_type="linear",
                        seed=0,
                        output_dir=self.config["output_dir"],
                    ),
                )
        trainer.train()

        # Save the finetuned model and tokenizer.
        model.save_pretrained(self.config["output_dir"])
        self.tokenizer.save_pretrained(self.config["output_dir"])

        return
    def transform(self, contexts, forecast_attribute_name, forecast_prob_attribute_name):
        FastLanguageModel.for_inference(self.model)
        utt_ids = []
        preds = []
        reasonings = []
        for context in tqdm(contexts):
            inputs = self._tokenize(context).to(self.config['device'])
            model_response = self.model.generate(
                input_ids=inputs,
                streamer=None,
                max_new_tokens=200,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )[0]
            prompt_len = len(inputs[0])
            reasoning = self.tokenizer.decode(model_response[prompt_len + 1 :])
            utt_pred = self.tokenizer.decode(model_response[prompt_len : prompt_len + 1]).lower()
            
            # Binarize the answer
            if utt_pred.startswith("yes"):
                utt_pred = 1
            elif utt_pred.startswith("no"):
                utt_pred = 0
            else:
                raise ValueError(f"Invalid prediction: {utt_pred}")
            
            utt_ids.append(context.current_utterance.id)
            preds.append(utt_pred)
            reasonings.append(reasoning)        #TO-DO, Do we want this? 
            forecasts_df = pd.DataFrame({forecast_attribute_name: preds, 
                                        forecast_prob_attribute_name: preds}, 
                                        index=utt_ids)
            prediction_file = os.path.join(self.config["output_dir"], "test_predictions.csv")
            forecasts_df.to_csv(prediction_file)
        return forecasts_df