try:
    import unsloth
except NotImplementedError as e:
    raise ImportError("Unsloth GPU requirement not met") from e

from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import get_chat_template

import torch
from datasets import load_from_disk, DatasetDict
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq

from typing import Callable, Optional, Union, Any, List, Iterator
from tqdm import tqdm
import pandas as pd

from .utteranceSimulator import UtteranceSimulator
from .utteranceSimulatorModel import UtteranceSimulatorModel
from .util import contexts_to_dataset, default_prompt_fn, ContextTuple

from .config import (
    DEFAULT_MODEL_CONFIG,
    DEFAULT_TRAIN_CONFIG,
    DEFAULT_GENERATION_CONFIG,
    DEFAULT_NUM_SIMULATIONS,
    DEFAULT_LLAMA_CHAT_TEMPLATE,
    DEFAULT_LLAMA_CHAT_TEMPLATE_MAPPING,
)
from .util import default_prompt_fn


class UnslothUtteranceSimulatorModel(UtteranceSimulatorModel):
    """
    UtteranceSimulatorModel supported for Unsloth framework.

    :param model_name: Name or path of model to use
    :param chat_template: Optional chat template
    :param chat_template_mapping: Optional chat template mapping
    :param device: Device to use
    :param model_config: Model config
    :param train_config: Training config
    :param num_simulations: Number of simulations for each context
    :param prompt_fn: Prompt function to convert from context tuples to prompts
    """

    def __init__(
        self,
        model_name="unsloth/Meta-Llama-3.1-8B-bnb-4bit",
        chat_template=DEFAULT_LLAMA_CHAT_TEMPLATE,
        chat_template_mapping=DEFAULT_LLAMA_CHAT_TEMPLATE_MAPPING,
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_config=DEFAULT_MODEL_CONFIG,
        train_config=DEFAULT_TRAIN_CONFIG,
        num_simulations=DEFAULT_NUM_SIMULATIONS,
        prompt_fn=default_prompt_fn,
    ):
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=model_config["max_seq_length"],
            dtype=model_config["dtype"],
            load_in_4bit=model_config["load_in_4bit"],
        )

        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template=chat_template,
            mapping=chat_template_mapping,
        )

        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=model_config["r"],
            target_modules=model_config["target_modules"],
            lora_alpha=model_config["lora_alpha"],
            lora_dropout=model_config["lora_dropout"],
            bias=model_config["bias"],
            use_gradient_checkpointing=model_config["use_gradient_checkpointing"],
            use_rslora=model_config["use_rslora"],
            loftq_config=model_config["loftq_config"],
        )

        self.device = device
        self.model_config = model_config
        self.train_config = train_config
        self.num_simulations = num_simulations
        self.prompt_fn = prompt_fn

    def get_num_simulations(self):
        """
        Get number of simulations for each utterance.
        """
        return self.num_simulations

    def name(self):
        """
        Get name of the simulator model.
        """
        return self.__class__.name

    def fit(
        self,
        contexts: Iterator[ContextTuple],
        val_contexts: Iterator[ContextTuple],
    ):
        """
        Finetunes this model on the given training `contexts` and validates on
        `val_contexts`.

        :param contexts: Iterator over context tuples for training
        :param val_contexts: Iterator over context tuples for validation
        """
        train_dataset = contexts_to_dataset(
            contexts=contexts, tokenizer=self.tokenizer, prompt_fn=self.prompt_fn, stage="train"
        )
        train_dataset = train_dataset.map(
            lambda examples: self.tokenizer(
                examples["text"],
                padding=True,
                truncation=True,
                max_length=self.model_config["max_seq_length"],
            ),
            batched=True,
        )
        val_dataset = contexts_to_dataset(
            contexts=val_contexts, tokenizer=self.tokenizer, prompt_fn=self.prompt_fn, stage="val"
        )
        val_dataset = val_dataset.map(
            lambda examples: self.tokenizer(
                examples["text"],
                padding=True,
                truncation=True,
                max_length=self.model_config["max_seq_length"],
            ),
            batched=True,
        )

        training_args = TrainingArguments(
            per_device_train_batch_size=self.train_config["per_device_train_batch_size"],
            per_device_eval_batch_size=self.train_config["per_device_eval_batch_size"],
            eval_strategy=self.train_config["eval_strategy"],
            save_strategy=self.train_config["save_strategy"],
            save_steps=self.train_config["save_steps"],
            gradient_accumulation_steps=self.train_config["gradient_accumulation_steps"],
            warmup_steps=self.train_config["warmup_steps"],
            num_train_epochs=self.train_config["num_train_epochs"],
            eval_steps=self.train_config["eval_steps"],
            learning_rate=self.train_config["learning_rate"],
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=self.train_config["logging_steps"],
            optim=self.train_config["optim"],
            weight_decay=self.train_config["weight_decay"],
            lr_scheduler_type=self.train_config["lr_scheduler_type"],
            output_dir=self.train_config["output_dir"],
            logging_dir=self.train_config["logging_dir"],
            load_best_model_at_end=self.train_config["load_best_model_at_end"],
        )
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            dataset_text_field="text",
            max_seq_length=self.model_config["max_seq_length"],
            dataset_num_proc=2,
            packing=False,
            args=training_args,
        )
        trainer.train()

    def transform(
        self,
        contexts: Iterator[ContextTuple],
        simulated_reply_attribute_name: str = "sim_replies",
        generation_config=DEFAULT_GENERATION_CONFIG,
    ):
        """
        Apply this model to the given data, and return its simulated responses
        in the form of a DataFrame indexed by (current) utterance ID

        :param contexts: Iterator over context tuples
        :param simulated_reply_attribute_name: Name of metadata field to save
            simulated replies generated by the model
        :param generation_config: Config containing generation parameters

        :return: a Pandas DataFrame, with one row for each context, indexed by
            the ID of that context's current utterance. Contains list of
            simulated replies in the `simulated_reply_attribute_name` column.
        """
        FastLanguageModel.for_inference(self.model)

        test_dataset = contexts_to_dataset(
            contexts=contexts, tokenizer=self.tokenizer, prompt_fn=self.prompt_fn, stage="test"
        )

        simulations = []
        utt_ids = []
        for data in tqdm(test_dataset):
            text = data["text"]
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.model_config["max_seq_length"],
            ).to(self.device)
            assert inputs["input_ids"].shape[1] <= self.model_config["max_seq_length"], (
                f"Input length {inputs['input_ids'].shape[1]} exceeds max_seq_length "
                f"{self.model_config['max_seq_length']}"
            )
            input_len = inputs["input_ids"].shape[1]
            responses = self.model.generate(
                **inputs,
                max_new_tokens=generation_config["max_new_tokens"],
                use_cache=generation_config["use_cache"],
                do_sample=generation_config["do_sample"],
                temperature=generation_config["temperature"],
                top_p=generation_config["top_p"],
                repetition_penalty=generation_config["repetition_penalty"],
                eos_token_id=self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                pad_token_id=self.tokenizer.pad_token_id,
                num_return_sequences=self.num_simulations
            )
            responses = responses[:, input_len:]
            responses = self.tokenizer.batch_decode(responses, skip_special_tokens=True)

            simulations.append(responses)
            utt_ids.append(data["utt_ids"])

        simulations_df = pd.DataFrame({simulated_reply_attribute_name: simulations}, index=utt_ids)
        return simulations_df
