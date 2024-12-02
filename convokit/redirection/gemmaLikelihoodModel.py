from .likelihoodModel import LikelihoodModel
import torch
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM, PeftModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
)
from trl import SFTTrainer
from .config import DEFAULT_TRAIN_CONFIG, DEFAULT_BNB_CONFIG, DEFAULT_LORA_CONFIG


class GemmaLikelihoodModel(LikelihoodModel):
    """
    Likelihood model supported by Gemma, used to compute utterance likelihoods.

    :param hf_token: Huggingface authentication token
    :param model_id: Gemma model id version
    :param device: Device to use
    :param train_config: Training config for fine-tuning
    :param bnb_config: bitsandbytes config for quantization
    :param lora_config: LoRA config for fine-tuning
    """

    def __init__(
        self,
        hf_token,
        model_id="google/gemma-2b",
        device="cuda" if torch.cuda.is_available() else "cpu",
        train_config=DEFAULT_TRAIN_CONFIG,
        bnb_config=DEFAULT_BNB_CONFIG,
        lora_config=DEFAULT_LORA_CONFIG,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, token=hf_token, padding_side="right"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb_config, device_map="auto", token=hf_token
        )
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        self.hf_token = hf_token
        self.device = device
        self.train_config = train_config
        self.lora_config = lora_config
        self.bnb_config = bnb_config
        self.max_length = self.train_config["max_seq_length"]

    def name(self):
        return self.__class__.name

    def fit(self, train_data, val_data):
        """
        Fine-tunes the Gemma model on the provided `train_data` and validates
        on `val_data`.

        :param train_data: Data to fine-tune model
        :param val_data: Data to validate model
        """
        training_args = TrainingArguments(
            output_dir=self.train_config["output_dir"],
            logging_dir=self.train_config["logging_dir"],
            logging_steps=self.train_config["logging_steps"],
            eval_steps=self.train_config["eval_steps"],
            num_train_epochs=self.train_config["num_train_epochs"],
            per_device_train_batch_size=self.train_config["per_device_train_batch_size"],
            per_device_eval_batch_size=self.train_config["per_device_eval_batch_size"],
            evaluation_strategy=self.train_config["evaluation_strategy"],
            save_strategy=self.train_config["save_strategy"],
            save_steps=self.train_config["save_steps"],
            optim=self.train_config["optim"],
            learning_rate=self.train_config["learning_rate"],
            load_best_model_at_end=self.train_config["load_best_model_at_end"],
        )

        trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_data,
            eval_dataset=val_data,
            args=training_args,
            peft_config=self.lora_config,
            max_seq_length=self.train_config["max_seq_length"],
        )
        trainer.train()

    def _calculate_likelihood_prob(self, past_context, future_context):
        """
        Computes the utterance likelihoods given the previous context to
        condition on and the future context to predict.

        :param past_context: Context to condition
        :param future_context: Context to predict likelihood

        :return: likelihoods of contexts
        """
        past_context = "\n\n".join(past_context)
        future_context = "\n\n".join(future_context)

        context_ids = self.tokenizer.encode(
            past_context,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        future_ids = self.tokenizer.encode(
            future_context,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = torch.cat([context_ids, future_ids], dim=1)
        if input_ids.shape[1] > self.max_length:
            input_ids = input_ids[:, -self.max_length :]
        input_ids = input_ids.to(self.device)
        with torch.no_grad():
            probs = torch.nn.functional.softmax(self.model(input_ids)[0], dim=-1)
        cond_log_probs = []
        for i, future_id in enumerate(future_ids[0]):
            index = i + (input_ids.shape[1] - future_ids.shape[1]) - 1
            logprob = torch.log(probs[0, index, future_id])
            cond_log_probs.append(logprob.item())
        result = sum(cond_log_probs)
        return result

    def transform(self, test_data, verbosity=5):
        """
        Computes the utterance likelihoods for the provided `test_data`.

        :param test_data: Data to compute likelihoods over
        :param verbosity: Verbosity to print updated messages

        :return: likelihoods of the `test_data`
        """
        prev_contexts, future_contexts = test_data
        likelihoods = []
        for i in range(len(prev_contexts)):
            if i % verbosity == 0 and i > 0:
                print(i, "/", len(test_data))
            convo_likelihoods = {}
            convo_prev_contexts = prev_contexts[i]
            convo_future_contexts = future_contexts[i]
            for utt_id in convo_prev_contexts:
                if utt_id not in convo_future_contexts:
                    continue
                utt_prev_context = convo_prev_contexts[utt_id]
                utt_future_context = convo_future_contexts[utt_id]
                convo_likelihoods[utt_id] = self._calculate_likelihood_prob(
                    past_context=utt_prev_context, future_context=utt_future_context
                )
            likelihoods.append(convo_likelihoods)
        return likelihoods
