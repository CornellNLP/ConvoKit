import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from sklearn.metrics import roc_curve
from datasets import Dataset, DatasetDict
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from .forecasterModel import ForecasterModel
import shutil


os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEFAULT_CONFIG = {
    "output_dir": "BERTCGAModel", 
    "per_device_batch_size": 4, 
    "num_train_epochs": 2, 
    "learning_rate": 6.7e-6,
    "random_seed": 1,
    "device": "cuda"
    }
class BERTCGAModel(ForecasterModel):
    """
    Wrapper for Huggingface Transformers AutoModel
    """
    def __init__(
        self,
        model_name_or_path,
        config = DEFAULT_CONFIG
    ):
        super().__init__()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                        model_name_or_path, model_max_length=512, truncation_side="left", padding_side="right"
                        )
        except:
            # The checkpoint didn't save tokenizer
            model_config_file = os.path.join(model_name_or_path, "config.json")
            with open(model_config_file, 'r') as file:
                original_model = json.load(file)['_name_or_path']
            self.tokenizer = AutoTokenizer.from_pretrained(
                        original_model, model_max_length=512, truncation_side="left", padding_side="right"
                        )
            
        model_config = AutoConfig.from_pretrained(model_name_or_path, 
                                                  num_labels=2, 
                                                  problem_type ="single_label_classification")
        self.model = AutoModelForSequenceClassification.from_pretrained(
                                                                model_name_or_path,
                                                                ignore_mismatched_sizes=True,
                                                                config = model_config).to(config["device"])
        if not os.path.exists(config['output_dir']):
            os.makedirs(config['output_dir'])
        self.config = config
        return 
        
    def _tokenize(self, context):
        tokenized_context = self.tokenizer.encode_plus(
            text=f" {self.tokenizer.sep_token} ".join([u.text for u in context]), 
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=512,
            )
        return tokenized_context
    
    def _context_to_bert_data(self, contexts):
        pairs = {"id": [], "input_ids": [], "attention_mask": [], "labels": []}
        for context in contexts:
            convo = context.current_utterance.get_conversation()
            label = self.labeler(convo)

            if ('context_mode' not in self.config) or self.config['context_mode'] == "normal":
                context_utts = context.context
            elif self.config['context_mode'] == "no-context":
                context_utts = [context.current_utterance]
            
            tokenized_context = self._tokenize(context_utts)
            pairs['input_ids'].append(tokenized_context['input_ids'])
            pairs['attention_mask'].append(tokenized_context['attention_mask'])
            pairs['labels'].append(label)
            pairs['id'].append(context.current_utterance.id)
        return Dataset.from_dict(pairs)
    
    @torch.inference_mode
    @torch.no_grad
    def _predict(self, dataset, model=None, threshold=0.5,
                    forecast_prob_attribute_name = None, forecast_attribute_name = None):
        """
        Return predictions in DataFrame
        """
        if not forecast_prob_attribute_name:
            forecast_prob_attribute_name = "score"
        if not forecast_attribute_name:
            forecast_attribute_name = "pred"
        if not model:
            model = self.model.to(self.config['device'])
        utt_ids = []
        preds = []
        scores = []
        for data in tqdm(dataset):
            input_ids = data['input_ids'].to(self.config['device'], dtype = torch.long).reshape([1,-1])
            attention_mask = data['attention_mask']\
                .to(self.config['device'], dtype = torch.long).reshape([1,-1])
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = F.softmax(outputs.logits, dim=-1)
            utt_ids.append(data["id"])
            raw_score = probs[0,1].item()
            preds.append(int(raw_score > threshold))
            scores.append(raw_score)

        return pd.DataFrame({forecast_attribute_name: preds, forecast_prob_attribute_name: scores}, 
                            index=utt_ids)

    def _tune_best_val_accuracy(self, val_dataset, val_contexts):
        """
        Save the tuned model to self.best_threshold and self.model
        """
        checkpoints = os.listdir(self.config["output_dir"])
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
            full_model_path = os.path.join(self.config["output_dir"], cp)
            finetuned_model = AutoModelForSequenceClassification.from_pretrained(full_model_path)\
                                .to(self.config['device'])
            val_scores = self._predict(val_dataset, model=finetuned_model)
            # for each CONVERSATION, whether or not it triggers will be effectively determined by what the highest score it ever got was
            highest_convo_scores = {convo_id: -1 for convo_id in val_convo_ids}
            count_correct = 0
            for utt_id in val_scores.index:
                count_correct += 1
                convo_id = utt2convo[utt_id]
                utt_score = val_scores.loc[utt_id].score
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
        eval_prediction_file = os.path.join(self.config['output_dir'], "val_predictions.csv")
        eval_forecasts_df.to_csv(eval_prediction_file)

        # Save the best config
        best_config = {}
        best_config["best_checkpoint"] = best_checkpoint
        best_config["best_threshold"] = self.best_threshold
        best_config["best_val_accuracy"] = best_val_accuracy
        config_file = os.path.join(self.config['output_dir'], "dev_config.json")
        with open(config_file, 'w') as outfile:
            json_object = json.dumps(best_config, indent=4)
            outfile.write(json_object)
        
        # Clean other checkpoints to save disk space.
        for root, dirs, files in os.walk(self.config['output_dir']):
            if ("checkpoint" in root) and (best_checkpoint not in root):
                print("Deleting:", root)
                shutil.rmtree(root)
        return
        
    def fit(self, train_contexts, val_contexts):
        """
        Description: Train the conversational forecasting model on the given data
        Parameters:
        contexts: an iterator over context tuples, as defined by the above data format
        val_contexts: an optional second iterator over context tuples to be used as a separate held-out validation set. 
                        The generator for this must be the same as test generator
        """
        val_contexts = list(val_contexts)
        train_pairs = self._context_to_bert_data(train_contexts)
        val_for_tuning_pairs = self._context_to_bert_data(val_contexts)
        dataset = DatasetDict({
            "train": train_pairs, 
            "val_for_tuning": val_for_tuning_pairs
        })
        dataset.set_format("torch")
        
        training_args = TrainingArguments(
            output_dir=self.config["output_dir"],
            per_device_train_batch_size=self.config["per_device_batch_size"],
            num_train_epochs=self.config["num_train_epochs"],
            learning_rate=self.config["learning_rate"], 
            logging_strategy="epoch",
            weight_decay=0.01,
            eval_strategy="no",
            save_strategy="epoch",
            prediction_loss_only=False,
            seed=self.config["random_seed"],
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset['train']
        )
        trainer.train()

        self._tune_best_val_accuracy(dataset['val_for_tuning'], val_contexts)
        return 

    def transform(self, contexts, forecast_attribute_name, forecast_prob_attribute_name):
        test_pairs = self._context_to_bert_data(contexts)
        dataset = DatasetDict({
            "test": test_pairs
        })
        dataset.set_format("torch")
        forecasts_df = self._predict(dataset["test"], threshold=self.best_threshold,
                                    forecast_attribute_name = forecast_attribute_name, 
                                    forecast_prob_attribute_name=forecast_prob_attribute_name)
        
        prediction_file = os.path.join(self.config["output_dir"], "test_predictions.csv")
        forecasts_df.to_csv(prediction_file)
        
        return forecasts_df