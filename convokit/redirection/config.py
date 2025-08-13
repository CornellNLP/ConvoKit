try:
    from peft import LoraConfig
    from transformers import BitsAndBytesConfig
    import torch

    REDIRECTION_ML_AVAILABLE = True
except (ModuleNotFoundError, ImportError) as e:
    raise ModuleNotFoundError(
        "peft, transformers, or torch is not currently installed. Run 'pip install convokit[ml]' if you would like to use the redirection module."
    ) from e

DEFAULT_BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

DEFAULT_LORA_CONFIG = LoraConfig(
    r=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

DEFAULT_TRAIN_CONFIG = {
    "output_dir": "checkpoints",
    "logging_dir": "logging",
    "logging_steps": 25,
    "eval_steps": 50,
    "num_train_epochs": 2,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "evaluation_strategy": "steps",
    "save_strategy": "steps",
    "save_steps": 50,
    "optim": "paged_adamw_8bit",
    "learning_rate": 2e-4,
    "max_seq_length": 512,
    "load_best_model_at_end": True,
}
