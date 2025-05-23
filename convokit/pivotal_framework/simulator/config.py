DEFAULT_NUM_SIMULATIONS = 10

DEFAULT_LLAMA_CHAT_TEMPLATE = "llama3"
DEFAULT_LLAMA_CHAT_TEMPLATE_MAPPING = {
    "role": "from",
    "content": "value",
    "user": "human",
    "assistant": "gpt",
}

DEFAULT_MODEL_CONFIG = {
    "load_in_4bit": True,
    "max_seq_length": 2048,
    "dtype": None,
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "embed_tokens",
        "lm_head",
    ],
    "r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0,
    "bias": "none",
    "use_gradient_checkpointing": "unsloth",
    "use_rslora": False,
    "loftq_config": None,
}

DEFAULT_TRAIN_CONFIG = {
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    "eval_strategy": "steps",
    "save_strategy": "steps",
    "save_steps": 30,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 5,
    "num_train_epochs": 1,
    "eval_steps": 30,
    "learning_rate": 2e-4,
    "logging_steps": 5,
    "optim": "adamw_8bit",
    "weight_decay": 0.01,
    "lr_scheduler_type": "linear",
    "output_dir": "outputs",
    "logging_dir": "logs",
    "load_best_model_at_end": True,
}

DEFAULT_GENERATION_CONFIG = {
    "max_new_tokens": 120,
    "use_cache": True,
    "do_sample": True,
    "temperature": 0.8,
    "top_p": 0.8,
    "repetition_penalty": 1.2,
}
