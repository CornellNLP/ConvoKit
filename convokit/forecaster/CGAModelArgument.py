from dataclasses import dataclass, field

@dataclass
class CGAModelArgument:
    """
    Arguments for fine-tuning CGA Model.
    """
    output_dir: str = field(
        metadata={"help": "Path to the directory where outputs (e.g., predictions, checkpoints, logs) will be saved."}
    )
    per_device_batch_size: int = field(
        default=4, metadata={"help": "Number of samples processed per device (e.g., GPU) in a single batch."}
    )
    gradient_accumulation_steps: int = field(
        default=1, metadata={"help": "Number of steps to accumulate gradients before performing a backward/update pass."}
    )
    num_train_epochs: int = field(
        default=4, metadata={"help": "Total number of epochs for training the model."}
    )
    learning_rate: float = field(
        default=1e-4, metadata={"help": "Initial learning rate used for model training."}
    )
    random_seed: int = field(
        default=1,
        metadata={"help": "Seed for reproducibility and deterministic behavior during training."},
    )
    do_finetune: bool = field(
        default=True,
        metadata={"help": "Whether to fine-tune the model on the provided dataset (True) or use it as-is (False). Only False for TransformerDecoderCGA Model"},
    )
    do_tune_threshold: bool = field(
        default=None,
        metadata={
            "help": "Whether to perform decision threshold tuning based to maximize the accuracy on validation split."
        },
    )
    device: str = field(
        default="cuda",
        metadata={
            "help": "Device identifier specifying where the model runs, e.g., 'cpu', 'cuda', or 'cuda:0'."
        },
    )
    context_mode: str = field(
        default="normal",
        metadata={
            "help": "Mode specifying whether conversational context is included ('normal') or excluded ('no-context')."
        },
    )
