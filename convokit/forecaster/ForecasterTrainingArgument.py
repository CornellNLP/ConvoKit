from dataclasses import dataclass, field

@dataclass
class ForecasterTrainingArgument:
    """
    Configuration class for defining training arguments used during fine-tuning of a TransformerDecoderModel or TransformerEncoderModel.

    This class encapsulates all relevant hyperparameters and system settings required for training,
    evaluation, and reproducibility. Each field is accompanied by descriptive metadata to aid in
    configuration parsing and command-line interfacing (e.g., via `argparse` or `transformers.HfArgumentParser`).

    Attributes:
        output_dir (str): Path to the directory where outputs such as predictions, model checkpoints,
            and training logs will be saved.

        per_device_batch_size (int): Number of samples processed per device (e.g., GPU) in a single batch.
            Default is 4.

        gradient_accumulation_steps (int): Number of steps to accumulate gradients before performing
            a backward pass and optimizer update. Useful for simulating larger batch sizes. Default is 1.

        num_train_epochs (int): Total number of epochs for model training. Default is 4.

        learning_rate (float): Initial learning rate for the optimizer. Default is 1e-4.

        random_seed (int): Seed value to ensure reproducible training behavior. Default is 1.

        device (str): Device identifier on which the model will be trained and evaluated.
            Typically 'cuda', 'cuda:0', or 'cpu'. Default is "cuda".

        context_mode (str): Specifies how the input context is constructed:
            - "normal": Use full conversational context (previous utterances).
            - "no-context": Use only the current utterance.
            Default is "normal".
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
