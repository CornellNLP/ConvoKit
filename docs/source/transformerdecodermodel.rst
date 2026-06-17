Transformer Decoder-based Forecasting Model
===========================================
Please note: this transformer requires installing convokit with the optional LLM packages via `pip install convokit[llm]` (or `pip install convokit[llm-mac]` if you are on macOS)

A ConvoKit Forecaster-adherent implementation of conversational forecasting model based on Transformer Decoder Model (e.g. LlaMA, Gemma, GPT).
This class is first used in the paper "Conversations Gone Awry, But Then? Evaluating Conversational Forecasting Models" (Tran et al., 2025).

NOTE: This implementation fine-tunes a TransformerDecoder-based language model using parameter-efficient fine-tuning (PEFT) with LoRA (Low-Rank Adaptation) with a rank of 64 and LoRA-specific configurations targeting key transformer modules such as q_proj, k_proj, v_proj. Gradient checkpointing is enabled via Unsloth to support training with long contexts efficiently.
Supported model families include: Gemma2, Gemma3, Mistral, Zephyr, Phi-4, and LLaMA 3.

.. automodule:: convokit.forecaster.TransformerDecoderModel
    :members:
