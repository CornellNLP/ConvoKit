Transformer Encoder-based Forecasting Model
===========================================

A ConvoKit Forecaster-adherent implementation of conversational forecasting model based on Transformer Encoder Model (e.g. BERT, RoBERTa, SpanBERT, DeBERTa).
This class is first used in the paper "Conversations Gone Awry, But Then? Evaluating Conversational Forecasting Models"(Tran et al., 2025).

IMPORTANT NOTE: This implementation can, in fact, support any model compatible with `HuggingFace's AutoModelForSequenceClassification <https://huggingface.co/docs/transformers/v4.52.3/en/model_doc/auto#transformers.AutoModelForSequenceClassification>`_, including decoder-based models such as Gemma and LLaMA. However, we suggest using parameter-efficient fine-tuning (e.g., LoRA) techniques for large language models. To facilitate this, we provide a separate class specifically designed for decoder-based architectures.

.. automodule:: convokit.forecaster.TransformerEncoderModel
    :members:
