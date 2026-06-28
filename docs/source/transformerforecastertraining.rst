Transformer Forecaster Configuration
========================================

``TransformerForecasterConfig`` is a dataclass that holds training and runtime settings for transformer-based
``ForecasterModel`` implementations, including :doc:`transformerencodermodel` and :doc:`transformerdecodermodel`.
Pass a config when constructing the model; the model stores it on ``self.config`` and uses it during ``fit()``,
``transform()``, checkpoint loading, and inference.

Install LLM dependencies first: ``pip install convokit[llm]``.

Usage
-----

``output_dir`` is the only required field. Other hyperparameters have defaults:

.. code-block:: python

   from convokit.forecaster import TransformerForecasterConfig, TransformerEncoderModel

   config = TransformerForecasterConfig(
       output_dir="YOUR_SAVING_DIRECTORY",
       per_device_batch_size=4,
       gradient_accumulation_steps=1,
       num_train_epochs=4,
       learning_rate=6.7e-6,
       random_seed=1,
       context_mode="normal",
       device="cuda",
   )
   model = TransformerEncoderModel("bert-base-uncased", config=config)

For inference-only runs (loading an existing checkpoint), you can pass a minimal config with ``output_dir``,
``context_mode``, and ``device``; training hyperparameters are ignored.

Configuration fields
--------------------

``output_dir``
  Directory for checkpoints, prediction CSVs, and training logs. Created automatically if it does not exist.
  Encoder models write ``test_predictions.csv`` and ``val_predictions.csv`` here; decoder models write
  ``predictions.csv``. Checkpoints are subfolders named ``checkpoint-*``.

``per_device_batch_size``
  Batch size per GPU during training. Effective batch size is
  ``per_device_batch_size * gradient_accumulation_steps`` (times the number of GPUs if using distributed training).

``gradient_accumulation_steps``
  Accumulate gradients over this many steps before an optimizer update. Useful when GPU memory limits batch size;
  decoder models often use a larger value (e.g. 32) to simulate a larger batch.

``num_train_epochs``
  Number of passes over the training data.

``learning_rate``
  Optimizer learning rate. Encoder and decoder models ship with different class-level defaults (see below).

``random_seed``
  Seed passed to the Hugging Face ``Trainer`` (encoder models). Improves reproducibility across runs.

``device``
  Where tensors are placed at inference time, e.g. ``"cuda"``, ``"cuda:0"``, or ``"cpu"``.

``context_mode``
  How conversational input is built for each context tuple:

  * ``"normal"`` — include prior utterances plus the current utterance (default).
  * ``"no-context"`` — use only the current utterance, ablating conversational history.

  Must be one of these two values; anything else raises ``ValueError``.

Model defaults
--------------

If you omit ``config``, each model class uses its own ``DEFAULT_CONFIG``:

**TransformerEncoderModel** — ``per_device_batch_size=4``, ``gradient_accumulation_steps=1``,
``num_train_epochs=1``, ``learning_rate=6.7e-6``.

**TransformerDecoderModel** — ``per_device_batch_size=2``, ``gradient_accumulation_steps=32``,
``num_train_epochs=1``, ``learning_rate=1e-4``.

These defaults are starting points; demos and benchmarks often override them (see
`Transformer Forecaster demo <https://github.com/CornellNLP/ConvoKit/blob/master/examples/forecaster/Transformer%20Forecaster%20demo.ipynb>`_).

See also
--------

* :doc:`forecaster` — Forecaster wrapper and decision policies
* :doc:`transformerencodermodel` — BERT-style encoder forecasters
* :doc:`transformerdecodermodel` — LLM decoder forecasters (Llama, Gemma, etc.)

.. automodule:: convokit.forecaster.TransformerForecasterConfig
    :members:
