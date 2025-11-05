LLMPromptTransformer
====================

The LLMPromptTransformer is a flexible ConvoKit transformer that allows you to apply custom LLM prompts to corpus objects at different levels (utterances, conversations, speakers, or the entire corpus). It provides fine-grained control over how objects are formatted for LLM processing and where the results are stored as metadata.

This transformer is part of the GenAI module (see :doc:`GenAI <genai>`) and integrates seamlessly with the GenAI client infrastructure to support multiple LLM providers (OpenAI GPT, Google Gemini, and local models).

Example usage: `GenAI module demo <https://github.com/CornellNLP/ConvoKit/blob/master/convokit/genai/example/example.ipynb>`_.

.. automodule:: convokit.genai.llmprompttransformer
    :members: