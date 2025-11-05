GenAI
======

The GenAI module provides a unified interface for working with LLMs while doing conversational analysis in ConvoKit. The current implementation supports multiple providers including OpenAI GPT and Google Gemini, but is designed to be extensible to LLMs from other model providers and local models. This module makes it easy to integrate AI-powered text generation into your ConvoKit workflows for diverse tasks. The module handles API key management, response formatting, and provides consistent interfaces across different LLM providers.

The module includes ConvoKit transformers that allow you to apply LLM processing directly to corpus objects at different levels (utterances, conversations, speakers, or entire corpus), making it seamless to integrate AI analysis into your conversational data processing pipelines.

Example usage: `GenAI module demo <https://github.com/CornellNLP/ConvoKit/blob/master/convokit/genai/example/example.ipynb>`_.

Overview
--------

The GenAI module consists of several key components:

* **LLMClient**: Abstract base class that defines the interface for all LLM clients
* **LLMResponse**: Unified response wrapper that standardizes output from different LLM providers
* **Factory Pattern**: Simple factory function to create appropriate client instances
* **Configuration Management**: Centralized API key and configuration management
* **Provider Clients**: Concrete implementations for different LLM providers (GPT, Gemini, Local)
* **LLMPromptTransformer**: Flexible ConvoKit transformer that applies custom LLM prompts to corpus objects at any level

Basic Interface and Configuration
---------------------------------

.. automodule:: convokit.genai.base
    :members:

.. automodule:: convokit.genai.genai_config
    :members:

.. automodule:: convokit.genai.factory
    :members:

Provider Clients
----------------

Supported Providers
^^^^^^^^^^^^^^^^^^^

Currently supported LLM providers:

* **OpenAI GPT**: Access to OpenAI GPT models through the OpenAI API. See `OpenAI API setup <https://platform.openai.com/docs/quickstart>`_.
* **Google Gemini**: Access to Google Gemini models via Vertex AI. See `Vertex AI setup guide <https://cloud.google.com/vertex-ai/docs/start/cloud-environment>`_.
* **Local Models**: Template implementation for local LLM models (requires custom implementation)

GPT Client
^^^^^^^^^^

.. automodule:: convokit.genai.gpt_client
    :members:

Gemini Client
^^^^^^^^^^^^^

.. automodule:: convokit.genai.gemini_client
    :members:

Local Client
^^^^^^^^^^^^

The LocalClient provides a template implementation for integrating local LLM models. The current implementation returns mock responses and serves as a starting point for implementing actual local model support.

.. automodule:: convokit.genai.local_client
    :members:

Adding New Providers
^^^^^^^^^^^^^^^^^^^^

To add support for a new LLM provider:

1. Create a new client class that inherits from `LLMClient`
2. Update the configuration manager to support the new provider
3. Implement the required `generate()` method and optionally `stream()` method if applicable
4. Add the provider to the factory function in `factory.py`

Configuration
-------------

The GenAIConfigManager handles API key storage and retrieval for different LLM providers. It supports:

* **File-based storage**: Configuration is stored in `~/.convokit/config.yml`
* **Environment variables**: API keys can be set via environment variables (e.g., `GPT_API_KEY`)
* **Secure storage**: API keys are stored locally and not exposed in code
* **Provider-specific settings**: Support for different configuration requirements per provider (e.g., Google Cloud project settings for Gemini)

**Basic Usage:**

.. code-block:: python

    from convokit.genai.genai_config import GenAIConfigManager
    
    config = GenAIConfigManager()
    
    # Set OpenAI API key
    config.set_api_key("gpt", "your-openai-api-key")
    
    # Set Google Cloud configuration for Gemini
    config.set_google_cloud_config("your-project-id", "your-location")
    
    # Configuration is automatically saved and can be reused

LLMPromptTransformer
--------------------

The LLMPromptTransformer is a flexible and powerful ConvoKit transformer that allows you to apply custom LLM prompts to corpus objects at different levels (utterances, conversations, speakers, or the entire corpus). It provides fine-grained control over how objects are formatted for LLM processing and where the results are stored.

.. toctree::
    LLMPromptTransformer <llmprompttransformer.rst>