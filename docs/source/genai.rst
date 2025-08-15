GenAI
======

The GenAI module provides a unified interface for working with LLMs while doing conversational analysis in ConvoKit. The current implementation supports multiple providers including OpenAI GPT and Google Gemini, but is designed to be extensible to LLMs from other model providers and local models. This module makes it easy to integrate AI-powered text generation into your ConvoKit workflows for diverse tasks. The module handles API key management, response formatting, and provides consistent interfaces across different LLM providers.

Example usage: `GenAI module demo <https://github.com/CornellNLP/ConvoKit/blob/master/convokit/genai/example/example.ipynb>`_.

Overview
--------

The GenAI module consists of several key components:

* **LLMClient**: Abstract base class that defines the interface for all LLM clients
* **LLMResponse**: Unified response wrapper that standardizes output from different LLM providers
* **Factory Pattern**: Simple factory function to create appropriate client instances
* **Configuration Management**: Centralized API key and configuration management
* **Provider Clients**: Concrete implementations for different LLM providers (GPT, Gemini, Local)

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

* **OpenAI GPT**: Access to OpenAI GPT text models
* **Google Gemini**: Access to Google Gemini models

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

The GenAIConfigManager handles API key storage and retrieval. It supports:

* **File-based storage**: Configuration is stored in `~/.convokit/config.yml`
* **Environment variables**: API keys can be set via environment variables (e.g., `GPT_API_KEY`)
* **Secure storage**: API keys are stored locally and not exposed in code

