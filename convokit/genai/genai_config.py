import os
from pathlib import Path
import yaml
from typing import Optional

class GenAIConfigManager:
    """Manages configuration for GenAI clients, including setting and accessing API keys.
    
    Handles loading and saving of GenAI related configuration data, with support
    for environment variable overrides. Provides a centralized way to manage API keys
    and other configuration settings for different LLM providers.
    
    :param path: Path to the configuration file (default: ~/.convokit/config.yml)
    """
    def __init__(self, path: Optional[str] = None):
        if path is None:
            path = os.path.expanduser("~/.convokit/config.yml")
        self.path = Path(path)
        self._data = {}
        self._load()

    def _load(self):
        """Load configuration data from the YAML file.
        
        Creates the configuration file and directory if they don't exist.
        """
        if self.path.exists():
            self._data = yaml.safe_load(self.path.read_text()) or {}
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._data = {}
            self._save()
            
    def _save(self):
        """Save configuration data to the YAML file."""
        self.path.write_text(yaml.safe_dump(self._data))

    def set_api_key(self, provider: str, key: str):
        """Set an API key for a specific provider.
        
        :param provider: Name of the LLM provider (e.g., "gpt", "gemini")
        :param key: API key for the provider
        """
        self._data.setdefault('api_keys', {})[provider] = key
        self._save()

    def get_api_key(self, provider: str) -> Optional[str]:
        """Get the API key for a specific provider.
        
        First checks environment variables, then falls back to the configuration file.
        
        :param provider: Name of the LLM provider (e.g., "gpt", "gemini")
        :return: API key if found, None otherwise
        """
        env = os.getenv(f"{provider.upper()}_API_KEY")
        if env:
            return env
        return self._data.get('api_keys', {}).get(provider)
