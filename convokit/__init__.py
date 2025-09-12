import warnings
from typing import Any

# Core modules - always imported immediately
from .model import *
from .util import *
from .transformer import *
from .convokitConfig import *
from .convokitPipeline import *

# Module mapping for lazy loading
# Each entry maps module_name -> import_path
_LAZY_MODULES = {
    'coordination': '.coordination',
    'politenessStrategies': '.politenessStrategies', 
    'hyperconvo': '.hyperconvo',
    'speakerConvoDiversity': '.speakerConvoDiversity',
    'text_processing': '.text_processing',
    'phrasing_motifs': '.phrasing_motifs',
    'prompt_types': '.prompt_types',
    'classifier': '.classifier',
    'ranker': '.ranker',
    'forecaster': '.forecaster',
    'fighting_words': '.fighting_words',
    'paired_prediction': '.paired_prediction',
    'bag_of_words': '.bag_of_words',
    'expected_context_framework': '.expected_context_framework',
    'surprise': '.surprise',
    'redirection': '.redirection',
    'pivotal_framework': '.pivotal_framework',
    'utterance_simulator': '.utterance_simulator',
    'utterance_likelihood': '.utterance_likelihood',
    'speaker_convo_helpers': '.speaker_convo_helpers',
    'politeness_collections': '.politeness_collections',
}

# Cache for loaded modules
_loaded_modules = {}

def _lazy_import(module_name: str) -> Any:
    """Import a module lazily and cache the result."""
    if module_name in _loaded_modules:
        return _loaded_modules[module_name]
    
    if module_name not in _LAZY_MODULES:
        raise AttributeError(f"module '{__name__}' has no attribute '{module_name}'")
    
    import_path = _LAZY_MODULES[module_name]
    
    try:
        import importlib
        module = importlib.import_module(import_path, package=__name__)
        _loaded_modules[module_name] = module
        
        globals_dict = globals()
        if hasattr(module, '__all__'):
            for name in module.__all__:
                if hasattr(module, name):
                    globals_dict[name] = getattr(module, name)
        else:
            for name in dir(module):
                if not name.startswith('_'):
                    globals_dict[name] = getattr(module, name)
        
        return module
        
    except Exception as e:
        # Simply re-raise whatever the module throws
        # Let each module handle its own error messaging
        raise

def __getattr__(name: str) -> Any:
    """Handle attribute access for lazy-loaded modules."""
    # Check if it's a module we can lazy load
    if name in _LAZY_MODULES:
        return _lazy_import(name)
    
    # Check if it's an exported symbol from a lazy module
    # We need to check each module to see if it exports this symbol
    for module_name in _LAZY_MODULES:
        if module_name not in _loaded_modules:
            # Try to import the module to see if it has the requested attribute
            try:
                import importlib
                import_path = _LAZY_MODULES[module_name]
                module = importlib.import_module(import_path, package=__name__)
                
                # Check if this module has the requested attribute
                if hasattr(module, name):
                    # Import the full module (which will add all symbols to globals)
                    _lazy_import(module_name)
                    # Return the requested attribute
                    return getattr(module, name)
                    
            except Exception:
                # If module fails to import, just skip it and try next module
                # The module's own error handling will take care of proper error messages
                continue
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

