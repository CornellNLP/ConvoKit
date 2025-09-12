try:
    from .pivotal import *
except (ImportError, ModuleNotFoundError) as e:
    if "Unsloth GPU requirement not met" in str(e):
        raise ImportError(
            "Error from Unsloth: NotImplementedError: Unsloth currently only works on NVIDIA GPUs and Intel GPUs."
        ) from e
    elif (
        "not currently installed" in str(e)
        or "torch" in str(e)
        or "unsloth" in str(e)
        or "transformers" in str(e)
    ):
        raise ImportError(
            "Pivotal framework requires ML dependencies. Run 'pip install convokit[llm]' to install them."
        ) from e
    else:
        raise
