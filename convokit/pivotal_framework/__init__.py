try:
    from .pivotal import *
except ImportError as e:
    if "Unsloth GPU requirement not met" in str(e):
        print(
            "Error from Unsloth: NotImplementedError: Unsloth currently only works on NVIDIA GPUs and Intel GPUs."
        )
    elif "not currently installed" in str(e):
        print(
            "Pivotal framework requires ML dependencies. Run 'pip install convokit[llm]' to install them."
        )
    else:
        raise
