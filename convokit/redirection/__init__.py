from .redirection import *

try:
    from .likelihoodModel import *
except ImportError as e:
    if "not currently installed" in str(e):
        print(
            "LikelihoodModel requires ML dependencies. Run 'pip install convokit[ml]' to install them."
        )
    else:
        raise
