from abc import ABC, abstractmethod


class Perplexity(ABC):
    """

    :param perplexity_type:
    :param kwargs:
    """

    def __init__(self, perplexity_type='', **kwargs):
        self._perplexity_type = perplexity_type
        self.__dict__.update((f'_{arg}', value) for arg, value in kwargs.items())

    @property
    def type(self):
        """

        :return:
        """
        return self._perplexity_type

    @property
    def config(self):
        """

        :return:
        """
        private_var_prefix = f'_{self.__class__.__name__}'
        return {arg[1:]: value for arg, value in self.__dict__.items() if not arg.startswith(private_var_prefix)}

    def overwrite_args(self, args_to_overwrite, kwargs):
        """

        :param args_to_overwrite:
        :param kwargs:
        :return:
        """
        for arg in args_to_overwrite:
            self.__dict__[f'_{arg}'] = kwargs[arg] if arg in kwargs else self.__dict__[f'_{arg}']

    @abstractmethod
    def perplexity_fn(self, target_samples, context_samples, **kwargs):
        """

        :param target_samples:
        :param context_samples:
        :param kwargs:
        :return:
        """
        raise NotImplementedError('the subclass needs to implement it\'s own perplexity function')
