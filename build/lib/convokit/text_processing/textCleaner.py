from convokit.model import Corpus
from .textProcessor import TextProcessor
from typing import Callable, Optional
from cleantext import clean


clean_str = lambda s: clean(
    s,
    fix_unicode=True,  # fix various unicode errors
    to_ascii=True,  # transliterate to closest ASCII representation
    lower=True,  # lowercase text
    no_line_breaks=True,  # fully strip line breaks as opposed to only normalizing them
    no_urls=True,  # replace all URLs with a special token
    no_emails=True,  # replace all email addresses with a special token
    no_phone_numbers=True,  # replace all phone numbers with a special token
    no_numbers=True,  # replace all numbers with a special token
    no_digits=False,  # replace all digits with a special token
    no_currency_symbols=True,  # replace all currency symbols with a special token
    no_punct=False,  # fully remove punctuation
    replace_with_url="<URL>",
    replace_with_email="<EMAIL>",
    replace_with_phone_number="<PHONE>",
    replace_with_number="<NUMBER>",
    replace_with_digit="0",
    replace_with_currency_symbol="<CUR>",
    lang="en",
)


class TextCleaner(TextProcessor):
    """
    Transformer that cleans the text of utterances in an input Corpus. By default, the text cleaner assumes the
    text is in English. It fixes unicode errors, transliterates text to the closest ASCII representation,
    lowercases text, removes line breaks, and replaces URLs, emails, phone numbers, numbers, currency symbols with
    special tokens.

    This transformer can be configured with any custom text cleaning function that takes a text as input
    and outputs the cleaned version of the text.

    :param text_cleaner: an optional function for cleaning text. If unfilled, uses ConvoKit's default text cleaner
        as described above.
    :param input_field: name of attribute to use as input. This attribute must point to a string, and defaults to utterance.text.
    :param input_filter: a boolean function of signature `input_filter(utterance, aux_input)`.
        Text cleaning will only be applied to utterances where `input_filter` returns `True`.
        By default, will always return `True`, meaning that all utterances will be cleaned.
    :param verbosity: frequency of status messages
    :param replace_text: whether to replace the text being cleaned with the cleaned version. True by default.
        If False, the cleaned text is stored under attribute 'cleaned'.
    :param save_original: if replacing text, whether to save the original version of the text. If True, saves it
        under the 'original' attribute.
    """

    def __init__(
        self,
        text_cleaner: Optional[Callable[[str], str]] = None,
        input_field=None,
        input_filter=lambda utt, aux: True,
        verbosity: int = 100,
        replace_text: bool = True,
        save_original: bool = True,
    ):
        if replace_text:
            if save_original:
                output_field = "original"
            else:
                output_field = "cleaned_temp"
        else:
            output_field = "cleaned"
        self.replace_text = replace_text
        self.save_original = save_original
        proc_fn = text_cleaner if text_cleaner is not None else clean_str
        super().__init__(
            proc_fn=proc_fn,
            input_field=input_field,
            input_filter=input_filter,
            verbosity=verbosity,
            output_field=output_field,
        )

    def transform(self, corpus: Corpus) -> Corpus:
        super().transform(corpus)
        if self.replace_text:
            selector = lambda utt_: self.input_filter(utt_, None)
            for utt in corpus.iter_utterances(selector):
                cleaned_text = utt.retrieve_meta(self.output_field)
                if self.save_original:
                    utt.add_meta(self.output_field, utt.text)
                utt.text = cleaned_text

            if not self.save_original:
                corpus.delete_metadata("utterance", self.output_field)
        return corpus
