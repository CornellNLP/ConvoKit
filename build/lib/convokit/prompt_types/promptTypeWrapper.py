import os

from convokit.convokitPipeline import ConvokitPipeline
from convokit.model import Utterance
from convokit.phrasing_motifs import CensorNouns, QuestionSentences, PhrasingMotifs
from convokit.prompt_types import PromptTypes
from convokit.text_processing import TextParser, TextToArcs
from convokit.transformer import Transformer


class PromptTypeWrapper(Transformer):
    """
    This is a wrapper class implementing a pipeline that infers types of rhetorical intentions encapsulated by utterances in a corpus, in terms of their anticipated responses.

    The pipeline involves:
            * parsing input text via `TextParser`
            * representing input text as dependency tree arcs, with nouns censored out, via `CensorNouns`, `TextToArcs` and `QuestionSentences`
            * extracting a set of "phrasings" from the corpus, using a `PhrasingMotifs` model
            * inferring prompt types and type assignments per-utterance, using a `PromptTypes` model.

    While the pipeline computes many attributes of an utterance along the way, the overall goal is to assign each utterance to a prompt type.
    By default, the pipeline will focus on learning types of *questions*, in terms of how the questions are phrased. However, other options are possible (see parameters below).
    For further details, see the respective classes listed above.

    :param output_field:  the name of the attribute to write to in the transform step. the transformer outputs several fields, corresponding to both vector representations and discrete type assignments.
    :param n_types: the number of prompt types to infer.
    :param use_prompt_motifs: whether to represent prompts in terms of how they are phrased. defaults to `True`. if `False`, will use individual dependency arcs as input (this might be better for noisier text)
    :param root_only: whether to only use dependency arcs attached to the root of the parse. defaults to `True`. if `False` will also consider arcs beyond the root (may be better for noisier text)
    :param questions_only: whether to only learn representations of questions (i.e., utterances containing sentences that end in question marks); defaults to `True`.
    :param enforce_caps: whether to only fit and transform on sentences that start with capital letters. defaults to `True`, which is appropriate for formal settings like transcripts of institutional proceedings, where this is a check on how well-formed the input is. in less formal settings like social media, setting to `False` may be more appropriately permissive.
    :param min_support: the minimum frequency of phrasings to extract.
    :param min_df: the minimum frequency of prompt and response terms to consider when inferring types.
    :param max_df: the maximum frequency of prompt and response terms to use. defaults to 0.1 (i.e., occurs in at most 10% of prompt-response pairs). Setting higher is more permissive, but may result in many stopword-like terms adding noise to the model.
    :param svd__n_components: the number of SVD dimensions to use when inferring types, defaults to 25. higher values result in richer vector representations, perhaps at the cost of the model learning overly-specific types.
    :param max_dist: the maximum distance between a vector representation of an utterance and the cluster centroid; a cluster whose distance to all centroids is above this cutoff will get assigned to a null type, denoted by -1. defaults to 0.9.
    :param recompute_all: if `False` (the default), checks utterances to see if they already have an attribute computed, skipping over that utterance in the relevant step of the pipeline. if `True`, recomputes all attributes.
    :param random_state: the random seed to use.
    :param verbosity: frequency of status messages.

    """

    def __init__(
        self,
        output_field="prompt_types",
        n_types=8,
        use_prompt_motifs=True,
        root_only=True,
        questions_only=True,
        enforce_caps=True,
        recompute_all=False,
        min_support=100,
        min_df=100,
        svd__n_components=25,
        max_df=0.1,
        max_dist=0.9,
        random_state=None,
        verbosity=10000,
    ):
        self.use_motifs = use_prompt_motifs
        self.random_state = random_state
        pipe = [
            (
                "parser",
                TextParser(
                    verbosity=verbosity,
                    input_filter=lambda utt, aux: recompute_all
                    or (utt.retrieve_meta("parsed") is None),
                ),
            ),
            (
                "censor_nouns",
                CensorNouns(
                    "parsed_censored",
                    input_filter=lambda utt, aux: recompute_all
                    or (utt.retrieve_meta("parsed_censored") is None),
                    verbosity=verbosity,
                ),
            ),
            (
                "shallow_arcs",
                TextToArcs(
                    "arcs",
                    input_field="parsed_censored",
                    input_filter=lambda utt, aux: recompute_all
                    or (utt.retrieve_meta("arcs") is None),
                    root_only=root_only,
                    verbosity=verbosity,
                ),
            ),
        ]

        if questions_only:
            pipe.append(
                (
                    "question_sentence_filter",
                    QuestionSentences(
                        "question_arcs",
                        input_field="arcs",
                        input_filter=lambda utt, aux: recompute_all or utt.meta["is_question"],
                        use_caps=enforce_caps,
                        verbosity=verbosity,
                    ),
                )
            )

            prompt_input_field = "question_arcs"
            self.prompt_selector = lambda utt: utt.meta["is_question"]
            self.reference_selector = lambda utt: (not utt.meta["is_question"]) and (
                utt.reply_to is not None
            )
        else:
            prompt_input_field = "arcs"
            self.prompt_selector = lambda utt: True
            self.reference_selector = lambda utt: True
        if use_prompt_motifs:
            pipe.append(
                (
                    "pm_model",
                    PhrasingMotifs(
                        "motifs",
                        prompt_input_field,
                        min_support=min_support,
                        fit_filter=self.prompt_selector,
                        verbosity=verbosity,
                    ),
                )
            )
            prompt_field = "motifs"
            prompt_transform_field = "motifs__sink"
        else:
            prompt_field = "arcs"
            prompt_transform_field = "arcs"
        pipe.append(
            (
                "pt_model",
                PromptTypes(
                    prompt_field=prompt_field,
                    reference_field="arcs",
                    prompt_transform_field=prompt_transform_field,
                    output_field=output_field,
                    n_types=n_types,
                    svd__n_components=svd__n_components,
                    prompt__tfidf_min_df=min_df,
                    prompt__tfidf_max_df=max_df,
                    reference__tfidf_min_df=min_df,
                    reference__tfidf_max_df=max_df,
                    max_dist=max_dist,
                    random_state=random_state,
                    verbosity=verbosity,
                ),
            )
        )
        self.pipe = ConvokitPipeline(pipe)

    def fit(self, corpus, y=None):
        """
        Fits the model for a corpus -- that is, computes all necessary utterance attributes, and fits the underlying `PhrasingMotifs` and `PromptTypes` models.

        :param corpus: Corpus
        :return: None
        """

        self.pipe.fit(
            corpus,
            pt_model__prompt_selector=self.prompt_selector,
            pt_model__reference_selector=self.reference_selector,
        )

    def transform(self, corpus):
        """
        Computes prompt type assignments for utterances in a corpus.

        :param corpus: Corpus
        :return: the corpus, with per-utterance representations and type assignments.
        """

        return self.pipe.transform(corpus)

    def transform_utterance(self, utterance):
        """
        Computes prompt type assignments for individual utterances. can take as input ConvoKit Utterances or raw strings. will return assignments for *all* string input, even if the input is not a question.

        :param utterance: the utterance, as an Utterance or string.
        :return: the utterance, annotated with type assignments.
        """

        if isinstance(utterance, str):
            utterance = Utterance(text=utterance)
            utterance.meta["is_question"] = True
        return self.pipe.transform_utterance(utterance)

    def dump_model(self, model_dir, type_keys="default"):
        """
        Writes the `PhrasingMotifs` (if applicable) and `PromptTypes` models to disk.

        :param model_dir: directory to write to.
        :return: None
        """
        try:
            os.mkdir(model_dir)
        except:
            pass
        if self.use_motifs:
            self.pipe.named_steps["pm_model"].dump_model(os.path.join(model_dir, "pm_model"))
        self.pipe.named_steps["pt_model"].dump_model(
            os.path.join(model_dir, "pt_model"), type_keys=type_keys
        )

    def load_model(self, model_dir, type_keys="default"):
        """
        Reads the `PhrasingMotifs` (if applicable) and `PromptTypes` models from disk.

        :param model_dir: directory to read from.
        :return: None
        """

        if self.use_motifs:
            self.pipe.named_steps["pm_model"].load_model(os.path.join(model_dir, "pm_model"))
        self.pipe.named_steps["pt_model"].load_model(
            os.path.join(model_dir, "pt_model"), type_keys=type_keys
        )

    def get_model(self, type_keys="default"):
        """
        Returns the model:
                * pm_model: PhrasingMotifs model (if applicable, i.e., use_motifs=True)
                * pt_model: PromptTypes model

        :param type_keys: which numbers of prompt types to return corresponding PromptTypes model for
        :return: model
        """
        to_return = {}
        if self.use_motifs:
            to_return["pm_model"] = self.pipe.named_steps["pm_model"].get_model()
        to_return["pt_model"] = self.pipe.named_steps["pt_model"].get_model(type_keys=type_keys)
        return to_return

    def print_top_phrasings(self, k):
        """
        prints the k most frequent phrasings from the `PhrasingMotifs` component of the pipeline, if phrasings are used.

        :param k: number of phrasings to print
        :return: None
        """

        if self.use_motifs:
            self.pipe.named_steps["pm_model"].print_top_phrasings(k)
        else:
            print("phrasing motifs unavailable")

    def display_type(self, type_id, corpus=None, type_key=None, k=10):
        """
        for a particular prompt type, displays the representative prompt and response terms. can also display representative prompt and response utterances.

        :param type_id: ID of the prompt type to display.
        :param corpus: pass in the training corpus to also display representative utterances.
        :param type_key: the name of the prompt type clustering model to use. defaults to `n_types` that the model was initialized with, but if `refit_types` is called with different number of types, can be modified to display this updated model as well.
        :param k: the number of sample terms (or utteranceS) to display.
        :return: None

        """
        self.pipe.named_steps["pt_model"].display_type(
            type_id, corpus=corpus, type_key=type_key, k=k
        )

    def summarize(self, corpus, type_ids=None, type_key=None, k=10):
        """
        Displays representative prompt and response terms and utterances for each type learned.

        :param corpus: corpus to display utterances for (must have `transform()` called on it)
        :param type_ids: ID of the prompt type to display. if None, will display all types.
        :param type_key: the name of the prompt type clustering model to use. defaults to `n_types` that the model was initialized with, but if `refit_types` is called with different number of types, can be modified to display this updated model as well.
        :param k: the number of sample terms (or utteranceS) to display.
        :return: None
        """
        self.pipe.named_steps["pt_model"].summarize(
            corpus=corpus, type_ids=type_ids, type_key=type_key, k=k
        )

    def refit_types(self, n_types, random_state=None, name=None):
        """
        infers a different number of prompt types than was originally called.

        :param n_types: number of types to learn
        :param random_state: random seed
        :param name: the name of the new type model. defaults to n_types.
        :return: None
        """
        self.pipe.named_steps["pt_model"].refit_types(n_types, random_state, name)
