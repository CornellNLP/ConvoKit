import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from collections import defaultdict

from convokit.transformer import Transformer
from convokit.model import Corpus
from typing import Optional


class CommunityEmbedder(Transformer):
    """
    Must be run after threadEmbedder.fit_transform()

    Groups threads together into communities
    in this space for visualization or other such purposes.

    :param community_key: Key in "meta" dictionary of each utterance
            whose corresponding value we'll use as the community label for that
            utterance (see threadEmbedder)
    :param n_components: Number of dimensions to embed communities into
    :param method: Embedding method; "svd", "tsne" or "none"
    """

    def __init__(
        self, community_key: Optional[str] = None, n_components: int = 2, method: str = "none"
    ):
        self.community_key = community_key
        self.n_components = n_components
        self.method = method

    def transform(self, corpus: Corpus) -> Corpus:
        """
        Same as fit_transform()
        """
        return self.fit_transform(corpus)

    def fit_transform(self, corpus: Corpus) -> Corpus:
        """
        Groups threads together into communities.

        :param corpus: the Corpus to use

        :return: Modifies and returns Corpus with new meta key: "communityEmbedder", value: Dict,
                containing "pts": an array with rows corresponding to embedded communities,
                and "labels": an array whose ith entry is the community of the ith row of X.
        """
        if self.community_key is None:
            raise RuntimeError(
                "Must specify community_key to retrieve label information from utterance"
            )

        corpus_meta = corpus.get_meta()
        if "threadEmbedder" not in corpus_meta:
            raise RuntimeError(
                "Missing threadEmbedder metadata: "
                "threadEmbedder.fit_transform() must be run on the Corpus first"
            )

        thread_embed_data = corpus_meta["threadEmbedder"]

        X_mid = thread_embed_data["X"]
        roots = thread_embed_data["roots"]

        if self.method.lower() == "svd":
            f = TruncatedSVD
        elif self.method.lower() == "tsne":
            f = TSNE
        elif self.method.lower() == "none":
            f = None
        else:
            raise Exception("Invalid embed_communities embedding method")

        if f is not None:
            X_embedded = f(n_components=self.n_components).fit_transform(X_mid)
        else:
            X_embedded = X_mid

        labels = [corpus.get_utterance(root).meta[self.community_key] for root in roots]
        # label_counts = Counter(labels)
        subs = defaultdict(list)
        for x, label in zip(X_embedded, labels):
            subs[label].append(x / np.linalg.norm(x))

        labels, subs = zip(*subs.items())
        pts = [np.mean(sub, axis=0) for sub in subs]

        retval = {"pts": pts, "labels": labels}
        corpus.add_meta("communityEmbedder", retval)

        return corpus
