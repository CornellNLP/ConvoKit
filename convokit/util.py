import json
import os
import shutil
import urllib.request
import uuid
import warnings
import zipfile
from typing import Dict
from .convokitConfig import ConvoKitConfig
import requests


# returns a path to the dataset file
def download(
    name: str,
    verbose: bool = True,
    data_dir: str = None,
    use_newest_version: bool = True,
    use_local: bool = False,
) -> str:
    """Use this to download (or use saved) convokit data by name.

    :param name: Which item to download. Currently supported:

        - "wiki-corpus": Wikipedia Talk Page Conversations Corpus
            A medium-size collection of conversations from Wikipedia editors' talk pages.
            (see http://www.cs.cornell.edu/~cristian/Echoes_of_power.html)
        - "wikiconv-<year>": Wikipedia Talk Page Conversations Corpus
            Conversations data for the specified year.
        - "supreme-corpus": Supreme Court Dialogs Corpus
            A collection of conversations from the U.S. Supreme Court Oral Arguments.
            (see http://www.cs.cornell.edu/~cristian/Echoes_of_power.html)
        - "parliament-corpus": UK Parliament Question-Answer Corpus
            Parliamentary question periods from May 1979 to December 2016
            (see http://www.cs.cornell.edu/~cristian/Asking_too_much.html)
        - "conversations-gone-awry-corpus": Wiki Personal Attacks Corpus
            Wikipedia talk page conversations that derail into personal attacks as labeled by crowdworkers
            (see http://www.cs.cornell.edu/~cristian/Conversations_gone_awry.html)
        - "conversations-gone-awry-cmv-corpus"
            Discussion threads on the subreddit ChangeMyView (CMV) that derail into rule-violating behavior
            (see http://www.cs.cornell.edu/~cristian/Conversations_gone_awry.html)
        -  "movie-corpus": Cornell Movie-Dialogs Corpus
            A large metadata-rich collection of fictional conversations extracted from raw movie scripts.
            (see https://www.cs.cornell.edu/~cristian/Chameleons_in_imagined_conversations.html)
        -  "tennis-corpus": Tennis post-match press conferences transcripts
            Transcripts for tennis singles post-match press conferences for major tournaments between 2007 to 2015
            (see http://www.cs.cornell.edu/~liye/tennis.html)
        -  "reddit-corpus-small": Reddit Corpus (sampled):
            A sample from 100 highly-active subreddits
        -  "subreddit-<subreddit-name>": Subreddit Corpus
            A corpus made from the given subreddit
        -  "friends-corpus": Friends TV show Corpus
            A collection of all the conversations that occurred over 10 seasons of Friends, a popular American TV sitcom
            that ran in the 1990s.
        -  "switchboard-corpus": Switchboard Dialog Act Corpus
             A collection of 1,155 five-minute telephone conversations between two participants,
            annotated with speech act tags.
        -  "persuasionforgood-corpus": Persuasion For Good Corpus
            A collection of online conversations where a persuader tries to convince a persuadee to donate to charity.
        -  "iq2-corpus": Intelligence Squared Debates Corpus
            Transcripts of debates held as part of Intelligence Squared Debates.
        -  "diplomacy-corpus": Deception in Diplomacy Corpus
            Dataset with intended and perceived deception labels in the negotiation-based game Diplomacy.
        -  "reddit-coarse-discourse-corpus": Coarse Discourse Sequence Corpus
            Reddit dataset with utterances containing discourse act labels.
        -  "chromium-corpus": Chromium Conversations Corpus
            A collection of almost 1.5 million conversations and 2.8 million comments posted by developers reviewing
            proposed code changes in the Chromium project.
        -  "wikipedia-politeness-corpus": Wikipedia Politeness Corpus
            A corpus of politeness annotations on requests from Wikipedia talk pages.
        -  "stack-exchange-politeness-corpus": Stack Exchange Politeness Corpus
            A corpus of politeness annotations on requests from stack exchange.
    :param verbose: Print checkpoint statements for download
    :param data_dir: Output path of downloaded file (default: ~/.convokit)
    :param use_newest_version: Re-download if new version is found
    :param use_local: if True, use the local version of corpus if it exists
        (regardless of whether a newer version exists)

    :return: The path to the downloaded item.
    """
    if use_local:
        return download_local(name, data_dir)

    dataset_config = requests.get(
        "https://raw.githubusercontent.com/CornellNLP/ConvoKit/master/download_config.json"
    ).json()

    cur_version = dataset_config["cur_version"]
    DatasetURLs = dataset_config["DatasetURLs"]
    ModelURLs = dataset_config["ModelURLS"]

    if name.startswith("subreddit"):
        subreddit_name = name.split("-", maxsplit=1)[1]
        # print(subreddit_name)
        cur_version[name] = cur_version["subreddit"]
        DatasetURLs[name] = get_subreddit_info(subreddit_name)
        # print(DatasetURLs[name])
    elif name.startswith("wikiconv"):
        wikiconv_year = name.split("-")[1]
        cur_version[name] = cur_version["wikiconv"]
        DatasetURLs[name] = _get_wikiconv_year_info(wikiconv_year)
    elif name.startswith("supreme-"):
        supreme_year = name.split("-")[1]
        cur_version[name] = cur_version["supreme"]
        DatasetURLs[name] = _get_supreme_info(supreme_year)
    else:
        name = name.lower()

    custom_data_dir = data_dir

    config = ConvoKitConfig()
    data_dir = config.data_directory
    data_dir = os.path.expanduser(data_dir)
    # pkg_resources.resource_filename("convokit", "")
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    if not os.path.exists(os.path.join(data_dir, "downloads")):
        os.mkdir(os.path.join(data_dir, "downloads"))

    dataset_path = os.path.join(data_dir, name)

    if custom_data_dir is not None:
        dataset_path = os.path.join(custom_data_dir, name)

    if not os.path.exists(os.path.dirname(dataset_path)):
        os.makedirs(os.path.dirname(dataset_path))

    dataset_path = os.path.realpath(dataset_path)

    needs_download = False
    downloadeds_path = os.path.join(data_dir, "downloads", "downloaded.txt")
    if not os.path.isfile(downloadeds_path):
        open(downloadeds_path, "w").close()
    with open(downloadeds_path, "r") as f:
        downloaded_lines = f.read().splitlines()
        downloaded = {}
        downloaded_paths = {}
        for l in downloaded_lines:
            dname, path, version = l.split("$#$")
            version = int(version)
            if dname not in downloaded or downloaded[dname] < version:
                downloaded[dname, path] = version
                downloaded_paths[dname] = path
                if custom_data_dir is None and name == dname:
                    dataset_path = os.path.join(path, name)

        # print(list(downloaded.keys()))
        if (name, os.path.dirname(dataset_path)) in downloaded:
            if (
                use_newest_version
                and name in cur_version
                and downloaded[name, os.path.dirname(dataset_path)] < cur_version[name]
            ):
                needs_download = True
        else:
            needs_download = True

    if needs_download:
        print("Downloading {} to {}".format(name, dataset_path))
        # name not in downloaded or \
        #    (use_newest_version and name in cur_version and
        #        downloaded[name] < cur_version[name]):
        if name in ModelURLs:
            for url in ModelURLs[name]:
                full_name = name + url[url.rfind("/") :]
                model_file_path = dataset_path + url[url.rfind("/") :]
                if not os.path.exists(os.path.dirname(model_file_path)):
                    os.makedirs(os.path.dirname(model_file_path))
                _download_helper(model_file_path, url, verbose, full_name, downloadeds_path)
        elif name.endswith("-motifs"):
            for url in DatasetURLs[name]:
                full_name = name + url[url.rfind("/") :]
                if full_name not in downloaded:
                    motif_file_path = dataset_path + url[url.rfind("/") :]
                    if not os.path.exists(os.path.dirname(motif_file_path)):
                        os.makedirs(os.path.dirname(motif_file_path))
                    _download_helper(motif_file_path, url, verbose, full_name, downloadeds_path)
        else:
            url = DatasetURLs[name]
            _download_helper(dataset_path, url, verbose, name, downloadeds_path)
    else:
        print("Dataset already exists at {}".format(dataset_path))
        dataset_path = os.path.join(downloaded_paths[name], name)

    return dataset_path


def download_local(name: str, data_dir: str):
    """
    Get path to a previously-downloaded local version of the corpus (which may be an older version).

    :param name: name of Corpus
    :return: string path to local Corpus
    """
    custom_data_dir = data_dir
    config = ConvoKitConfig()
    data_dir = config.data_directory

    # pkg_resources.resource_filename("convokit", "")
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            "No convokit data directory found. No local corpus version available."
        )

    if not os.path.exists(os.path.join(data_dir, "downloads")):
        raise FileNotFoundError(
            "Local convokit data directory found, but no downloads folder exists. No local corpus version available."
        )

    dataset_path = os.path.join(data_dir, "downloads", name)

    if custom_data_dir is not None:
        dataset_path = os.path.join(custom_data_dir, name)

    if not os.path.exists(os.path.dirname(dataset_path)):
        os.makedirs(os.path.dirname(dataset_path))

    dataset_path = os.path.realpath(dataset_path)

    downloadeds_path = os.path.join(data_dir, "downloads", "downloaded.txt")
    if not os.path.isfile(downloadeds_path):
        raise FileNotFoundError("downloaded.txt is missing.")
    with open(downloadeds_path, "r") as f:
        downloaded_lines = f.read().splitlines()
        downloaded = {}
        downloaded_paths = {}
        for l in downloaded_lines:
            dname, path, version = l.split("$#$")
            version = int(version)
            if dname not in downloaded or downloaded[dname] < version:
                downloaded[dname, path] = version
                downloaded_paths[dname] = path
                if custom_data_dir is None and name == dname:
                    dataset_path = os.path.join(path, name)

        # print(list(downloaded.keys()))
        if (name, os.path.dirname(dataset_path)) not in downloaded:
            raise FileNotFoundError("Could not find corpus in local directory.")

        print("Dataset already exists at {}".format(dataset_path))
        dataset_path = os.path.join(downloaded_paths[name], name)

    return dataset_path


def _download_helper(
    dataset_path: str, url: str, verbose: bool, name: str, downloadeds_path: str
) -> None:
    is_corpus = False
    if (
        url.lower().endswith(".corpus")
        or url.lower().endswith(".corpus.zip")
        or url.lower().endswith(".zip")
    ):
        dataset_path += ".zip"
        is_corpus = True

    with urllib.request.urlopen(url) as response, open(dataset_path, "wb") as out_file:
        if verbose:
            length = float(response.info()["Content-Length"])
            length = (
                str(round(length / 1e6, 1)) + "MB"
                if length > 1e6
                else str(round(length / 1e3, 1)) + "KB"
            )
            print("Downloading", name, "from", url, "(" + length + ")...", end=" ", flush=True)
        shutil.copyfileobj(response, out_file)

    # post-process (extract) corpora
    if name.startswith("subreddit"):
        with zipfile.ZipFile(dataset_path, "r") as zipf:
            corpus_dir = os.path.join(os.path.dirname(dataset_path), name)
            if not os.path.exists(corpus_dir):
                os.mkdir(corpus_dir)
            zipf.extractall(corpus_dir)

    elif url.lower().endswith(".corpus") or url.lower().endswith(".zip"):
        # print(dataset_path)
        with zipfile.ZipFile(dataset_path, "r") as zipf:
            zipf.extractall(os.path.dirname(dataset_path))

    if verbose:
        print("Done")
    # for Corpus objects only: check the Corpus version
    if is_corpus:
        with open(downloadeds_path, "a") as f:
            fn = os.path.join(
                os.path.dirname(dataset_path), name
            )  # os.path.join(os.path.dirname(data), name)
            f.write(
                "{}$#${}$#${}\n".format(
                    name, os.path.realpath(os.path.dirname(dataset_path) + "/"), corpus_version(fn)
                )
            )
            # f.write(name + "\n")


def corpus_version(filename: str) -> int:
    with open(os.path.join(filename, "index.json")) as f:
        d = json.load(f)
        return int(d["version"])


# retrieve grouping and completes the download link for subreddit
def get_subreddit_info(subreddit_name: str) -> str:
    # base directory of subreddit corpuses
    subreddit_base = "http://zissou.infosci.cornell.edu/convokit/datasets/subreddit-corpus/"
    data_dir = subreddit_base + "corpus-zipped/"

    groupings_url = subreddit_base + "subreddit-groupings.txt"
    groups_fetched = urllib.request.urlopen(groupings_url)

    groups = [line.decode("utf-8").strip("\n") for line in groups_fetched]

    for group in groups:
        if subreddit_in_grouping(subreddit_name, group):
            # return os.path.join(data_dir, group, subreddit_name + ".corpus.zip")
            return data_dir + group + "/" + subreddit_name + ".corpus.zip"

    print("The subreddit requested is not available.")

    return ""


def subreddit_in_grouping(subreddit: str, grouping_key: str) -> bool:
    """
    :param subreddit: subreddit name
    :param grouping_key: example: "askreddit~-~blackburn"
    :return: if string is within the grouping range
    """
    bounds = grouping_key.split("~-~")
    if len(bounds) == 1:
        print(subreddit, grouping_key)
    return bounds[0] <= subreddit <= bounds[1]


def _get_wikiconv_year_info(year: str) -> str:
    """completes the download link for wikiconv"""

    # base directory of wikicon corpuses
    wikiconv_base = "http://zissou.infosci.cornell.edu/convokit/datasets/wikiconv-corpus/"
    data_dir = wikiconv_base + "corpus-zipped/"

    return data_dir + year + "/full.corpus.zip"


def _get_supreme_info(year: str) -> str:
    supreme_base = "http://zissou.infosci.cornell.edu/convokit/datasets/supreme-corpus/"
    return supreme_base + "supreme-" + year + ".zip"


def meta_index(corpus=None, filename: str = None) -> Dict:
    keys = ["utterances-index", "conversations-index", "speakers-index", "overall-index"]
    if corpus is not None:
        return {k: v for k, v in corpus.meta_index.items() if k in keys}
    if filename is not None:
        with open(os.path.join(filename, "index.json")) as f:
            d = json.load(f)
            return d


def warn(text: str):
    """
    Pre-pends a red-colored 'WARNING: ' to [text]. This is a printed warning and cannot be suppressed.

    :param text: Warning message
    :return: 'WARNING: [text]'
    """
    print("\033[91m" + "WARNING: " + "\033[0m" + text)


def _deprecation_format(message, category, filename, lineno, file=None, line=None):
    return "{}:{}: {}: {}\n".format(filename, lineno, category.__name__, message)


def deprecation(prev_name: str, new_name: str, stacklevel: int = 3):
    """
    Suppressable deprecation warning.
    """
    warnings.formatwarning = _deprecation_format
    warnings.warn(
        "{} is deprecated and will be removed in a future release. "
        "Use {} instead.".format(prev_name, new_name),
        category=FutureWarning,
        stacklevel=stacklevel,
    )


def create_safe_id():
    return "_" + uuid.uuid4().hex
