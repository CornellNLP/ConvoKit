import os
import sys
from unittest import TestLoader, TextTestRunner


def setup_test_environment():
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    convokit_root = os.path.dirname(tests_dir)

    # Add both directories to Python path
    paths_to_add = [convokit_root, tests_dir]

    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)

    import types

    if "convokit.tests" not in sys.modules:
        tests_module = types.ModuleType("convokit.tests")
        tests_module.__path__ = [tests_dir]
        sys.modules["convokit.tests"] = tests_module

    test_subdirs = [
        "general",
        "general.binary_data",
        "general.fill_missing_convo_ids",
        "general.from_pandas",
        "general.load_and_dump_corpora",
        "general.merge_corpus",
        "general.metadata_operations",
        "general.traverse_convo",
        "bag_of_words",
        "phrasing_motifs",
        "politeness_strategies",
        "text_processing",
    ]

    for subdir in test_subdirs:
        module_name = f"convokit.tests.{subdir}"
        if module_name not in sys.modules:
            subdir_path = os.path.join(tests_dir, subdir.replace(".", os.sep))
            if os.path.exists(subdir_path):
                submodule = types.ModuleType(module_name)
                submodule.__path__ = [subdir_path]
                sys.modules[module_name] = submodule


def discover_tests_safely():
    setup_test_environment()

    loader = TestLoader()
    tests_dir = os.path.dirname(os.path.abspath(__file__))

    # Discover tests only in the tests directory
    tests = loader.discover(
        start_dir=tests_dir, pattern="test_*.py", top_level_dir=os.path.dirname(tests_dir)
    )

    return tests


if __name__ == "__main__":
    tests = discover_tests_safely()
    testRunner = TextTestRunner()
    test_results = testRunner.run(tests)

    if test_results.wasSuccessful():
        exit(0)
    else:
        exit(1)
