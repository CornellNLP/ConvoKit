"""
Handles merging portion of conversion separately. Runs in parallel on each year to speed up processing.
"""

import sys

sys.path.insert(0, "/home/jonathan/research/Cornell-Conversational-Analysis-Toolkit")
from convokit import Corpus
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial


def main():
    data_directory_intermediate = "/kitchen/wikiconv-convokit-processing/store_test_merging/"  # intermediate directory where the split Convokit files are kept
    data_directory_output = "/kitchen/wikiconv-convokit-processing/final/English/"  # directory to output the merged Convokit files
    delete_intermediate_files = True  # set to True to delete intermediate files after merging
    max_workers = 12  # number of threads/years to run in parallel

    print("Starting merge process...")
    print(f"Reading from: {data_directory_intermediate}")
    print(f"Writing to: {data_directory_output}")
    print(f"Using {max_workers} parallel workers")

    os.makedirs(data_directory_output, exist_ok=True)

    merge_files(data_directory_output, data_directory_intermediate, max_workers)

    print("\nMerge completed successfully!")

    if delete_intermediate_files:
        print(f"Deleting intermediate files from {data_directory_intermediate}")
        shutil.rmtree(data_directory_intermediate)
        print("Intermediate files deleted.")


def merge_files(final_directory, input_directory, max_workers):
    # build full list
    input_subdirectory_paths = [x[0] for x in os.walk(input_directory)]

    # organize files by year
    files_by_year = {}
    for year_x in range(2006, 2021):
        year_str = str(year_x)
        files_by_year[year_str] = [path for path in input_subdirectory_paths if year_str in path]

    # years in parallel
    process_year_func = partial(process_single_year, final_directory=final_directory)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_year = {}
        for year_x in range(2007, 2019):
            year_str = str(year_x)
            if len(files_by_year[year_str]) > 0:
                future = executor.submit(process_year_func, year_str, files_by_year[year_str])
                future_to_year[future] = year_str

        # process results as they complete
        for future in as_completed(future_to_year):
            year = future_to_year[future]
            try:
                result = future.result()
                print(f"\n✓ Completed year {year}: {result}")
            except Exception as e:
                print(f"\n✗ Error processing year {year}: {e}")


def process_single_year(year, paths_lst, final_directory):
    """process a single year"""
    if len(paths_lst) == 0:
        return f"Skipped - no files"

    print(f"\n[Year {year}] Processing {len(paths_lst)} corpus file(s)")

    if len(paths_lst) == 1:
        print(f"[Year {year}] Loading single corpus")
        corpus_1 = Corpus(filename=paths_lst[0])
        output_path = final_directory + "wikiconv_corpus_merged_" + year
        corpus_1.dump(output_path)
        return f"Saved single corpus"

    else:
        print(f"[Year {year}] Merging {len(paths_lst)} corpus files")

        # load all corpora
        corpora = []
        for idx, path in enumerate(paths_lst, start=1):
            print(f"[Year {year}] Loading corpus {idx}/{len(paths_lst)}")
            corpora.append(Corpus(filename=path))

        print(f"[Year {year}] Starting merge of {len(corpora)} corpora")

        # merge in a balanced binary tree pattern for increased efficiency
        round_num = 1
        while len(corpora) > 1:
            next_round = []
            pairs = (len(corpora) + 1) // 2
            for i in range(0, len(corpora), 2):
                if i + 1 < len(corpora):
                    print(f"[Year {year}] Round {round_num}: Merging pair {i//2 + 1}/{pairs}")
                    merged = Corpus.merge(corpora[i], corpora[i + 1])
                    next_round.append(merged)
                else:
                    # Odd one out, carry forward
                    next_round.append(corpora[i])
            corpora = next_round
            round_num += 1

        merged_corpus = corpora[0]

        output_path = final_directory + "wikiconv_corpus_merged_" + str(year)
        print(f"[Year {year}] Saving merged corpus")
        merged_corpus.dump(output_path)
        return f"Saved merged corpus ({len(paths_lst)} files merged)"


if __name__ == "__main__":
    main()
