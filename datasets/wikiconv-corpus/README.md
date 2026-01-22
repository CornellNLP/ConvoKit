# WikiConv Language Dataset Processing

This directory contains the updated files used to process the English, German, Russian, Chinese, and Greek WikiConv datasets. These datasets were released with "WikiConv: A Corpus of the Complete Conversational History of a Large Online Collaborative Community", and can be found [here](https://figshare.com/projects/WikiConv_A_Corpus_of_the_Complete_Conversational_History_of_a_Large_Online_Collaborative_Community/57110).

The files have been updated to handle datasets of all 5 languages; the original files were tailored for the English dataset only. In addition, the original English dataset contained significants amount of data containing empty or trivial text and no modification metadata (which suggests that the data was modified, deleted, or restored). This data has essentially no substance, so wikiconv_conversion_with_merging_04_28_20.py has been updated to filter these utterances out. Finally, optimizations were made to improve runtime of processing given the size of the raw data. 

## Files:
to_jsonlist_lang.py: Downloads raw data from the figshare and converts it into jsonlist format, deleting the raw files afterwards
wikiconv_conversion_with_merging_04_28_20.py: Updated code for converting jsonlist formatted files to convokit format. Includes the filter for utterances with empty text and no modifications. 
merge_parallel: parallelizes the merging portion of wikiconv_conversion_with_merging_04_28_20.py since it takes too long to run for datasets of large size.


## Current status
The English dataset, which is by far the largest, has been reprocessed. The processing for the Greek dataset is currently running, and over the next month the others will be completed as well. 