#!/bin/bash

set -e

echo "Creating directory structure..."
mkdir -p .data
mkdir -p .models
mkdir -p cs336_data/leaderboard/.data
mkdir -p leaderboard/.models

# echo "Downloading WARC sample files..."
# wget -O .data/sample.warc.gz https://data.commoncrawl.org/crawl-data/CC-MAIN-2025-18/segments/1744889135610.12/warc/CC-MAIN-20250417135010-20250417165010-00065.warc.gz
# wget -O .data/sample.warc.wet.gz https://data.commoncrawl.org/crawl-data/CC-MAIN-2025-18/segments/1744889135610.12/wet/CC-MAIN-20250417135010-20250417165010-00065.warc.wet.gz

# echo "Downloading FastText models..."
# wget -O .models/jigsaw_fasttext_bigrams_nsfw_final.bin https://dolma-artifacts.org/fasttext_models/jigsaw_fasttext_bigrams_20230515/jigsaw_fasttext_bigrams_nsfw_final.bin
# wget -O .models/jigsaw_fasttext_bigrams_hatespeech_final.bin https://dolma-artifacts.org/fasttext_models/jigsaw_fasttext_bigrams_20230515/jigsaw_fasttext_bigrams_hatespeech_final.bin
# wget -O .models/lid.176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

# echo "Downloading Wikipedia data..."
# wget -O .data/enwiki-20240420-extracted_urls.txt.gz https://nlp.stanford.edu/data/nfliu/cs336-spring-2024/assignment4/enwiki-20240420-extracted_urls.txt.gz
# gunzip .data/enwiki-20240420-extracted_urls.txt.gz
# mv .data/enwiki-20240420-extracted_urls.txt .data/wikipedia_HQ_urls.txt

echo "Downloading leaderboard WARC files for negative classifier sampels..."
wget -O cs336_data/leaderboard/.data/00000.warc.gz https://data.commoncrawl.org/crawl-data/CC-MAIN-2025-30/segments/1751905933612.63/warc/CC-MAIN-20250707183638-20250707213638-00000.warc.gz
wget -O cs336_data/leaderboard/.data/00001.warc.gz https://data.commoncrawl.org/crawl-data/CC-MAIN-2025-30/segments/1751905933612.63/warc/CC-MAIN-20250707183638-20250707213638-00001.warc.gz
wget -O cs336_data/leaderboard/.data/00002.warc.gz https://data.commoncrawl.org/crawl-data/CC-MAIN-2025-30/segments/1751905933612.63/warc/CC-MAIN-20250707183638-20250707213638-00002.warc.gz

echo "All downloads completed successfully!"