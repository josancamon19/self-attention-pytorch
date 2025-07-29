# ==== QUALITY CLASSIFIER =====
# - this also applies to search engines
# - in general the rule is, hq pages link other hq pages
# - heuristics, gpt2 reddit karma >= 3
# - or links that are only in wikipedia
# - 40GB of text, this is too small, but you can take it as example, and train a classifier
# -- with this as good examples, common crawl as negative examples. Giving you a quality score
# -- implement a classifier using a subset of hq pages + use some of your filters.


# This file contains a list of 43.5M external links
# found on Wikipedia pages in the English language as of April of 2024, but we expect you to subsample these
# URLs to get positive examples of “high-quality” text for training your classifier. Note that these positive
# examples may still contain undesirable content, so it may be useful to apply the other primitives you’ve built
# (lang id, filtering rules, ,etc)

# Scrape contents of an URL
# wget–-timeout=5 \
# -i subsampled_positive_urls.txt \
# --warc-file=subsampled_positive_urls.warc \
# -O /dev/null

#  Train a quality classifier that, given text, returns a numeric quality score.
# Deliverable: A quality classifier for use in the next subproblem.
# (b) Write a function that labels a page as high or low-quality, and provides a confidence score in the
# label.

# Steps
# - get some quality urls from wikipedia sampling (subsample, try 10k to start)
# - download into warc contents
# - preprocess this ones, and clean further, how many left? apply filters
# - subsample from warc common crawl, how many? same count?
# - 

def subsample(target_count: int = 10000):
    pass