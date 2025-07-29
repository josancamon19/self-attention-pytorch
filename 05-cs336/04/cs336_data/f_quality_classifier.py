# ==== QUALITY CLASSIFIER =====
# - this also applies to search engines
# - in general the rule is, hq pages link other hq pages
# - heuristics, gpt2 reddit karma >= 3
# - or links that are only in wikipedia
# - 40GB of text, this is too small, but you can take it as example, and train a classifier
# -- with this as good examples, common crawl as negative examples. Giving you a quality score
# -- implement a classifier using a subset of hq pages + use some of your filters.


