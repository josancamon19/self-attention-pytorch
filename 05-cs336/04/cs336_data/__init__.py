import importlib.metadata

__version__ = importlib.metadata.version("cs336-data")


# convert common crawl html to text
# filter (harmful, personal identifiable info, etc)
# deduplicate
# train models on different datasets to understand decision results

# researchers don't build a web crawler, use publicly available crawls, like common
# significant work to make it usable
# assignment: setup a pipeline that does a bunch of this cleaning.

# 2.1 Explore
# WARC, WAT, WET formats

# warc.gz
# wet files looks much cleaner, but still, it's strange to have title/content/lists in a single string of data, wouldn't that be like teaching the model how to write like this structured way of making websites
# hard to keep
# this data seem so trash, like forms? wtf? there's no text structure here, maybe some words, and few words, but most things have so much shit
# so many languages, woh, I wonder what's the web distribution of englih vs other languages
# there is some stuff that's reasonable, blogs, news articles
# a lot of things with images just get organized so poorly, especially menu's, wtf


# comparing both extraction methods, resil feels better, keeps the structure, but so much spacing as well.
# would do some extra to clean out that much spacing, took 70.62 seconds to parse, 28k records

# langauge finding, apparently fasttext does this 
# issues from lang id procedure downstream on training an LLM?
# I'd say model learning different text structures, model mixing languages
# mitigating? having enough sampling with each language, balanced?, also maybe not mixing languages?
# but some people mix them, what do you do about it?
# does being very good at 1 language translates to another one?