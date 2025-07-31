import numpy as np
from transformers import AutoTokenizer
import re
from collections import Counter
import urllib.parse
import fasttext

tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir="cs336_data/leaderboard/.cache")
validation_data = np.fromfile(
    "cs336_data/leaderboard/.data/tokenized_paloma_c4_100_domains_validation.bin", dtype=np.uint16
)
# allowed to make use of paloma validation data in constructing filters or classifiers to process the cc wet files
print(f"Total tokens: {len(validation_data):,}")  # < 10M tokens
full_text = tokenizer.decode(validation_data)
with open("cs336_data/leaderboard/.data/paloma_c4_100_domains_validation.txt", "w", encoding="utf-8") as f:
    f.write(full_text)

# raise Exception()
print("\n" + "=" * 60)
print("PALOMA C4 100 DOMAINS VALIDATION DATA STATISTICS")
print("=" * 60)

# Basic text statistics
lines = full_text.split("\n")
sentences = re.split(r"[.!?]+", full_text)
words = full_text.split()
chars = len(full_text)

print("\nBasic Statistics:")
print(f"  Total characters: {chars:,}")
print(f"  Total lines: {len(lines):,}")
print(f"  Total words: {len(words):,}")
print(f"  Total sentences: {len(sentences):,}")
print(f"  Avg chars per line: {chars / len(lines):.1f}")
print(f"  Avg words per line: {len(words) / len(lines):.1f}")
print(f"  Avg chars per word: {chars / len(words):.1f}")


# Document separation analysis (look for <|endoftext|> tokens)
endoftext_token = tokenizer.encode("<|endoftext|>")[0]
doc_boundaries = np.where(validation_data == endoftext_token)[0]
num_documents = len(doc_boundaries)

print("\nDocument Structure:")
print(f"  Document separator token ID: {endoftext_token}")
print(f"  Number of documents: {num_documents:,}")
if num_documents > 0:
    doc_lengths = np.diff(np.concatenate([[0], doc_boundaries, [len(validation_data)]]))
    print(f"  Avg tokens per document: {np.mean(doc_lengths):.1f}")
    print(f"  Median tokens per document: {np.median(doc_lengths):.1f}")
    print(f"  Min tokens per document: {np.min(doc_lengths):,}")
    print(f"  Max tokens per document: {np.max(doc_lengths):,}")

# Language patterns
print("\nLanguage Patterns:")
uppercase_ratio = sum(1 for c in full_text if c.isupper()) / chars
digit_ratio = sum(1 for c in full_text if c.isdigit()) / chars
punctuation_ratio = sum(1 for c in full_text if c in '.,!?;:"()[]{}') / chars

print(f"  Uppercase ratio: {uppercase_ratio:.3f}")
print(f"  Digit ratio: {digit_ratio:.3f}")
print(f"  Punctuation ratio: {punctuation_ratio:.3f}")

# Domain/URL analysis (look for common web patterns)
urls = re.findall(r'https?://[^\s<>"]+', full_text)
domains = []
for url in urls:
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.netloc:
            domains.append(parsed.netloc.lower())
    except:
        pass

if domains:
    domain_counts = Counter(domains)
    print("\nDomain Analysis:")
    print(f"  Total URLs found: {len(urls)}")
    print(f"  Unique domains: {len(domain_counts)}")
    print("  Most common domains:")
    for domain, count in domain_counts.most_common(5):
        print(f"    {domain:30s}: {count}")
        # www.w3.org
        # doc 1 has like some messages exchanges/blog from w3org people or so
        # doc 2 has like html on it, and has so much of this URL,
        # that same document has svn.apache.org a lot of times
        # go.microsoft.com
        # microsoft instruction manual
        # github.com ~ some sort of code, but very minimal, no need to include data with code,
        # - rather some PR's with comments, and readme updates
        # www.facebook.com is like for ads, people saying join us, follow us, etc

# Text quality indicators
print("\nText Quality Indicators:")
avg_word_length = np.mean([len(word) for word in words])
long_words = sum(1 for word in words if len(word) > 10)
short_lines = sum(1 for line in lines if len(line.strip()) < 20)

print(f"  Average word length: {avg_word_length:.2f}")
print(f"  Words > 10 chars: {long_words:,} ({long_words / len(words) * 100:.1f}%)")
print(f"  Short lines (<20 chars): {short_lines:,} ({short_lines / len(lines) * 100:.1f}%)")

# self-analysis + Gemini
# IGN game/tech reviews, a few articles, gotta check manually, like 20
# List 100 common news/reports entities like BBC, Times,
# or directly find ones involved in the dataset
# https://huggingface.co/datasets/allenai/paloma/tree/main/c4_100_domains
# here I have the URL's sort of in order to filter, but that'd be like cheating, thus gotta manually see what makes sense given the data

# News outlet analysis
print("\n" + "=" * 60)
print("SOURCES ANALYSIS")
print("=" * 60)

# Common news outlets with multiple name variations

# Split text into documents using endoftext token
documents = []
if num_documents > 0:
    doc_start = 0
    for boundary in doc_boundaries:
        doc_tokens = validation_data[doc_start:boundary]
        doc_text = tokenizer.decode(doc_tokens)
        documents.append(doc_text)
        doc_start = boundary + 1
    # Add final document
    if doc_start < len(validation_data):
        final_doc_tokens = validation_data[doc_start:]
        final_doc_text = tokenizer.decode(final_doc_tokens)
        documents.append(final_doc_text)
else:
    # If no document boundaries, treat as single document
    documents = [full_text]

print(f"Total documents to analyze: {len(documents)}")


def analyze_topic_sources(topic_name, sources_list, keywords):
    """
    Analyze a specific topic by looking for mentions of sources and related keywords

    Args:
        topic_name: Name of the topic (e.g., "NEWS", "GAMING", "SPORTS")
        sources_list: List of source names/outlets to search for
        keywords: List of topic-related keywords to count
    """
    print(f"\n{topic_name} OUTLET ANALYSIS")
    print("=" * 60)

    # Count source mentions per document and overall
    sources_stats = {}

    for source in sources_list:
        pattern = re.compile(r"\b" + re.escape(source) + r"\b", re.IGNORECASE)

        docs_containing_source = 0
        total_mentions = 0
        chars_in_docs_with_source = 0

        for doc in documents:
            matches = pattern.findall(doc)
            if matches:
                docs_containing_source += 1
                total_mentions += len(matches)
                chars_in_docs_with_source += len(doc)

        if total_mentions > 0:
            sources_stats[source] = {
                "total_mentions": total_mentions,
                "docs_with_source": docs_containing_source,
                "chars_in_docs": chars_in_docs_with_source,
                "doc_percentage": (docs_containing_source / len(documents)) * 100,
                "char_percentage": (chars_in_docs_with_source / chars) * 100,
            }

    # Count documents that contain any source from this topic
    docs_with_any_source = set()
    total_chars_any_source = 0

    for source in sources_list:
        pattern = re.compile(r"\b" + re.escape(source) + r"\b", re.IGNORECASE)
        for i, doc in enumerate(documents):
            if pattern.search(doc):
                if i not in docs_with_any_source:
                    docs_with_any_source.add(i)
                    total_chars_any_source += len(doc)

    # Sort by total mentions and show top 20
    sorted_sources = sorted(sources_stats.items(), key=lambda x: x[1]["total_mentions"], reverse=True)

    print(f"\nTop 20 Most Mentioned {topic_name} Sources:")
    print(f"{'Source':<25} {'Mentions':<8} {'Docs':<6} {'Doc %':<7} {'Char %':<8}")
    print("-" * 60)

    for i, (source, stats) in enumerate(sorted_sources[:20]):
        print(
            f"{source:<25} {stats['total_mentions']:<8} {stats['docs_with_source']:<6} "
            f"{stats['doc_percentage']:<6.1f}% {stats['char_percentage']:<7.1f}%"
        )

    print(f"\nOverall {topic_name} Content Statistics:")
    print(
        f"  Documents with any {topic_name.lower()} source: {len(docs_with_any_source):,} / {len(documents):,} ({len(docs_with_any_source) / len(documents) * 100:.1f}%)"
    )
    print(
        f"  Characters in {topic_name.lower()}-related docs: {total_chars_any_source:,} / {chars:,} ({total_chars_any_source / chars * 100:.1f}%)"
    )
    print(f"  Unique {topic_name.lower()} sources found: {len(sources_stats)}")

    # Keywords analysis
    keyword_counts = {}
    for keyword in keywords:
        count = len(re.findall(r"\b" + re.escape(keyword) + r"\b", full_text, re.IGNORECASE))
        if count > 0:
            keyword_counts[keyword] = count

    if keyword_counts:
        print(f"\n{topic_name}-related Keywords Found:")
        for keyword, count in sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {keyword:<20}: {count:,} mentions")

    return sources_stats, len(docs_with_any_source), total_chars_any_source


news_outlets = [
    # US Major Networks
    "cnn",
    "cable news network",
    "fox news",
    "foxnews",
    "msnbc",
    "ms-nbc",
    "abc news",
    "abcnews",
    "cbs news",
    "cbsnews",
    "nbc news",
    # Major Newspapers
    "new york times",
    "nytimes",
    "ny times",
    "washington post",
    "washingtonpost",
    "wall street journal",
    "wsj",
    "usa today",
    "usatoday",
    "los angeles times",
    "la times",
    "latimes",
    # International Major
    "bbc",
    "bbc news",
    "reuters",
    "associated press",
    "ap news",
    "the guardian",
    "daily mail",
    "dailymail",
    "daily telegraph",
    "the independent",
    # Business/Finance
    "bloomberg",
    "cnbc",
    "forbes",
    "financial times",
    # Magazines/Opinion
    "IBTimes",
    "time magazine",
    "times magazine",
    "newsweek",
    "the new yorker",
    "politico",
    "huffington post",
    " vox",
    # Public Radio/TV
    "pbs",
    "cspan",
    # International
    "al jazeera",
    "aljazeera",
    "china daily",
    "times of india",
    "indiatimes.com",
    # Canadian
    "globe and mail",
    "cbc",
]

news_keywords = [
    "breaking news",
    "reporter",
    "journalist",
    "correspondent",
    "editorial",
    "headline",
    "newsroom",
    "press release",
    "exclusive",
    "developing story",
]

# analyze_topic_sources("news", news_outlets, news_keywords)  # 12.5% of content, considering reuters like 3%
# analyze_topic_sources("general", [], [" NBA", " NFL", " stock", " USB", " laptop", "$"])
# # I still feel like most of this is news, even tho it says above 15% it feels like it should be like 50%? no?
# # has code?

# # FastText model analysis
# print("\n" + "=" * 60)
# print("FASTTEXT MODEL CLASSIFICATION ANALYSIS")
# print("=" * 60)

# print("Loading FastText models...")
# news_model = fasttext.load_model("cs336_data/leaderboard/.models/ag_news.bin")
# amazon_review_model = fasttext.load_model("cs336_data/leaderboard/.models/amazon_review_full.bin")
# yahoo_answer_model = fasttext.load_model("cs336_data/leaderboard/.models/yahoo_answers.bin")

# # Get model labels to understand what each model classifies
# print(f"AG News labels: {news_model.get_labels()}")
# print(f"Amazon Review labels: {amazon_review_model.get_labels()}")
# print(f"Yahoo Answers labels: {yahoo_answer_model.get_labels()}")


# def classify_documents_with_fasttext():
#     """
#     Run FastText models on all documents to classify content types
#     """
#     results = {
#         "news": {"classifications": {}, "total_docs": 0, "total_chars": 0},
#         "amazon": {"classifications": {}, "total_docs": 0, "total_chars": 0},
#         "yahoo": {"classifications": {}, "total_docs": 0, "total_chars": 0},
#     }

#     print(f"\nClassifying {len(documents)} documents...")

#     for i, doc in enumerate(documents):
#         if len(doc.strip()) < 50:  # Skip very short documents
#             continue

#         # Prepare text for FastText (remove newlines, limit length)
#         clean_doc = doc.replace("\n", " ").strip()[:1000]  # Limit to 1000 chars for speed

#         if not clean_doc:
#             continue

#         # Classify with AG News model
#         try:
#             news_pred = news_model.predict(clean_doc, k=1)
#             news_label = news_pred[0][0].replace("__label__", "")
#             news_score = news_pred[1][0]

#             if news_label not in results["news"]["classifications"]:
#                 results["news"]["classifications"][news_label] = {
#                     "count": 0,
#                     "chars": 0,
#                     "avg_score": 0,
#                     "total_score": 0,
#                 }

#             results["news"]["classifications"][news_label]["count"] += 1
#             results["news"]["classifications"][news_label]["chars"] += len(doc)
#             results["news"]["classifications"][news_label]["total_score"] += news_score
#             results["news"]["classifications"][news_label]["avg_score"] = (
#                 results["news"]["classifications"][news_label]["total_score"]
#                 / results["news"]["classifications"][news_label]["count"]
#             )

#             results["news"]["total_docs"] += 1
#             results["news"]["total_chars"] += len(doc)
#         except:
#             pass

#         # Classify with Amazon Review model
#         try:
#             amazon_pred = amazon_review_model.predict(clean_doc, k=1)
#             amazon_label = amazon_pred[0][0].replace("__label__", "")
#             amazon_score = amazon_pred[1][0]

#             if amazon_label not in results["amazon"]["classifications"]:
#                 results["amazon"]["classifications"][amazon_label] = {
#                     "count": 0,
#                     "chars": 0,
#                     "avg_score": 0,
#                     "total_score": 0,
#                 }

#             results["amazon"]["classifications"][amazon_label]["count"] += 1
#             results["amazon"]["classifications"][amazon_label]["chars"] += len(doc)
#             results["amazon"]["classifications"][amazon_label]["total_score"] += amazon_score
#             results["amazon"]["classifications"][amazon_label]["avg_score"] = (
#                 results["amazon"]["classifications"][amazon_label]["total_score"]
#                 / results["amazon"]["classifications"][amazon_label]["count"]
#             )

#             results["amazon"]["total_docs"] += 1
#             results["amazon"]["total_chars"] += len(doc)
#         except:
#             pass

#         # Classify with Yahoo Answers model
#         try:
#             yahoo_pred = yahoo_answer_model.predict(clean_doc, k=1)
#             yahoo_label = yahoo_pred[0][0].replace("__label__", "")
#             yahoo_score = yahoo_pred[1][0]

#             if yahoo_label not in results["yahoo"]["classifications"]:
#                 results["yahoo"]["classifications"][yahoo_label] = {
#                     "count": 0,
#                     "chars": 0,
#                     "avg_score": 0,
#                     "total_score": 0,
#                 }

#             results["yahoo"]["classifications"][yahoo_label]["count"] += 1
#             results["yahoo"]["classifications"][yahoo_label]["chars"] += len(doc)
#             results["yahoo"]["classifications"][yahoo_label]["total_score"] += yahoo_score
#             results["yahoo"]["classifications"][yahoo_label]["avg_score"] = (
#                 results["yahoo"]["classifications"][yahoo_label]["total_score"]
#                 / results["yahoo"]["classifications"][yahoo_label]["count"]
#             )

#             results["yahoo"]["total_docs"] += 1
#             results["yahoo"]["total_chars"] += len(doc)
#         except:
#             pass

#     return results


# # Run classification
# classification_results = classify_documents_with_fasttext()

# # Display results
# for model_name, model_results in classification_results.items():
#     model_display = {
#         "news": "AG News (News Categories)",
#         "amazon": "Amazon Reviews (Product Reviews)",
#         "yahoo": "Yahoo Answers (Q&A Categories)",
#     }

#     print(f"\n{model_display[model_name]} Results:")
#     print(f"  Total documents classified: {model_results['total_docs']:,}")
#     print(
#         f"  Total characters: {model_results['total_chars']:,} ({model_results['total_chars'] / chars * 100:.1f}% of dataset)"
#     )

#     if model_results["classifications"]:
#         print(f"  Classification breakdown:")
#         sorted_classes = sorted(model_results["classifications"].items(), key=lambda x: x[1]["count"], reverse=True)

#         for class_name, class_data in sorted_classes:
#             doc_pct = class_data["count"] / model_results["total_docs"] * 100
#             char_pct = class_data["chars"] / chars * 100
#             avg_score = class_data["avg_score"]
#             print(
#                 f"    {class_name:<20}: {class_data['count']:4d} docs ({doc_pct:5.1f}%), "
#                 f"{class_data['chars']:8,} chars ({char_pct:4.1f}%), "
#                 f"avg_score: {avg_score:.3f}"
#             )

# # High-confidence classifications analysis
# print(f"\n{'=' * 60}")
# print("HIGH-CONFIDENCE CLASSIFICATIONS (score > 0.8)")
# print(f"{'=' * 60}")

# high_confidence_stats = {
#     "news": {"total": 0, "by_class": {}},
#     "amazon": {"total": 0, "by_class": {}},
#     "yahoo": {"total": 0, "by_class": {}},
# }

# for i, doc in enumerate(documents):
#     if len(doc.strip()) < 50:
#         continue

#     clean_doc = doc.replace("\n", " ").strip()[:1000]
#     if not clean_doc:
#         continue

#     # Check high-confidence predictions
#     try:
#         news_pred = news_model.predict(clean_doc, k=1)
#         if news_pred[1][0] > 0.8:
#             label = news_pred[0][0].replace("__label__", "")
#             high_confidence_stats["news"]["total"] += 1
#             if label not in high_confidence_stats["news"]["by_class"]:
#                 high_confidence_stats["news"]["by_class"][label] = 0
#             high_confidence_stats["news"]["by_class"][label] += 1
#     except:
#         pass

#     try:
#         amazon_pred = amazon_review_model.predict(clean_doc, k=1)
#         if amazon_pred[1][0] > 0.8:
#             label = amazon_pred[0][0].replace("__label__", "")
#             high_confidence_stats["amazon"]["total"] += 1
#             if label not in high_confidence_stats["amazon"]["by_class"]:
#                 high_confidence_stats["amazon"]["by_class"][label] = 0
#             high_confidence_stats["amazon"]["by_class"][label] += 1
#     except:
#         pass

#     try:
#         yahoo_pred = yahoo_answer_model.predict(clean_doc, k=1)
#         if yahoo_pred[1][0] > 0.8:
#             label = yahoo_pred[0][0].replace("__label__", "")
#             high_confidence_stats["yahoo"]["total"] += 1
#             if label not in high_confidence_stats["yahoo"]["by_class"]:
#                 high_confidence_stats["yahoo"]["by_class"][label] = 0
#             high_confidence_stats["yahoo"]["by_class"][label] += 1
#     except:
#         pass

# for model_name, stats in high_confidence_stats.items():
#     model_display = {
#         "news": "AG News (High Confidence)",
#         "amazon": "Amazon Reviews (High Confidence)",
#         "yahoo": "Yahoo Answers (High Confidence)",
#     }

#     print(f"\n{model_display[model_name]}:")
#     print(
#         f"  High-confidence documents: {stats['total']:,} / {len(documents):,} ({stats['total'] / len(documents) * 100:.1f}%)"
#     )

#     if stats["by_class"]:
#         sorted_classes = sorted(stats["by_class"].items(), key=lambda x: x[1], reverse=True)
#         for class_name, count in sorted_classes:
#             pct = count / stats["total"] * 100 if stats["total"] > 0 else 0
#             print(f"    {class_name:<20}: {count:4d} ({pct:5.1f}%)")

# Answer the user's specific questions
print("\n" + "=" * 60)
print("ANSWERING SPECIFIC CONTENT TYPE QUESTIONS")
print("=" * 60)

# Company mentions analysis
print("\nCompany/Tech Company Analysis:")
tech_companies = [
    "google",
    "facebook",
    "twitter",
    "microsoft",
    "apple",
    "amazon",
    "netflix",
    "uber",
    "airbnb",
    "tesla",
    "nvidia",
    "intel",
    "ibm",
    "oracle",
    "salesforce",
    "adobe",
    "spotify",
    "linkedin",
    "youtube",
    "instagram",
]
company_mentions = {}
for company in tech_companies:
    count = len(re.findall(r"\b" + company + r"\b", full_text, re.IGNORECASE))
    if count > 0:
        company_mentions[company] = count

sorted_companies = sorted(company_mentions.items(), key=lambda x: x[1], reverse=True)
for company, count in sorted_companies[:15]:
    print(f"  {company:<12}: {count:,} mentions")

# Academic content detection (document-wise)
print("\nAcademic Content Analysis:")
academic_doc_stats = {
    "citations": 0,
    "academic_words": 0, 
    "author_patterns": 0,
    "doi_patterns": 0,
    "arxiv_patterns": 0,
    "any_academic": 0
}

academic_chars = 0
docs_with_academic = set()

for i, doc in enumerate(documents):
    doc_has_academic = False
    
    # Check for citations (more specific patterns)
    citation_patterns = [
        r"\[\d+\]",  # Numbered citations [1], [23]
        r"\([A-Z][a-z]+,?\s+\d{4}\)",  # Author citations (Smith, 2023)
        r"\([A-Z][a-z]+\s+et\s+al\.?,?\s+\d{4}\)",  # Et al citations (Smith et al., 2023)
        r"\([A-Z][a-z]+\s+&\s+[A-Z][a-z]+,?\s+\d{4}\)",  # Two author citations (Smith & Jones, 2023)
        r"\b[A-Z][a-z]+\s+et\s+al\.\s+\(\d{4}\)",  # In-text et al (Smith et al. (2023))
        r"\b[A-Z][a-z]+\s+\(\d{4}\)",  # In-text single author (Smith (2023))
    ]
    citation_count = sum(len(re.findall(pattern, doc)) for pattern in citation_patterns)
    if citation_count > 0:
        academic_doc_stats["citations"] += 1
        doc_has_academic = True
    
    # Check for academic words
    if len(re.findall(r"\b(research|study|analysis|methodology|hypothesis|conclusion|abstract|introduction|references|bibliography|journal|conference|university|professor|phd|doctorate)\b", doc, re.IGNORECASE)) > 2:
        academic_doc_stats["academic_words"] += 1
        doc_has_academic = True
    
    # Check for author patterns
    if len(re.findall(r"\bet al\.|\b[A-Z][a-z]+ et al\b", doc)) > 0:
        academic_doc_stats["author_patterns"] += 1
        doc_has_academic = True
    
    # Check for DOI patterns
    if len(re.findall(r"doi:|DOI:", doc)) > 0:
        academic_doc_stats["doi_patterns"] += 1
        doc_has_academic = True
    
    # Check for arXiv patterns
    if len(re.findall(r"arxiv|arXiv", doc, re.IGNORECASE)) > 0:
        academic_doc_stats["arxiv_patterns"] += 1
        doc_has_academic = True
    
    if doc_has_academic:
        docs_with_academic.add(i)
        academic_chars += len(doc)
        academic_doc_stats["any_academic"] += 1

print(f"  Documents with citations: {academic_doc_stats['citations']} ({academic_doc_stats['citations']/len(documents)*100:.1f}%)")
print(f"  Documents with academic words (>2): {academic_doc_stats['academic_words']} ({academic_doc_stats['academic_words']/len(documents)*100:.1f}%)")
print(f"  Documents with author patterns: {academic_doc_stats['author_patterns']} ({academic_doc_stats['author_patterns']/len(documents)*100:.1f}%)")
print(f"  Documents with DOI patterns: {academic_doc_stats['doi_patterns']} ({academic_doc_stats['doi_patterns']/len(documents)*100:.1f}%)")
print(f"  Documents with arXiv patterns: {academic_doc_stats['arxiv_patterns']} ({academic_doc_stats['arxiv_patterns']/len(documents)*100:.1f}%)")
print(f"  Documents with any academic indicators: {academic_doc_stats['any_academic']} ({academic_doc_stats['any_academic']/len(documents)*100:.1f}%)")
print(f"  Characters in academic docs: {academic_chars:,} ({academic_chars/chars*100:.1f}%)")

# Email/Forum format detection (document-wise)
print("\nEmail/Forum Format Analysis:")
email_forum_doc_stats = {
    "email_quotes": 0,
    "email_headers": 0,
    "forum_quotes": 0,
    "thread_patterns": 0,
    "reddit_patterns": 0,
    "any_email_forum": 0
}

email_forum_chars = 0
docs_with_email_forum = set()

for i, doc in enumerate(documents):
    doc_has_email_forum = False
    
    # Check for email quotes
    if len(re.findall(r"^>\s*[^\s]", doc, re.MULTILINE)) > 0:
        email_forum_doc_stats["email_quotes"] += 1
        doc_has_email_forum = True
    
    # Check for email headers
    if len(re.findall(r"^(From:|To:|Subject:|Date:|Reply-To:)", doc, re.MULTILINE)) > 0:
        email_forum_doc_stats["email_headers"] += 1
        doc_has_email_forum = True
    
    # Check for forum quotes
    if len(re.findall(r"wrote:|said:|posted:", doc, re.IGNORECASE)) > 0:
        email_forum_doc_stats["forum_quotes"] += 1
        doc_has_email_forum = True
    
    # Check for thread patterns
    if len(re.findall(r"\b(thread|post #|reply #|username|karma|upvote|downvote)\b", doc, re.IGNORECASE)) > 0:
        email_forum_doc_stats["thread_patterns"] += 1
        doc_has_email_forum = True
    
    # Check for reddit patterns
    if len(re.findall(r"\b(subreddit|/r/|reddit|redditor)\b", doc, re.IGNORECASE)) > 0:
        email_forum_doc_stats["reddit_patterns"] += 1
        doc_has_email_forum = True
    
    if doc_has_email_forum:
        docs_with_email_forum.add(i)
        email_forum_chars += len(doc)
        email_forum_doc_stats["any_email_forum"] += 1

print(f"  Documents with email quotes: {email_forum_doc_stats['email_quotes']} ({email_forum_doc_stats['email_quotes']/len(documents)*100:.1f}%)")
print(f"  Documents with email headers: {email_forum_doc_stats['email_headers']} ({email_forum_doc_stats['email_headers']/len(documents)*100:.1f}%)")
print(f"  Documents with forum quotes: {email_forum_doc_stats['forum_quotes']} ({email_forum_doc_stats['forum_quotes']/len(documents)*100:.1f}%)")
print(f"  Documents with thread patterns: {email_forum_doc_stats['thread_patterns']} ({email_forum_doc_stats['thread_patterns']/len(documents)*100:.1f}%)")
print(f"  Documents with reddit patterns: {email_forum_doc_stats['reddit_patterns']} ({email_forum_doc_stats['reddit_patterns']/len(documents)*100:.1f}%)")
print(f"  Documents with any email/forum indicators: {email_forum_doc_stats['any_email_forum']} ({email_forum_doc_stats['any_email_forum']/len(documents)*100:.1f}%)")
print(f"  Characters in email/forum docs: {email_forum_chars:,} ({email_forum_chars/chars*100:.1f}%)")

# Blog indicators (document-wise)
print("\nBlog Content Analysis:")
blog_doc_stats = {
    "blog_words": 0,
    "personal_heavy": 0,  # Documents with >10 personal pronouns
    "opinion_words": 0,
    "informal_language": 0,
    "any_blog": 0
}

blog_chars = 0
docs_with_blog = set()

for i, doc in enumerate(documents):
    doc_has_blog = False
    
    # Check for blog words
    if len(re.findall(r"\b(blog|blogger|blogging|wordpress|tumblr|medium\.com)\b", doc, re.IGNORECASE)) > 0:
        blog_doc_stats["blog_words"] += 1
        doc_has_blog = True
    
    # Check for heavy personal pronoun usage
    personal_count = len(re.findall(r"\b(I|my|me|myself)\b", doc))
    if personal_count > 10:  # Threshold for personal content
        blog_doc_stats["personal_heavy"] += 1
        doc_has_blog = True
    
    # Check for opinion words
    if len(re.findall(r"\b(I think|in my opinion|personally|I believe|I feel)\b", doc, re.IGNORECASE)) > 0:
        blog_doc_stats["opinion_words"] += 1
        doc_has_blog = True
    
    # Check for informal language
    if len(re.findall(r"\b(gonna|wanna|kinda|sorta|yeah|nah|ok|okay)\b", doc, re.IGNORECASE)) > 2:
        blog_doc_stats["informal_language"] += 1
        doc_has_blog = True
    
    if doc_has_blog:
        docs_with_blog.add(i)
        blog_chars += len(doc)
        blog_doc_stats["any_blog"] += 1

print(f"  Documents with blog words: {blog_doc_stats['blog_words']} ({blog_doc_stats['blog_words']/len(documents)*100:.1f}%)")
print(f"  Documents with heavy personal pronoun use (>10): {blog_doc_stats['personal_heavy']} ({blog_doc_stats['personal_heavy']/len(documents)*100:.1f}%)")
print(f"  Documents with opinion words: {blog_doc_stats['opinion_words']} ({blog_doc_stats['opinion_words']/len(documents)*100:.1f}%)")
print(f"  Documents with informal language (>2): {blog_doc_stats['informal_language']} ({blog_doc_stats['informal_language']/len(documents)*100:.1f}%)")
print(f"  Documents with any blog indicators: {blog_doc_stats['any_blog']} ({blog_doc_stats['any_blog']/len(documents)*100:.1f}%)")
print(f"  Characters in blog-style docs: {blog_chars:,} ({blog_chars/chars*100:.1f}%)")

# Books detection (document-wise)
print("\nBook Content Analysis:")
book_doc_stats = {
    "chapter_headings": 0,
    "isbn_patterns": 0,
    "publisher_patterns": 0,
    "book_references": 0,
    "any_book": 0
}

book_chars = 0
docs_with_book = set()

for i, doc in enumerate(documents):
    doc_has_book = False
    
    # Check for chapter headings
    if len(re.findall(r"Chapter \d+|CHAPTER \d+", doc)) > 0:
        book_doc_stats["chapter_headings"] += 1
        doc_has_book = True
    
    # Check for ISBN patterns
    if len(re.findall(r"ISBN|isbn", doc)) > 0:
        book_doc_stats["isbn_patterns"] += 1
        doc_has_book = True
    
    # Check for publisher patterns
    if len(re.findall(r"\b(publisher|published by|copyright|©|edition)\b", doc, re.IGNORECASE)) > 0:
        book_doc_stats["publisher_patterns"] += 1
        doc_has_book = True
    
    # Check for book references
    if len(re.findall(r"\b(page \d+|pp\. \d+|chapter \d+)\b", doc, re.IGNORECASE)) > 0:
        book_doc_stats["book_references"] += 1
        doc_has_book = True
    
    if doc_has_book:
        docs_with_book.add(i)
        book_chars += len(doc)
        book_doc_stats["any_book"] += 1

print(f"  Documents with chapter headings: {book_doc_stats['chapter_headings']} ({book_doc_stats['chapter_headings']/len(documents)*100:.1f}%)")
print(f"  Documents with ISBN patterns: {book_doc_stats['isbn_patterns']} ({book_doc_stats['isbn_patterns']/len(documents)*100:.1f}%)")
print(f"  Documents with publisher patterns: {book_doc_stats['publisher_patterns']} ({book_doc_stats['publisher_patterns']/len(documents)*100:.1f}%)")
print(f"  Documents with book references: {book_doc_stats['book_references']} ({book_doc_stats['book_references']/len(documents)*100:.1f}%)")
print(f"  Documents with any book indicators: {book_doc_stats['any_book']} ({book_doc_stats['any_book']/len(documents)*100:.1f}%)")
print(f"  Characters in book-related docs: {book_chars:,} ({book_chars/chars*100:.1f}%)")


# def is_code_document(doc_text):
#     """Smarter code detection that avoids false positives from emails/forums"""
#     lines = doc_text.split("\n")
#     total_lines = len(lines)

#     if total_lines == 0:
#         return 0, {}

#     indicators = {}

#     # Very specific code patterns that rarely appear in regular text
#     patterns = {
#         # Function definitions with parentheses
#         "function_defs": len(
#             re.findall(r"^\s*(def|function|func|void|int|string|bool)\s+\w+\s*\(", doc_text, re.MULTILINE)
#         ),
#         # Class definitions
#         "class_defs": len(re.findall(r"^\s*class\s+\w+", doc_text, re.MULTILINE)),
#         # Import statements
#         "imports": len(re.findall(r"^\s*(import|#include|require|using\s+namespace)", doc_text, re.MULTILINE)),
#         # Method calls with dots
#         "method_calls": len(re.findall(r"\w+\.\w+\([^)]*\)", doc_text)),
#         # Variable assignments with equals
#         "assignments": len(re.findall(r"^\s*\w+\s*=\s*[^=]", doc_text, re.MULTILINE)),
#         # Code blocks with braces
#         "brace_blocks": len(re.findall(r"{\s*$", doc_text, re.MULTILINE)),
#         # Semicolon terminated statements
#         "semicolon_statements": len(re.findall(r"\w+.*;\s*$", doc_text, re.MULTILINE)),
#         # Array/list access
#         "array_access": len(re.findall(r"\w+\[\d*\]", doc_text)),
#         # HTML/XML tags
#         "html_tags": len(re.findall(r"<\w+[^>]*>", doc_text)),
#         # Code file extensions
#         "code_files": len(re.findall(r"\.(py|js|java|cpp|c|h|php|rb|go|ts|css|html|xml)\b", doc_text, re.IGNORECASE)),
#     }

#     # Anti-patterns (things that suggest it's NOT code)
#     anti_patterns = {
#         # Email indicators
#         "email_headers": len(re.findall(r"^(To:|From:|Cc:|Subject:|Date:)", doc_text, re.MULTILINE)),
#         "email_quotes": len(re.findall(r"^>\s*\w", doc_text, re.MULTILINE)),
#         "email_addresses": len(re.findall(r"\b\w+@\w+\.\w+", doc_text)),
#         # Forum/discussion indicators
#         "forum_quotes": len(re.findall(r"Quote:", doc_text, re.IGNORECASE)),
#         "user_said": len(re.findall(r"\w+ said:", doc_text, re.IGNORECASE)),
#         # News/article indicators
#         "news_phrases": len(
#             re.findall(r"\b(according to|sources say|breaking news|reported that)\b", doc_text, re.IGNORECASE)
#         ),
#     }

#     indicators.update(patterns)
#     indicators.update(anti_patterns)

#     # Calculate score
#     score = 0

#     # Strong positive indicators
#     score += min(patterns["function_defs"] / total_lines * 5, 0.4)
#     score += min(patterns["class_defs"] / total_lines * 10, 0.3)
#     score += min(patterns["imports"] / total_lines * 8, 0.2)
#     score += min(patterns["brace_blocks"] / total_lines * 3, 0.2)
#     score += min(patterns["method_calls"] / len(doc_text.split()) * 30, 0.2)
#     score += min(patterns["code_files"] / len(doc_text.split()) * 100, 0.2)

#     # Penalties for non-code content
#     penalty = 0
#     penalty += min(anti_patterns["email_headers"] / total_lines * 10, 0.8)
#     penalty += min(anti_patterns["email_quotes"] / total_lines * 3, 0.5)
#     penalty += min(anti_patterns["news_phrases"] / len(doc_text.split()) * 50, 0.3)

#     final_score = max(0, score - penalty)
#     return min(final_score, 1.0), indicators


# # Analyze documents
# code_docs = []
# total_code_chars = 0
# code_threshold = 0.2

# for i, doc in enumerate(documents):
#     score, indicators = is_code_document(doc)

#     if score > code_threshold:
#         code_docs.append({"doc_index": i, "score": score, "length": len(doc), "indicators": indicators})
#         total_code_chars += len(doc)

# # Sort by score
# code_docs.sort(key=lambda x: x["score"], reverse=True)

# chars = len(tokenizer.decode(validation_data))
# print(f"\nImproved Code Document Detection Results:")
# print(
#     f"  Documents identified as code: {len(code_docs)} / {len(documents)} ({len(code_docs) / len(documents) * 100:.1f}%)"
# )
# print(f"  Characters in code documents: {total_code_chars:,} / {chars:,} ({total_code_chars / chars * 100:.1f}%)")

# if code_docs:
#     print(f"\nTop 10 Code Documents (by score):")
#     print(f"{'Doc#':<5} {'Score':<6} {'Length':<8} {'Key Indicators'}")
#     print("-" * 60)

#     for i, doc_info in enumerate(code_docs[:10]):
#         # Show key positive indicators
#         key_indicators = []
#         ind = doc_info["indicators"]
#         if ind.get("function_defs", 0) > 0:
#             key_indicators.append(f"funcs:{ind['function_defs']}")
#         if ind.get("class_defs", 0) > 0:
#             key_indicators.append(f"classes:{ind['class_defs']}")
#         if ind.get("imports", 0) > 0:
#             key_indicators.append(f"imports:{ind['imports']}")
#         if ind.get("html_tags", 0) > 5:
#             key_indicators.append(f"html:{ind['html_tags']}")
#         if ind.get("code_files", 0) > 0:
#             key_indicators.append(f"files:{ind['code_files']}")

#         indicator_str = ", ".join(key_indicators) if key_indicators else "general patterns"

#         print(f"{doc_info['doc_index']:<5} {doc_info['score']:<6.3f} {doc_info['length']:<8} {indicator_str}")

#     # Show sample from highest scoring document
#     if code_docs:
#         highest_code_doc = documents[code_docs[0]["doc_index"]]
#         print(f"\nSample from highest scoring code document (score: {code_docs[0]['score']:.3f}):")
#         print("-" * 60)
#         sample = highest_code_doc[:800] + "..." if len(highest_code_doc) > 800 else highest_code_doc
#         print(sample)

# # Additional critical analyses for data filtering strategy
# print("\n" + "=" * 60)
# print("ADDITIONAL DATASET INSIGHTS FOR FILTERING STRATEGY")
# print("=" * 60)

# # 1. Document length distribution analysis
# print("\nDocument Length Distribution:")
# doc_lengths = [len(doc) for doc in documents]
# doc_token_lengths = []
# for i, doc in enumerate(documents):
#     if i < len(doc_boundaries):
#         start_idx = doc_boundaries[i - 1] + 1 if i > 0 else 0
#         end_idx = doc_boundaries[i]
#         doc_token_lengths.append(end_idx - start_idx)

# print(f"  Character lengths - Mean: {np.mean(doc_lengths):.0f}, Median: {np.median(doc_lengths):.0f}")
# print(f"  Token lengths - Mean: {np.mean(doc_token_lengths):.0f}, Median: {np.median(doc_token_lengths):.0f}")
# print(
#     f"  Short docs (<500 chars): {sum(1 for x in doc_lengths if x < 500)} ({sum(1 for x in doc_lengths if x < 500) / len(doc_lengths) * 100:.1f}%)"
# )
# print(
#     f"  Long docs (>5000 chars): {sum(1 for x in doc_lengths if x > 5000)} ({sum(1 for x in doc_lengths if x > 5000) / len(doc_lengths) * 100:.1f}%)"
# )

# # 2. Readability and complexity metrics
# import textstat

# readability_scores = []
# for doc in documents[:100]:  # Sample first 100 docs for speed
#     if len(doc.strip()) > 100:
#         try:
#             flesch = textstat.flesch_reading_ease(doc)
#             readability_scores.append(flesch)
#         except:
#             pass

# if readability_scores:
#     print(f"\nReadability Analysis (Flesch Reading Ease, sample of 100 docs):")
#     print(f"  Mean readability: {np.mean(readability_scores):.1f}")
#     print(
#         f"  Very easy (90-100): {sum(1 for x in readability_scores if x >= 90)} ({sum(1 for x in readability_scores if x >= 90) / len(readability_scores) * 100:.1f}%)"
#     )
#     print(
#         f"  Easy (80-90): {sum(1 for x in readability_scores if 80 <= x < 90)} ({sum(1 for x in readability_scores if 80 <= x < 90) / len(readability_scores) * 100:.1f}%)"
#     )
#     print(
#         f"  Standard (70-80): {sum(1 for x in readability_scores if 70 <= x < 80)} ({sum(1 for x in readability_scores if 70 <= x < 80) / len(readability_scores) * 100:.1f}%)"
#     )
#     print(
#         f"  Difficult (<70): {sum(1 for x in readability_scores if x < 70)} ({sum(1 for x in readability_scores if x < 70) / len(readability_scores) * 100:.1f}%)"
#     )

# # 3. Repetition and template detection
# print(f"\nRepetition Analysis:")
# # Check for highly repetitive n-grams that suggest template/spam content
# from collections import defaultdict


# def get_ngram_repetition(text, n=5):
#     words = text.lower().split()
#     if len(words) < n:
#         return 0
#     ngrams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
#     ngram_counts = Counter(ngrams)
#     total_ngrams = len(ngrams)
#     if total_ngrams == 0:
#         return 0
#     # Return ratio of most common ngram
#     return ngram_counts.most_common(1)[0][1] / total_ngrams if ngram_counts else 0


# repetition_scores = []
# for doc in documents[:200]:  # Sample for speed
#     if len(doc.split()) > 20:
#         rep_score = get_ngram_repetition(doc)
#         repetition_scores.append(rep_score)

# if repetition_scores:
#     print(f"  Mean 5-gram repetition ratio: {np.mean(repetition_scores):.3f}")
#     high_repetition = sum(1 for x in repetition_scores if x > 0.1)
#     print(f"  High repetition docs (>10%): {high_repetition} ({high_repetition / len(repetition_scores) * 100:.1f}%)")

# # 4. Language model perplexity estimation (using word frequency as proxy)
# print(f"\nText Quality Indicators:")
# # Check for uncommon patterns that suggest low quality
# patterns = {
#     "excessive_caps": sum(1 for doc in documents if sum(1 for c in doc if c.isupper()) / len(doc) > 0.1),
#     "excessive_numbers": sum(1 for doc in documents if sum(1 for c in doc if c.isdigit()) / len(doc) > 0.15),
#     "excessive_punct": sum(1 for doc in documents if sum(1 for c in doc if c in '.,!?;:"()[]{}') / len(doc) > 0.15),
#     "very_short_sentences": sum(
#         1
#         for doc in documents
#         if len([s for s in re.split(r"[.!?]+", doc) if len(s.strip()) < 10]) / max(1, len(re.split(r"[.!?]+", doc)))
#         > 0.5
#     ),
#     "no_punctuation": sum(1 for doc in documents if "." not in doc and "!" not in doc and "?" not in doc),
# }

# for pattern_name, count in patterns.items():
#     pct = count / len(documents) * 100
#     print(f"  {pattern_name.replace('_', ' ').title()}: {count} docs ({pct:.1f}%)")

# # 5. Vocabulary richness analysis
# print(f"\nVocabulary Analysis:")
# word_counts = Counter(full_text.lower().split())
# vocab_size = len(word_counts)
# total_words = sum(word_counts.values())
# print(f"  Total vocabulary: {vocab_size:,} unique words")
# print(f"  Total words: {total_words:,}")
# print(f"  Type-token ratio: {vocab_size / total_words:.4f}")

# # Most common words that might indicate low quality
# stop_words = [
#     "the",
#     "a",
#     "an",
#     "and",
#     "or",
#     "but",
#     "in",
#     "on",
#     "at",
#     "to",
#     "for",
#     "of",
#     "with",
#     "by",
#     "is",
#     "are",
#     "was",
#     "were",
#     "be",
#     "been",
#     "have",
#     "has",
#     "had",
#     "do",
#     "does",
#     "did",
#     "will",
#     "would",
#     "could",
#     "should",
#     "may",
#     "might",
#     "can",
#     "this",
#     "that",
#     "these",
#     "those",
# ]
# content_words = {word: count for word, count in word_counts.items() if word not in stop_words and len(word) > 2}
# print(f"  Content words (non-stop, >2 chars): {len(content_words):,}")

# # Check for spam/template indicators
# spam_indicators = [
#     "click",
#     "here",
#     "buy",
#     "now",
#     "free",
#     "offer",
#     "deal",
#     "sale",
#     "discount",
#     "limited",
#     "time",
#     "call",
#     "contact",
#     "email",
#     "website",
#     "link",
#     "follow",
#     "subscribe",
#     "join",
# ]
# spam_word_counts = {word: word_counts.get(word, 0) for word in spam_indicators}
# spam_total = sum(spam_word_counts.values())
# print(f"  Spam indicator words: {spam_total:,} ({spam_total / total_words * 100:.3f}% of all words)")
