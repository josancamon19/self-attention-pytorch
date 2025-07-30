import numpy as np
from transformers import AutoTokenizer
import re
from collections import Counter
import urllib.parse

tokenizer = AutoTokenizer.from_pretrained("gpt2")
validation_data = np.fromfile(
    "cs336_data/leaderboard/.data/tokenized_paloma_c4_100_domains_validation.bin", dtype=np.uint16
)
# allowed to make use of paloma validation data in constructing filters or classifiers to process the cc wet files
print(f"Total tokens: {len(validation_data):,}")  # < 10M tokens
full_text = tokenizer.decode(validation_data)
with open("cs336_data/leaderboard/.data/paloma_c4_100_domains_validation.txt", "w", encoding="utf-8") as f:
    f.write(full_text)


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
    for domain, count in domain_counts.most_common(10):
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
