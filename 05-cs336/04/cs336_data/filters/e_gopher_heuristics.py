import nltk
from nltk.tokenize import word_tokenize


def gopher_filters(text: str) -> dict:
    """
    Apply Gopher quality filters to text content.

    Based on https://arxiv.org/pdf/2112.11446 Appendix A1.1

    Filters:
    - contain less than 50 words or more than 100k words
    - have a mean word length outside the range of 3 to 10 chars
    - more than 30% of lines ending with an ellipsis
    - contain less than 80% of words with at least one alphabetic character

    Args:
        text: Input text to filter

    Returns:
        dict: Contains 'pass_filter' (bool) and individual filter results
    """

    # Download required NLTK data if not already present
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    # Initialize results
    results = {
        "pass_filter": True,
        "word_count": 0,
        "mean_word_length": 0,
        "ellipsis_line_ratio": 0,
        "alphabetic_word_ratio": 0,
        "filters": {
            "word_count_filter": True,
            "mean_word_length_filter": True,
            "ellipsis_filter": True,
            "alphabetic_filter": True,
        },
    }

    if not text or not text.strip():
        results["pass_filter"] = False
        return results

    # Tokenize text into words
    words = word_tokenize(text.lower())
    word_count = len(words)
    results["word_count"] = word_count

    # Filter 1: Word count (less than 50 or more than 100,000)
    if word_count < 50 or word_count > 100000:
        results["filters"]["word_count_filter"] = False
        results["pass_filter"] = False

    if word_count > 0:
        # Filter 2: Mean word length (outside range 3-10 characters)
        alphabetic_words = [word for word in words if word.isalpha()]

        if alphabetic_words:
            mean_word_length = sum(len(word) for word in alphabetic_words) / len(alphabetic_words)
            results["mean_word_length"] = mean_word_length

            if mean_word_length < 3 or mean_word_length > 10:
                results["filters"]["mean_word_length_filter"] = False
                results["pass_filter"] = False

        # Filter 3: Lines ending with ellipsis (more than 30%)
        lines = text.split("\n")
        non_empty_lines = [line.strip() for line in lines if line.strip()]

        if non_empty_lines:
            ellipsis_lines = sum(1 for line in non_empty_lines if line.endswith("..."))
            ellipsis_ratio = ellipsis_lines / len(non_empty_lines)
            results["ellipsis_line_ratio"] = ellipsis_ratio

            if ellipsis_ratio > 0.3:
                results["filters"]["ellipsis_filter"] = False
                results["pass_filter"] = False

        # Filter 4: Alphabetic character ratio (less than 80%)
        alphabetic_word_count = sum(1 for word in words if any(c.isalpha() for c in word))
        alphabetic_ratio = alphabetic_word_count / word_count
        results["alphabetic_word_ratio"] = alphabetic_ratio

        if alphabetic_ratio < 0.6:  # 0.8
            results["filters"]["alphabetic_filter"] = False
            results["pass_filter"] = False

    return results


sample = """
  • Shows
  • CBS NEWS
  • CBS SPORTS
  • Live TV
  • Schedule
  • TV Provider
  • Sign In
  • Try Paramount+
Back to video
    • Search shows
    • Sign Up
    • Sign In
    • Shows
    • CBS NEWS
    • CBS SPORTS
    • Live TV
    • Schedule
    • TV Provider

404

Sorry to interrupt your programming...
...but the page you’re looking for doesn’t exist. You’ll be redirected to our latest episodes in 10 seconds, or you can browse your favorite CBS shows now.
CBS Home

  • Site Navigation

  • Shows
  • Live TV
  • Schedule
  • TV Provider
  • Paramount+
  • CBS News
  • CBS Sports
  • Shop

  • Privacy & Terms

  • Terms of Use
  • Privacy Policy
  • Your Privacy Choices
  • California Notice

  • Information

  • Help/Contact Us
  • Show Feedback
  • Casting
  • Closed Captioning
  • Video Description
  • Ratings Guidelines
  • About Paramount
  • Careers
  • Anti-Bias Statement

    Follow Us

Terms of Use | Privacy Policy
Do Not Sell My Personal Information |
© 2025 Paramount. All rights reserved.
    • Site Navigation
    • Shows
    • Live TV
    • Schedule
    • TV Provider
    • Paramount+
    • CBS News
    • CBS Sports
    • Shop
    • Privacy & Terms
    • Terms of Use
    • Privacy Policy
    • Your Privacy Choices
    • California Notice
    • Information
    • Help/Contact Us
    • Show Feedback
    • Casting
    • Closed Captioning
    • Video Description
    • Ratings Guidelines
    • About Paramount
    • Careers
    • Anti-Bias Statement
"""

if __name__ == "__main__":
    import json

    print(json.dumps(gopher_filters(sample), indent=2))
