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
from cs336_data._pipeline import warc_extract_pipeline

subsample_size = 100
negative_sampling = subsample_size * 2  # rate of discard is ≈ 2x as high
dataset_path = ".data/wikipedia_HQ_urls.txt"
subsample_path = f".data/wikipedia_subsampled_{subsample_size}_urls.txt"
subsample_warc_path = subsample_path.replace("_urls.txt", ".warc.gz")


def subsample():
    import random
    from pathlib import Path

    wiki_file = Path(dataset_path)
    if not wiki_file.exists():
        raise FileNotFoundError(f"Wikipedia URLs file not found: {wiki_file}")

    with open(wiki_file, "r", encoding="utf-8") as f:
        all_lines = f.readlines()

    total_lines = len(all_lines)
    print(f"Found {total_lines} total URLs in Wikipedia file")

    if subsample_size >= total_lines:
        print(f"Target count ({subsample_size}) >= total lines ({total_lines}), returning all URLs")
        sampled_lines = all_lines
    else:
        sampled_lines = random.sample(all_lines, subsample_size)

    output_file = Path(subsample_path)
    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(sampled_lines)

    print(f"Subsampled {len(sampled_lines)} URLs to {output_file}")
    return output_file


def urls_into_warc():
    # had to be faster than warc
    import asyncio
    import aiohttp
    import gzip
    from datetime import datetime
    from pathlib import Path

    async def fetch_url(session, url, semaphore):
        async with semaphore:
            try:
                async with session.get(url.strip(), timeout=aiohttp.ClientTimeout(total=5)) as response:
                    content = await response.read()
                    return {
                        "url": url.strip(),
                        "status": response.status,
                        "content": content,
                        "headers": dict(response.headers),
                        "timestamp": datetime.utcnow().isoformat(),
                    }
            except Exception as e:
                return {
                    "url": url.strip(),
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                }

    async def download_all_urls():
        with open(subsample_path, "r") as f:
            urls = f.readlines()

        semaphore = asyncio.Semaphore(10)  # Limit concurrent requests

        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=10,
            ttl_dns_cache=300,
            use_dns_cache=True,
        )

        async with aiohttp.ClientSession(
            connector=connector, headers={"User-Agent": "Mozilla/5.0 (compatible; research-bot)"}
        ) as session:
            tasks = [fetch_url(session, url, semaphore) for url in urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Write to WARC format
        warc_path = Path(subsample_warc_path)
        with gzip.open(warc_path, "wt", encoding="utf-8") as f:
            for result in results:
                if isinstance(result, dict) and result.get("content"):
                    try:
                        content_text = result["content"].decode("utf-8", errors="ignore")
                    except:
                        content_text = str(result["content"])

                    # Proper WARC record format
                    warc_record = f"""WARC/1.0\r
WARC-Type: response\r
WARC-Target-URI: {result["url"]}\r
WARC-Date: {result["timestamp"]}\r
Content-Type: text/html\r
Content-Length: {len(content_text.encode("utf-8"))}\r
\r
{content_text}\r
\r
"""
                    f.write(warc_record)

        successful = sum(1 for r in results if isinstance(r, dict) and r.get("content"))
        print(f"Downloaded {successful}/{len(urls)} URLs successfully to {warc_path}")

    try:
        asyncio.run(download_all_urls())
    except ImportError:
        print("aiohttp not installed. Install with: pip install aiohttp")
        print(f"Fallback: wget –-timeout=5 -i {subsample_path} --warc-file={subsample_warc_path} -O /dev/null")


def create_dataset():
    pcount, ppath = warc_extract_pipeline(subsample_warc_path)  # from 100, getting 84 downloaded, then filters ~ 17
    ncount, npath = warc_extract_pipeline(".data/sample.warc.gz", subsample_count=negative_sampling)
    print("create_dataset retrieved positives, negatives:", pcount, ncount)
    print(ppath)
    print(npath)



if __name__ == "__main__":
    subsample()
    urls_into_warc()
    create_dataset()
