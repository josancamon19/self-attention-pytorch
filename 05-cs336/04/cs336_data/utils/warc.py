import asyncio
import aiohttp
import gzip
from datetime import datetime
from pathlib import Path


def retrieve_multiple_urls_to_warc(urls_path, warc_path):
    # receives websites URL's, and parses them into warc
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
        with open(urls_path) as f:
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
        warc_path = Path(warc_path)
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
        print(f"Fallback: wget –-timeout=5 -i {urls_path} --warc-file={warc_path} -O /dev/null")
