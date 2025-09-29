import asyncio
import json
from typing import List, Dict
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator


async def fetch_urls(urls: List[str]) -> Dict[str, dict]:
    """Fetch multiple URLs using Crawl4AI and return results as a dict keyed by URL."""
    browser = BrowserConfig(
        browser_type="chromium",
        headless=True,
        viewport_width=1920,
        viewport_height=3000,
    )

    run = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        wait_until="domcontentloaded",
        wait_for="js:() => !!document.querySelector('section') && document.body.innerText.includes('Basic')",
        markdown_generator=DefaultMarkdownGenerator(),
        page_timeout=120000,
        verbose=False,
    )

    results = {}
    async with AsyncWebCrawler(config=browser) as crawler:
        tasks = [crawler.arun(url=url, config=run) for url in urls]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        for url, resp in zip(urls, responses):
            if isinstance(resp, Exception):
                results[url] = {
                    "success": False,
                    "status_code": None,
                    "error": str(resp),
                    "markdown": "",
                }
            else:
                results[url] = {
                    "success": resp.success,
                    "status_code": resp.status_code,
                    "error": resp.error_message,
                    "markdown": str(getattr(resp, "markdown", "")),
                }
    return results


def run_fetch(urls: List[str]) -> Dict[str, dict]:
    """Sync wrapper for external calls with a list of URLs."""
    return asyncio.run(fetch_urls(urls))


def fetch_and_print(urls: List[str]) -> None:
    """Helper function to fetch and print results (for testing or CLI usage)."""
    results = run_fetch(urls)
    previews = {
        url: {**res, "markdown": res["markdown"] if res["markdown"] else None}
        for url, res in results.items()
    }
    print(json.dumps(previews, ensure_ascii=False, indent=2))



###################################### Testing and Calling methods ######################################
# if __name__ == "__main__":
#     test_urls = [
#         "https://www.shopify.com/in/pricing",
#         "https://www.python.org/",
#     ]
#     fetch_and_print(test_urls)


# from crawler_module import run_fetch

# urls = [
#         "https://www.shopify.com/in/pricing",
#         "https://www.python.org/",
#     ]

# results = run_fetch(urls)

# for url, data in results.items():
#     print(f"\n--- {url} ---")
#     print(f"Success: {data['success']}")
#     print(f"Status Code: {data['status_code']}")
#     print(f"Error: {data['error']}")
#     print(f"Markdown Preview:\n{data['markdown']}...")  
