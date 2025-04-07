import asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from pydantic import BaseModel, Field
from typing import Any, Dict, List
import os
from crawl4ai.extraction_strategy import LLMExtractionStrategy
import json
import re
from urllib.parse import urlparse
import glob
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time
import html2text
from pathlib import Path
from collections import defaultdict
import logging
from url_cache import URLCache  # Import our new URL cache module

# Constants
COMBINED_URLS_FILE = "data/saved_urls_combined.json"
CACHE_FILE = "data/url_cache.json"

### !!!Currently cannot crawl facebook, instagram, tiktok and google forms yet since they require login
SKIP_DOMAINS = [
    "www.facebook.com",
    "www.instagram.com",
    "www.tiktok.com",
    "35.174.52.228",
    "docs.google.com/forms",
]
SAVED_PDF_DIR = os.path.join(os.getcwd(), "documents", "pdf")
# create the pdf directory if it doesn't exist
if not os.path.exists(SAVED_PDF_DIR):
    os.makedirs(SAVED_PDF_DIR)
# Custom headers
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Connection": "keep-alive",
    "Sec-Fetch-Site": "same-origin",
    "DNT": "1",
    "Sec-GPC": "1",
    "Upgrade-Insecure-Requests": "1",
}

# """
#     return new Promise((resolve) => {
#         const totalHeight = document.body.scrollHeight;
#         let currentPosition = 0;
#         const distance = Math.min(800, totalHeight / 10); // Scroll by 800px or 1/10 of page
#         const timer = setInterval(() => {
#             window.scrollBy(0, distance);
#             currentPosition += distance;

#             // If we've reached the bottom or scrolled past it
#             if(currentPosition >= document.body.scrollHeight - window.innerHeight) {
#                 // Wait a moment at the bottom for any final loads
#                 setTimeout(() => {
#                     clearInterval(timer);
#                     resolve();
#                 }, 1500);
#             }
#         }, 300);
#     });
#     """
# Multiple commands
js_commands = [
    "window.scrollTo(0, document.body.scrollHeight);",
    "window.scrollBy(0, -100);",
]  # Scroll up a bit to trigger lazy loading

config = CrawlerRunConfig(
    # Enable full page scanning with progressive scrolling
    scan_full_page=True,
    scroll_delay=1.0,  # Delay between scroll steps (seconds)
    # Execute custom JS to scroll progressively and wait for content to load
    js_code=js_commands,
    # Other settings
    cache_mode=CacheMode.BYPASS,
    verbose=True,
    # wait_for_render_complete=True,  # Wait for page to stabilize
    # wait_for_images=True,  # Wait for images to load
    page_timeout=90000,  # Increase timeout for pages that take longer to load
)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (INFO, DEBUG, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("documents_crawl.log"),  # Log to a file
        logging.StreamHandler(),  # Log to the console
    ],
)
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)


# --- Helper Functions ---
def pp_json(json_thing: Any, sort: bool = False, indents: int = 4) -> str:
    """Pretty-prints JSON data."""
    if isinstance(json_thing, str):
        return json.dumps(
            json.loads(json_thing), sort_keys=sort, indent=indents, ensure_ascii=False
        )
    else:
        return json.dumps(
            json_thing, sort_keys=sort, indent=indents, ensure_ascii=False
        )


def safe_open_w(path: str):
    """Opens a file for writing, creating the directory if necessary."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    return open(path, "w", encoding="utf-8")


def write_file(file_path: str, content: str, overwrite: bool = True) -> None:
    """Writes content to a file, with an option to prevent overwriting."""
    if os.path.exists(file_path) and not overwrite:
        logger.info(f"File '{file_path}' already exists. Not overwriting.")
        return
    with safe_open_w(file_path) as fw:
        fw.write(content)
    logger.info(f"File '{file_path}' saved successfully.")


def load_file(file_path, file_type="json"):
    """Load a file and return the content. Default file type is JSON."""
    with open(file_path, "r", encoding="utf-8") as f:
        if file_type == "json":
            _data = json.load(f)
        else:
            _data = f.read()
    return _data


def check_html_empty(html_content):
    """
    Check if the HTML content is empty
    Returns:
        bool: True if the HTML is empty, False otherwise
    """
    # Remove comments, scripts, and whitespace
    cleaned_html = re.sub(
        r"<!--.*?-->", "", html_content, flags=re.DOTALL
    )  # Remove comments
    cleaned_html = re.sub(
        r"<script.*?</script>", "", cleaned_html, flags=re.DOTALL
    )  # Remove scripts
    cleaned_html = re.sub(r"\s+", "", cleaned_html)  # Remove all whitespace

    # Check if the cleaned HTML matches an empty HTML structure
    empty_html_pattern = r"<html><head></head><body></body></html>"
    return re.search(empty_html_pattern, cleaned_html) is not None


def url_to_filename(URL):
    # remove http:// or https:// and www. from the URL
    URL = re.sub(r"https?://", "", URL)
    URL = re.sub(r"www.", "", URL)
    # replace special characters with underscores
    file_name = re.sub(r"[^a-zA-Z0-9]", "_", URL)
    return file_name


def find_json_files(directory):
    """Find all JSON files in a directory and its subdirectories"""
    return glob.glob(f"{directory}/**/*.json", recursive=True)


def load_json_file(file_path):
    """Load a JSON file and return its contents"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


async def safe_crawl(
    id_str,
    urls,
    url_cache,
    excluded_tags=["script", "style"],
    css_selector=False,
    save_html=False,
    save_md=False,
    failed_urls_dict=None,
    delay=4,
    wait_for_pdf=15,
):
    async with AsyncWebCrawler(config=BrowserConfig(headless=False)) as crawler:
        # create a documents folder in the current directory if it doesn't exist
        output_dir = "./documents"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if save_html and not os.path.exists(f"{output_dir}/html"):
            os.makedirs(f"{output_dir}/html")
        if save_md and not os.path.exists(f"{output_dir}/txt"):
            os.makedirs(f"{output_dir}/txt")
        session_id = id_str

        for url in urls:
            # Skip already processed URLs
            if url_cache.is_url_processed(url):
                logger.info(f"URL {url} already processed, skipping...")
                continue

            # Skip URLs that contain any of the skip domains
            if any(skip_domain in url for skip_domain in SKIP_DOMAINS):
                logger.info(f"Skipping {url} as it contains a skip domain")
                # Still mark as processed to avoid rechecking
                url_cache.mark_url_processed(url)
                url_cache.save_cache()  # Save after each URL to avoid reprocessing on crashes
                continue

            logger.info(f"Start crawling {url}")

            try:
                result = await crawler.arun(
                    url=url,
                    session_id=session_id,
                    excluded_tags=excluded_tags,
                    exclude_domains=["ads.com", "trackers.com"],
                    markdown_generator=DefaultMarkdownGenerator(
                        options={
                            "protect_links": False,
                            "ignore_images": True,
                            "single_line_break": True,
                            "ignore_mailto_links": False,
                            "unicode_snob": True,
                            "use_automatic_links": True,
                            "body_width": 0,
                        }
                    ),
                    remove_overlay_elements=True,
                    css_selector=css_selector,
                    cache_mode=CacheMode.BYPASS,
                    process_iframes=True,
                    page_timeout=60000,
                    delay_before_return_html=delay,
                )
                assert (
                    result.success
                ), f"Failed to crawl {url} with CRAWL4AI, switching to selenium"
                logger.info(f"Finished crawling {url}")
                # save the markdown to a file
                base_file_name = url_to_filename(url)
                # Prepend the ID to the filename
                file_name = f"{id_str}_{base_file_name}"

                # count pdfs file
                initial_pdf_count = len(os.listdir(SAVED_PDF_DIR))

                if save_md:
                    # skip if the markdown is empty or the file already exists
                    assert (
                        result.markdown.strip()
                    ), f"Markdown is empty for {url} with CRAWL4AI, switching to selenium"
                    md_path = os.path.join(output_dir, "txt", f"{file_name}.md")
                    write_file(md_path, result.markdown, overwrite=False)
                # save the html to a file
                if save_html:
                    assert (
                        check_html_empty(result.html)
                    ) is False, (
                        f"HTML is empty for {url} with CRAWL4AI, switching to selenium"
                    )
                    html_path = os.path.join(output_dir, "html", f"{file_name}.html")
                    write_file(html_path, result.html, overwrite=False)

                # Mark URL as processed
                url_cache.mark_url_processed(url)
                # Save cache periodically
                url_cache.save_cache()

                # save links
                if result.links:
                    links_path = os.path.join(output_dir, "links", f"{file_name}.json")
                    write_file(links_path, pp_json(result.links), overwrite=False)

            except Exception as e:
                logger.warning(
                    f"Failed to crawl {url} with CRAWL4AI: , the error is: {e}, switching to selenium"
                )
                # use selenium to crawl the page
                try:
                    # For PDF downloads, we'll store in documents/pdf

                    # set up selenium
                    options = Options()
                    options.add_argument("--headless")  # Run in headless mode
                    options.add_experimental_option(
                        "prefs",
                        {
                            "download.default_directory": SAVED_PDF_DIR,
                            "download.prompt_for_download": False,
                            "plugins.always_open_pdf_externally": True,
                        },
                    )
                    selenium_driver = webdriver.Chrome(
                        service=Service(), options=options
                    )
                    selenium_driver.get(url)
                    # save the html to a file
                    base_file_name = url_to_filename(url)
                    # Prepend the ID to the filename
                    file_name = f"{id_str}_{base_file_name}"

                    success = False

                    if save_html:
                        html_source = selenium_driver.page_source
                        # Check if the HTML is empty
                        if check_html_empty(html_source):
                            logger.info(
                                f"HTML is empty for {url} with Selenium, skipping"
                            )
                        else:
                            html_path = os.path.join(
                                output_dir, "html", f"{file_name}.html"
                            )
                            write_file(html_path, html_source)
                            logger.info(
                                f"Successfully saved {url} to {file_name}.html with Selenium"
                            )
                            success = True

                    if save_md:
                        text = html2text.html2text(selenium_driver.page_source)
                        text = text.strip()
                        # Check if the markdown is empty
                        if not text:
                            logger.info(
                                f"Markdown is empty for {url} with Selenium, skipping"
                            )
                        else:
                            md_path = os.path.join(output_dir, "txt", f"{file_name}.md")
                            write_file(md_path, text)
                            logger.info(
                                f"Successfully saved {url} to {file_name}.md with Selenium"
                            )
                            success = True

                    # PDF download handling remains the same
                    # ... PDF handling code ...

                    # If any part was successful, mark the URL as processed
                    if success:
                        url_cache.mark_url_processed(url)
                        url_cache.save_cache()

                except Exception as e:
                    logger.error(
                        f"Failed to crawl {url} with Selenium: , the error is: {e}"
                    )
                    # Add to failed URLs dictionary
                    if failed_urls_dict is not None:
                        failed_urls_dict[id_str].append(url)
                    continue
                finally:
                    # Close the selenium driver after `wait_for_pdf` seconds to wait for the download to finish
                    logger.info("Closing the selenium driver")
                    time.sleep(wait_for_pdf)

                    try:
                        # Get list of files in the download directory
                        recount_pdf = len(os.listdir(SAVED_PDF_DIR))

                        # If no files are found, wait a bit longer and check again
                        if recount_pdf <= initial_pdf_count:
                            logger.info(
                                "No files found initially, waiting 10 more seconds..."
                            )
                            time.sleep(10)  # Wait an additional 10 seconds
                            downloaded_files = os.listdir(SAVED_PDF_DIR)  # Check again

                            # Get the most recently downloaded file if any new files appeared
                            if len(downloaded_files) > initial_pdf_count:
                                # Get files sorted by creation time
                                recent_files = sorted(
                                    [
                                        os.path.join(SAVED_PDF_DIR, f)
                                        for f in downloaded_files
                                    ],
                                    key=os.path.getctime,
                                    reverse=True,
                                )
                                if recent_files:
                                    downloaded_file = recent_files[0]
                                    # Check if the file looks like a PDF or other download we might expect
                                    if os.path.isfile(
                                        downloaded_file
                                    ) and not downloaded_file.endswith(".part"):
                                        # Append the ID to the filename
                                        filename = os.path.basename(downloaded_file)
                                        new_file_name = f"{id_str}_{filename}"
                                        new_file_path = os.path.join(
                                            SAVED_PDF_DIR, new_file_name
                                        )

                                        # Rename the file
                                        if not os.path.exists(new_file_path):
                                            os.rename(downloaded_file, new_file_path)
                                            logger.info(
                                                f"Renamed downloaded file to {new_file_name}"
                                            )
                                        else:
                                            logger.info(
                                                f"File {new_file_name} already exists, not renaming"
                                            )
                                    else:
                                        logger.warning(
                                            "No valid download detected or download incomplete"
                                        )
                            else:
                                logger.warning(
                                    "No new files detected even after extended wait"
                                )
                        else:
                            downloaded_files = os.listdir(SAVED_PDF_DIR)
                            # Get the most recently downloaded file
                            # Get files sorted by creation time
                            recent_files = sorted(
                                [
                                    os.path.join(SAVED_PDF_DIR, f)
                                    for f in downloaded_files
                                ],
                                key=os.path.getctime,
                                reverse=True,
                            )
                            if recent_files:
                                downloaded_file = recent_files[0]
                                # Check if the file looks like a PDF or other download we might expect
                                if os.path.isfile(
                                    downloaded_file
                                ) and not downloaded_file.endswith(".part"):
                                    # Append the ID to the filename
                                    filename = os.path.basename(downloaded_file)
                                    new_file_name = f"{id_str}_{filename}"
                                    new_file_path = os.path.join(
                                        SAVED_PDF_DIR, new_file_name
                                    )

                                    # Rename the file
                                    if not os.path.exists(new_file_path):
                                        os.rename(downloaded_file, new_file_path)
                                        logger.info(
                                            f"Renamed downloaded file to {new_file_name}"
                                        )
                                    else:
                                        logger.info(
                                            f"File {new_file_name} already exists, not renaming"
                                        )
                                else:
                                    logger.warning(
                                        "No valid download detected or download incomplete"
                                    )

                    except Exception as e:
                        logger.error(f"Error handling downloaded file: {e}")
                    finally:
                        # Always make sure to close the driver
                        selenium_driver.close()


# Load data from saved URLs files
logger.info("Loading saved URLs files from Step 1...")

if os.path.exists(COMBINED_URLS_FILE):
    urls_dict = load_file(COMBINED_URLS_FILE)
else:
    logger.error(f"File {COMBINED_URLS_FILE} does not exist.")
    exit(1)

# Initialize URL cache
url_cache = URLCache(CACHE_FILE)

# Initialize dictionary to track failed URLs
failed_urls_dict = defaultdict(list)

# Process each ID and its URLs
for id_str, urls in urls_dict.items():
    # Skip already completed IDs
    if url_cache.is_id_completed(id_str):
        logger.info(f"ID {id_str} already processed, skipping...")
        continue

    # Get unprocessed URLs for this ID
    unprocessed_urls = url_cache.get_unprocessed_urls(id_str, urls)

    if not unprocessed_urls:
        logger.info(f"All URLs for ID {id_str} already processed, marking as completed")
        url_cache.mark_id_completed(id_str)
        url_cache.save_cache()
        continue

    logger.info(
        f"Start crawling for ID {id_str} ({len(unprocessed_urls)}/{len(urls)} URLs remaining)"
    )
    # excluded_tags = ["script", "style", "head", "footer"]
    asyncio.run(
        safe_crawl(
            id_str,
            unprocessed_urls,
            url_cache,
            save_md=True,
            save_html=True,
            excluded_tags=None,
            failed_urls_dict=failed_urls_dict,
            delay=4,  # delay in seconds between crawls, adjust this as needed to avoid getting blocked
            wait_for_pdf=10,  # wait for 10 seconds for the PDF to download, adjust this as needed
        )
    )

    # Mark the ID as completed after processing all its URLs
    url_cache.mark_id_completed(id_str)
    url_cache.save_cache()
    logger.info(f"ID {id_str} marked as completed")

# Save failed URLs to a JSON file

if failed_urls_dict:
    failed_urls_dict = {k: v for k, v in failed_urls_dict.items() if v}
    write_file("failed_urls.json", pp_json(failed_urls_dict))
    logger.info("Crawling complete. Failed URLs have been saved to failed_urls.json")
else:
    logger.info("No failed URLs found. All URLs crawled successfully.")
