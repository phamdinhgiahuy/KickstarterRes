import html
import json
import os
import re
import time
import urllib.request
import urllib.parse
import logging
from datetime import datetime
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from webdriver_manager.chrome import ChromeDriverManager
from tqdm import tqdm
import undetected_chromedriver as uc
import pandas as pd
import json


def setup_logging():
    """Configure logging with timestamp-based filename"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"kickstarter_scraper_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=log_filename,
        filemode="w",
    )

    # Add a console handler to also log to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)

    logging.info(f"Logging to file: {log_filename}")
    return log_filename


def load_cache(cache_file="scraped_urls_cache.json"):
    """
    Load the cache of already scraped URLs

    Args:
        cache_file (str): Path to the cache file

    Returns:
        dict: Dictionary containing lists of urls and project_ids
    """
    try:
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                cache = json.load(f)
                logging.info(
                    f"Loaded cache with {len(cache['urls'])} URLs and {len(cache['project_ids'])} project IDs"
                )
                return cache
        logging.info(f"No cache file found at {cache_file}, creating new cache")
        return {"urls": [], "project_ids": []}
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding cache file: {e}")
        return {"urls": [], "project_ids": []}
    except IOError as e:
        logging.error(f"I/O error loading cache: {e}")
        return {"urls": [], "project_ids": []}
    except Exception as e:
        logging.error(f"Unexpected error loading cache: {e}")
        return {"urls": [], "project_ids": []}


def save_cache(cache, url=None, project_id=None, cache_file="scraped_urls_cache.json"):
    """
    Save the cache of scraped URLs and project IDs

    Args:
        cache (dict): Cache dictionary with 'urls' and 'project_ids' lists
        url (str, optional): URL to add to cache
        project_id (str or int, optional): Project ID to add to cache
        cache_file (str): Path to the cache file
    """
    if not cache or not isinstance(cache, dict):
        logging.error("Invalid cache object provided")
        return

    try:
        # Ensure the cache has the required structure
        if "urls" not in cache:
            cache["urls"] = []
        if "project_ids" not in cache:
            cache["project_ids"] = []

        if url and url not in cache["urls"]:
            cache["urls"].append(url)
            logging.debug(f"Added URL to cache: {url}")

        if project_id and str(project_id) not in cache["project_ids"]:
            cache["project_ids"].append(str(project_id))
            logging.debug(f"Added project ID to cache: {project_id}")

        # Write to a temporary file first, then rename for atomicity
        temp_file = f"{cache_file}.tmp"
        with open(temp_file, "w") as f:
            json.dump(cache, f, indent=2)
            f.flush()
            os.fsync(f.fileno())  # Ensure data is written to disk

        # Replace the original file with the temporary file
        os.replace(temp_file, cache_file)
        logging.debug(f"Cache saved to {cache_file}")

    except TypeError as e:
        logging.error(f"Error encoding cache to JSON: {e}")
    except IOError as e:
        logging.error(f"I/O error saving cache: {e}")
    except Exception as e:
        logging.error(f"Unexpected error saving cache: {e}")


def url_to_filename(url):
    """
    Convert a URL to a valid filename.
    """
    parsed = urlparse(url)
    # Get the path without leading and trailing slashes
    path = parsed.path.strip("/")
    # Replace slashes with underscores
    path = path.replace("/", "_")
    # Add the netloc (domain) as a prefix
    filename = f"{parsed.netloc}_{path}"
    # Remove any other characters that aren't suitable for filenames
    filename = re.sub(r"[^\w\-_.]", "_", filename)
    return filename


def refresh_project_api_urls(project_url, driver=None):
    """Get fresh API URLs from project page when signatures expire"""
    logging.info(f"Refreshing API URLs from {project_url}")

    close_driver = False
    if not driver:
        close_driver = True
        options = uc.ChromeOptions()
        driver = uc.Chrome(
            driver_executable_path=ChromeDriverManager().install(), options=options
        )

    try:
        driver.get(project_url)
        time.sleep(2)

        html_source = driver.page_source
        soup = BeautifulSoup(html_source, "html.parser")

        # Extract project data using same logic as in get_projects function
        script_tags = soup.find_all("script")
        for script in script_tags:
            if script.string and "window.current_project" in script.string:
                match = re.search(
                    r'window\.current_project\s*=\s*"(.*?)";', script.string
                )
                if match:
                    json_str = html.unescape(match.group(1))
                    json_str = json_str.replace('\\"', '"')

                    try:
                        project_data = json.loads(json_str)
                        logging.info("Successfully extracted fresh API URLs from page")

                        # Return only the API URLs section
                        if "urls" in project_data and "api" in project_data["urls"]:
                            return project_data["urls"]["api"]
                    except:
                        pass

        return None
    finally:
        if close_driver:
            driver.quit()


def get_api_json(url, file_name, project_dir, timeout=1):
    """
    Save API JSON data from a URL to a file in the project directory.
    Handles 429 Too Many Requests and 401 Unauthorized errors with retry logic.
    """
    # Skip if URL is empty or None
    if not url:
        logging.warning(f"Skipping empty URL for {file_name}")
        return None

    max_retries = 10
    retry_count = 0
    retry_delay = 360  # 6 minutes in seconds

    while retry_count <= max_retries:
        try:
            logging.info(
                f"Fetching API data from: {url} (Attempt {retry_count+1}/{max_retries+1})"
            )

            req = urllib.request.Request(url)
            req.add_header(
                "User-Agent",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:106.0) Gecko/20100101 Firefox/106.0",
            )
            req.add_header("Accept-Language", "en-US,en;q=0.5")
            req.add_header("Accept", "application/json")

            # Add delay to avoid rate limiting
            time.sleep(timeout)

            response = urllib.request.urlopen(req)
            r = response.read().decode("utf-8")
            data = json.loads(r)

            # Make file name safe
            safe_file_name = re.sub(r"[^\w\-_.]", "_", file_name)
            file_path = os.path.join(project_dir, f"{safe_file_name}.json")

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)

            logging.info(f"Successfully saved API data to {file_path}")
            return data

        except urllib.error.HTTPError as e:
            if e.code == 429:
                retry_count += 1
                if retry_count <= max_retries:
                    logging.warning(
                        f"Received 429 Too Many Requests for {file_name}. "
                        f"Retry {retry_count}/{max_retries} after {retry_delay} seconds..."
                    )
                    time.sleep(retry_delay)
                    continue
                else:
                    logging.error(
                        f"Max retries ({max_retries}) reached for {file_name} after 429 errors"
                    )
                    return None
            elif e.code == 401:
                logging.error(
                    f"401 Unauthorized for {file_name}. URL signature has likely expired."
                )
                # No retry for 401 - needs fresh signature from the page
                return None
            else:
                logging.error(f"HTTP Error for {file_name}: {e.code} - {e.reason}")
        except urllib.error.URLError as e:
            logging.error(f"URL Error for {file_name}: {e.reason}")
        except json.JSONDecodeError as e:
            logging.error(f"JSON Decode Error for {file_name}: {e}")
        except Exception as e:
            logging.error(f"Error fetching {file_name}: {str(e)}")

        # If we get here, there was an error that wasn't a 429, so we don't retry
        return None


def get_projects(
    urls, base_output_path="projects", cache_file="scraped_urls_cache.json"
):
    """
    Get project data from a list of URLs and save files in folders organized by project ID.
    Skip URLs that have already been scraped.
    """
    # Setup logging first
    log_file = setup_logging()

    # Load the cache
    cache = load_cache(cache_file)

    try:
        # Create base output directory if it doesn't exist
        if not os.path.exists(base_output_path):
            os.makedirs(base_output_path)
            logging.info(f"Created output directory: {base_output_path}")

        # Initialize Chrome
        logging.info("Initializing Chrome driver")
        options = uc.ChromeOptions()
        # Uncomment for headless mode if needed
        # options.add_argument('--headless')

        driver = uc.Chrome(
            driver_executable_path=ChromeDriverManager().install(), options=options
        )

        # Filter URLs to only process new ones
        new_urls = [url for url in urls if url and url not in cache["urls"]]
        skipped_urls = len(urls) - len(new_urls)

        if skipped_urls > 0:
            logging.info(f"Skipping {skipped_urls} already scraped URLs")

        logging.info(f"Processing {len(new_urls)} new URLs")

        for url in tqdm(new_urls, desc="Processing URLs", unit="url"):
            # Skip if URL is empty or None
            if not url:
                logging.warning("Skipping empty URL")
                continue

            try:
                logging.info(f"\nProcessing URL: {url}")

                # Navigate to the URL
                driver.get(url)
                logging.info(f"Page title: {driver.title}")

                # Wait for page to load
                time.sleep(3)

                html_source = driver.page_source
                soup = BeautifulSoup(html_source, "html.parser")

                # Find project data in script tags
                script_tags = soup.find_all("script")
                project_data = None

                # Look for window.current_project in scripts
                for script in script_tags:
                    if script.string and "window.current_project" in script.string:
                        # Extract the JSON string
                        match = re.search(
                            r'window\.current_project\s*=\s*"(.*?)";', script.string
                        )
                        if match:
                            # Get the JSON string and unescape HTML entities
                            json_str = html.unescape(match.group(1))
                            json_str = json_str.replace('\\"', '"')

                            try:
                                # Parse the JSON string
                                project_data = json.loads(json_str)
                                logging.info(
                                    "Successfully extracted project data from page"
                                )
                                break
                            except json.JSONDecodeError as e:
                                logging.error(f"Error parsing project JSON: {e}")

                # If no project data was found, try an alternative method
                if not project_data:
                    logging.warning("Trying alternative extraction method...")
                    for script in script_tags:
                        if script.string and "preloaded" in script.string:
                            try:
                                # Look for project data in preloaded JSON
                                match = re.search(
                                    r'preload\s*\(\s*"(.*?)"\s*,\s*({.*?})\)',
                                    script.string,
                                )
                                if match and "project" in match.group(1):
                                    json_str = html.unescape(match.group(2))
                                    project_data = json.loads(json_str)
                                    logging.info(
                                        "Successfully extracted project data from preload"
                                    )
                                    break
                            except Exception as e:
                                logging.error(f"Error with alternative extraction: {e}")

                if project_data:
                    # Get project ID or use URL-based name as fallback
                    project_id = project_data.get("id")

                    # Check if project_id is already in cache
                    if project_id and str(project_id) in cache["project_ids"]:
                        logging.info(
                            f"Project ID {project_id} already scraped, skipping..."
                        )
                        save_cache(cache, url=url, project_id=None)  # Add URL to cache
                        continue

                    if project_id:
                        logging.info(f"Found project ID: {project_id}")
                        # Create project directory using the ID
                        project_dir = os.path.join(base_output_path, str(project_id))
                    else:
                        # Fallback to URL-based name if no ID
                        base_file_name = url_to_filename(url)
                        logging.warning(
                            f"No project ID found, using URL-based name: {base_file_name}"
                        )
                        project_dir = os.path.join(base_output_path, base_file_name)

                    # Create the project directory
                    if not os.path.exists(project_dir):
                        os.makedirs(project_dir)
                        logging.info(f"Created project directory: {project_dir}")

                    # Save the HTML file
                    html_file = os.path.join(project_dir, "project_page.html")
                    with open(html_file, "w", encoding="utf-8") as f:
                        f.write(html_source)
                    logging.info(f"Saved HTML to {html_file}")

                    # Save the main project data
                    project_file = os.path.join(project_dir, "project_data.json")
                    with open(project_file, "w", encoding="utf-8") as f:
                        json.dump(project_data, f, indent=4)
                    logging.info(f"Saved project data to {project_file}")

                    # Extract API URLs
                    logging.info("Extracting API URLs...")

                    # Get project API URLs
                    if "urls" in project_data and "api" in project_data["urls"]:
                        for key, value in project_data["urls"]["api"].items():
                            if key == "comments":
                                logging.info(f"{key}: {value}")
                                json_data = get_api_json(value, key, project_dir)
                                page = 0

                                # Fix for comments pagination
                                while (
                                    json_data
                                    and "comments" in json_data
                                    and len(json_data["comments"]) > 0
                                    and "urls" in json_data
                                    and "api" in json_data["urls"]
                                    and "more_comments" in json_data["urls"]["api"]
                                ):
                                    page += 1
                                    logging.info(f"Fetching comments page {page}...")
                                    json_data = get_api_json(
                                        json_data["urls"]["api"]["more_comments"],
                                        f"{key}_page{page}",
                                        project_dir,
                                    )
                                    if not json_data:
                                        break

                            elif key == "updates":
                                logging.info(f"{key}: {value}")
                                json_data = get_api_json(value, key, project_dir)
                                page = 0

                                # Fix for updates pagination
                                while (
                                    json_data
                                    and "updates" in json_data
                                    and len(json_data["updates"]) > 0
                                    and "urls" in json_data
                                    and "api" in json_data["urls"]
                                    and "more_updates" in json_data["urls"]["api"]
                                ):
                                    page += 1
                                    logging.info(f"Fetching updates page {page}...")
                                    json_data = get_api_json(
                                        json_data["urls"]["api"]["more_updates"],
                                        f"{key}_page{page}",
                                        project_dir,
                                    )
                                    if not json_data:
                                        break
                            else:
                                logging.info(f"{key}: {value}")
                                get_api_json(value, key, project_dir)

                    if (
                        "creator" in project_data
                        and "urls" in project_data["creator"]
                        and "api" in project_data["creator"]["urls"]
                    ):
                        creator_url = project_data["creator"]["urls"]["api"].get("user")
                        if creator_url:
                            logging.info(f"Creator URL: {creator_url}")
                            creator_data = get_api_json(
                                creator_url, "creator", project_dir
                            )

                            # If creator data fetch failed and it might be due to expired signature
                            if (
                                not creator_data
                                and "urls" in project_data
                                and "web" in project_data["urls"]
                                and "project" in project_data["urls"]["web"]
                            ):
                                project_web_url = project_data["urls"]["web"]["project"]
                                logging.info(
                                    f"Creator API fetch failed, refreshing URLs from {project_web_url}"
                                )

                                # Get fresh project data with updated signatures
                                fresh_api_urls = refresh_project_api_urls(
                                    project_web_url, driver
                                )
                                if fresh_api_urls:
                                    # Try to get fresh project data that contains creator info
                                    fresh_project_data = get_api_json(
                                        fresh_api_urls["project"],
                                        "refreshed_project",
                                        project_dir,
                                    )
                                    if (
                                        fresh_project_data
                                        and "creator" in fresh_project_data
                                    ):
                                        if (
                                            "urls" in fresh_project_data["creator"]
                                            and "api"
                                            in fresh_project_data["creator"]["urls"]
                                        ):
                                            fresh_creator_url = fresh_project_data[
                                                "creator"
                                            ]["urls"]["api"].get("user")
                                            if fresh_creator_url:
                                                logging.info(
                                                    f"Trying with fresh creator URL: {fresh_creator_url}"
                                                )
                                                get_api_json(
                                                    fresh_creator_url,
                                                    "creator",
                                                    project_dir,
                                                )

                    # Get reward URLs and process them
                    if "rewards" in project_data:
                        # Create rewards directory
                        rewards_dir = os.path.join(project_dir, "rewards")
                        if not os.path.exists(rewards_dir):
                            os.makedirs(rewards_dir)
                            logging.info(f"Created rewards directory: {rewards_dir}")

                        for i, reward in enumerate(project_data.get("rewards", [])):
                            if "urls" in reward and "api" in reward["urls"]:
                                # Check for various API URL formats
                                reward_api_url = None
                                if isinstance(reward["urls"]["api"], str):
                                    reward_api_url = reward["urls"]["api"]
                                elif (
                                    isinstance(reward["urls"]["api"], dict)
                                    and "reward" in reward["urls"]["api"]
                                ):
                                    reward_api_url = reward["urls"]["api"]["reward"]

                                if reward_api_url and isinstance(reward_api_url, str):
                                    # Save individual reward data to rewards directory
                                    logging.info(
                                        f"Processing reward {i+1}: {reward_api_url}"
                                    )
                                    reward_data = get_api_json(
                                        reward_api_url, f"reward_{i+1}", rewards_dir
                                    )

                                    # If reward has its own ID, also save by ID
                                    if reward_data and "id" in reward_data:
                                        reward_id = reward_data["id"]
                                        reward_id_path = os.path.join(
                                            rewards_dir, f"reward_{reward_id}.json"
                                        )
                                        with open(
                                            reward_id_path, "w", encoding="utf-8"
                                        ) as f:
                                            json.dump(reward_data, f, indent=4)
                                            logging.debug(
                                                f"Saved reward by ID: {reward_id}"
                                            )

                    # After successful processing, update the cache
                    save_cache(
                        cache, url=url, project_id=project_id, cache_file=cache_file
                    )
                    logging.info(f"Added URL and project_id {project_id} to cache")

                else:
                    logging.warning("No project data found on the page")

                    # Create a fallback directory using URL
                    base_file_name = url_to_filename(url)
                    project_dir = os.path.join(
                        base_output_path, f"unknown_{base_file_name}"
                    )

                    if not os.path.exists(project_dir):
                        os.makedirs(project_dir)
                        logging.info(f"Created fallback directory: {project_dir}")

                    # Save the HTML for debugging
                    html_file = os.path.join(project_dir, "project_page.html")
                    with open(html_file, "w", encoding="utf-8") as f:
                        f.write(html_source)
                    logging.info(f"Saved HTML to {html_file} for debugging")

                    # Still mark URL as processed
                    save_cache(cache, url=url, project_id=None, cache_file=cache_file)

            except Exception as e:
                logging.error(f"Error processing {url}: {str(e)}", exc_info=True)
                # Don't add to cache if there was an error

            # Add delay between URLs
            time.sleep(1)

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)
    finally:
        # Clean up
        driver.quit()
        logging.info("Driver closed")
        logging.info(f"Scraping completed. Log saved to {log_file}")

    return cache


demo_df = pd.read_csv("Kickstarter001.csv")
# filter only state = 'successful' and 'failed'
demo_df_filter = demo_df[demo_df["state"].isin(["successful", "failed"])]
demo_df_filter
urls = map(json.loads, demo_df.urls)
urls = list(urls)
project_urls = list(set([url["web"]["project"] for url in urls]))
print(len(project_urls), project_urls)


if __name__ == "__main__":
    # Test with a single URL
    get_projects(project_urls, base_output_path="kickstarter_data")
