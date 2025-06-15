"""
Kickstarter Projects Scraper

A clean, maintainable scraper for extracting Kickstarter project data.
Features: caching, rate limiting, pagination, error handling, and logging.
"""

import html
import json
import logging
import os
import re
import time
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, NamedTuple
from urllib.parse import urlparse
import urllib.request
import urllib.error

import undetected_chromedriver as uc
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field, ConfigDict
from tqdm import tqdm
from webdriver_manager.chrome import ChromeDriverManager


# =============================================================================
# CONFIGURATION & DATA MODELS
# =============================================================================

class ScrapingConfig(BaseModel):
    """Configuration for scraping operations."""
    
    model_config = ConfigDict(frozen=True, validate_assignment=True)
    
    # Paths
    output_dir: str = Field(default='projects', description="Output directory")
    cache_file: str = Field(default='cache.json', description="Cache file")
    
    # Timing
    request_delay: float = Field(default=0.5, ge=0.1, description="Delay between requests")
    page_load_delay: int = Field(default=1, ge=0, description="Page load wait time")
    retry_delay: int = Field(default=300, ge=60, description="Retry delay (seconds)")
    max_retries: int = Field(default=5, ge=1, description="Max retry attempts")
    
    # Browser
    headless: bool = Field(default=False, description="Run headless")
    chrome_version: str | None = Field(default=None, description="Chrome version")
    
    # Features
    scrape_comments: bool = Field(default=True, description="Scrape comments")
    scrape_updates: bool = Field(default=True, description="Scrape updates")
    scrape_rewards: bool = Field(default=True, description="Scrape rewards")
    
    # Logging
    debug_mode: bool = Field(default=False, description="Enable debug logging")


class ProjectData(NamedTuple):
    """Project data container."""
    project_id: Optional[str]
    url: str
    title: str
    data: Dict[str, Any]
    html_content: str


class APIResponse(NamedTuple):
    """API response container."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    status_code: Optional[int] = None


class ScrapingResult(NamedTuple):
    """Final scraping results."""
    total_urls: int
    successful: int
    failed: int
    cached: int
    duration: float
    log_file: str


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def setup_logging(config: ScrapingConfig) -> str:
    """Setup logging configuration."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f"scraper_{timestamp}.log"
    
    level = logging.DEBUG if config.debug_mode else logging.INFO
    
    # Clear existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Setup file and console logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Logging initialized: {log_file}")
    return log_file


def create_safe_filename(text: str) -> str:
    """Create filesystem-safe filename."""
    if not text:
        return "unnamed"
    return re.sub(r'[^\w\-_.]', '_', str(text)[:100])


def url_to_filename(url: str) -> str:
    """Convert URL to safe filename."""
    if not url:
        return "empty_url"
    
    parsed = urlparse(url)
    netloc = parsed.netloc.replace('.', '_')
    path = parsed.path.strip('/').replace('/', '_')
    
    return f"{netloc}_{path}" if path else netloc


def ensure_dir(path: str) -> bool:
    """Ensure directory exists."""
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logging.error(f"Failed to create directory {path}: {e}")
        return False


# =============================================================================
# FILE I/O FUNCTIONS
# =============================================================================

def read_json(filepath: str) -> Optional[Dict[str, Any]]:
    """Read JSON file safely."""
    try:
        if Path(filepath).exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logging.error(f"Failed to read {filepath}: {e}")
    return None


def write_json(data: Any, filepath: str) -> bool:
    """Write JSON file safely."""
    try:
        ensure_dir(str(Path(filepath).parent))
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logging.error(f"Failed to write {filepath}: {e}")
        return False


def write_html(content: str, filepath: str) -> bool:
    """Write HTML file safely."""
    try:
        ensure_dir(str(Path(filepath).parent))
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except Exception as e:
        logging.error(f"Failed to write HTML {filepath}: {e}")
        return False


# =============================================================================
# CACHE MANAGEMENT
# =============================================================================

class Cache:
    """Simple cache manager."""
    
    def __init__(self, config: ScrapingConfig):
        self.config = config
        self.data = self._load()
    
    def _load(self) -> Dict[str, List[str]]:
        """Load cache from file."""
        cache_data = read_json(self.config.cache_file)
        if cache_data:
            return {
                'urls': cache_data.get('urls', []),
                'project_ids': cache_data.get('project_ids', []),
                'failed_urls': cache_data.get('failed_urls', [])
            }
        
        logging.info("Creating new cache")
        return {'urls': [], 'project_ids': [], 'failed_urls': []}
    
    def save(self) -> None:
        """Save cache to file."""
        write_json(self.data, self.config.cache_file)
    
    def is_processed(self, url: str) -> bool:
        """Check if URL already processed."""
        return url in self.data['urls'] or url in self.data['failed_urls']
    
    def add_success(self, url: str, project_id: Optional[str] = None) -> None:
        """Add successful URL to cache."""
        if url not in self.data['urls']:
            self.data['urls'].append(url)
        
        if project_id and str(project_id) not in self.data['project_ids']:
            self.data['project_ids'].append(str(project_id))
    
    def add_failure(self, url: str) -> None:
        """Add failed URL to cache."""
        if url not in self.data['failed_urls']:
            self.data['failed_urls'].append(url)


# =============================================================================
# HTTP & API FUNCTIONS
# =============================================================================

def make_api_request(url: str, timeout: float = 30) -> APIResponse:
    """Make HTTP request to API endpoint."""
    if not url:
        return APIResponse(False, error="Empty URL")
    
    try:        
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0 (compatible; scraper/1.0)')
        req.add_header('Accept', 'application/json')
        
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read().decode('utf-8'))
            return APIResponse(True, data=data, status_code=response.getcode())
    
    except urllib.error.HTTPError as e:
        return APIResponse(False, error=f"HTTP {e.code}: {e.reason}", status_code=e.code)
    except Exception as e:
        return APIResponse(False, error=str(e))


def retry_api_request(url: str, config: ScrapingConfig) -> APIResponse:
    """Retry API request with backoff."""
    for attempt in range(config.max_retries + 1):
        response = make_api_request(url)
        
        if response.success:
            return response
        
        # Don't retry on client errors
        if response.status_code == 401:
            logging.error(f"401 Unauthorized for {url}. URL signature has likely expired.")
            break
        elif response.status_code == 429:
            logging.warning(f"Received 429 Too Many Requests for {url}. "
                            f"Retrying in {config.request_delay} seconds...")
            time.sleep(config.request_delay)
            continue

        # Retry on server errors or network issues
        if attempt < config.max_retries:
            wait_time = config.retry_delay * (2 ** attempt)  # Exponential backoff
            logging.warning(f"Request failed, retrying in {wait_time}s: {response.error}")
            time.sleep(wait_time)
    
    return response


# =============================================================================
# HTML PARSING
# =============================================================================

def extract_project_data(html_content: str) -> Optional[Dict[str, Any]]:
    """Extract project data from HTML."""
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        script_tags = soup.find_all('script')
        
        # Look for window.current_project
        for script in script_tags:
            if script.string and 'window.current_project' in script.string:
                match = re.search(r'window\.current_project\s*=\s*"(.*?)";', script.string)
                if match:
                    json_str = html.unescape(match.group(1)).replace('\\"', '"')
                    data = json.loads(json_str)
                    logging.info("Extracted project data from window.current_project")
                    return data
        
        # Fallback: look for preload data
        for script in script_tags:
            if script.string and 'preload' in script.string:
                match = re.search(r'preload\s*\(\s*".*?project.*?"\s*,\s*({.*?})\)', script.string)
                if match:
                    data = json.loads(html.unescape(match.group(1)))
                    logging.info("Extracted project data from preload")
                    return data
        
        logging.warning("No project data found in HTML")
        return None
    
    except Exception as e:
        logging.error(f"Error extracting project data: {e}")
        return None


# =============================================================================
# WEBDRIVER MANAGEMENT
# =============================================================================

class WebDriverManager:
    """Manage Chrome WebDriver lifecycle."""
    
    def __init__(self, config: ScrapingConfig):
        self.config = config
        self.driver = None
    
    def __enter__(self):
        self.driver = self._create_driver()
        return self.driver
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.driver:
            try:
                self.driver.quit()
                logging.info("WebDriver closed")
            except:
                pass
    
    def _create_driver(self) -> uc.Chrome:
        """Create Chrome WebDriver."""
        try:
            options = uc.ChromeOptions()
            options.add_argument("--disable-notifications")
            options.add_argument("--disable-popup-blocking")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            
            if self.config.headless:
                options.add_argument('--headless')
            
            driver = uc.Chrome(
                driver_executable_path=ChromeDriverManager(
                    driver_version=self.config.chrome_version
                ).install(),
                options=options
            )
            
            logging.info("WebDriver initialized")
            return driver
        
        except Exception as e:
            logging.error(f"Failed to create WebDriver: {e}")
            raise


def scrape_page(driver: uc.Chrome, url: str, config: ScrapingConfig) -> Tuple[str, str]:
    """Scrape single page and return title and HTML."""
    try:
        driver.get(url)
        time.sleep(config.page_load_delay)
        
        # Scroll to load dynamic content
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        # time.sleep(1)
        
        title = driver.title or "No Title"
        html_content = driver.page_source
        
        return title, html_content
    
    except Exception as e:
        logging.error(f"Error scraping {url}: {e}")
        return "Error", ""


# =============================================================================
# PROJECT PROCESSING
# =============================================================================

def save_project_files(project: ProjectData, output_dir: str) -> bool:
    """Save project files to disk."""
    project_dir = Path(output_dir) / (project.project_id or create_safe_filename(project.url))
    
    if not ensure_dir(str(project_dir)):
        return False
    
    # Save main files
    html_success = write_html(project.html_content, str(project_dir / "page.html"))
    json_success = write_json(project.data, str(project_dir / "data.json"))
    
    # Save metadata
    metadata = {
        'project_id': project.project_id,
        'url': project.url,
        'title': project.title,
        'scraped_at': datetime.now().isoformat(),
        'data_keys': list(project.data.keys()) if project.data else [],
        'data_apis': {
            'comments': project.data.get('urls', {}).get('api', {}).get('comments'),
            'updates': project.data.get('urls', {}).get('api', {}).get('updates'),
            'rewards': project.data.get('urls', {}).get('api', {}).get('rewards'),
            'creator': project.data.get('creator', {}).get('urls', {}).get('api', {}).get('user') 
        }
    }
    meta_success = write_json(metadata, str(project_dir / "metadata.json"))
    
    success = html_success and json_success and meta_success
    if success:
        logging.info(f"Saved project files to {project_dir}")
    
    return success


def process_api_endpoint(url: str, name: str, project_dir: Path, config: ScrapingConfig) -> bool:
    """Process single API endpoint."""
    if not url:
        logging.info(f"No URL for {name}")
        return True
    
    logging.info(f"Processing {name}: {url}")
    # handle comments with pagination separately
    if name == 'comments':
        logging.info(f"Starting comments scraping for {name} at {url}")
        # Here you would implement the logic to handle comments pagination
        comment_data = []
        base_reponse = retry_api_request(url, config)
        if base_reponse.success:
            reponse = base_reponse.data
            while len(reponse.get('comments', [])) > 0:
                logging.info(f"Found {len(reponse['comments'])} comments in this repose.")
                comment_data.append(reponse) 
                more_comments_url = reponse.get('urls', {}).get('api', {}).get('more_comments', {})
                if not more_comments_url:
                    logging.info(f"No more comments found")
                    break
                api_reponse = retry_api_request(more_comments_url, config)
                reponse = api_reponse.data if api_reponse.success else {}
        else:
            logging.error(f"Failed to fetch comments: {base_reponse.error}")
            return False
        # write comments to file
        filename = f"{create_safe_filename(name)}.json"
        filepath = project_dir / filename
        success = write_json(comment_data, str(filepath))
        if success:
            logging.info(f"Saved {name} data")
        return success
    elif name == 'updates':
        logging.info(f"Starting updates scraping for {name} at {url}")
        # handle updates with pagination separately
        update_data = []
        base_response = retry_api_request(url, config)
        if base_response.success:
            response = base_response.data
            while len(response.get('updates', [])) > 0:
                update_data.extend(response.get('updates', []))
                more_updates_url = response.get('urls', {}).get('api', {}).get('more_updates', {})
                if not more_updates_url:
                    break
                api_response = retry_api_request(more_updates_url, config)
                response = api_response.data if api_response.success else {}
        else:
            logging.error(f"Failed to fetch updates: {base_response.error}")
            return False
        # write updates to file
        filename = f"{create_safe_filename(name)}.json"
        filepath = project_dir / filename
        success = write_json(update_data, str(filepath))
        if success:
            logging.info(f"Saved {name} data")
        return success
    # for other endpoints like rewards, creator, etc.
    logging.info(f"Fetching {name} data from {url}")

    response = retry_api_request(url, config)

    if response.success:
        filename = f"{create_safe_filename(name)}.json"
        filepath = project_dir / filename
        success = write_json(response.data, str(filepath))
        if success:
            logging.info(f"Saved {name} data")
        return success
    # else:
    #     logging.error(f"Failed to fetch {name}: {response.error}")
    #     return False


def process_project_apis(project: ProjectData, output_dir: str, config: ScrapingConfig) -> bool:
    """Process all API endpoints for a project."""
    if 'urls' not in project.data or 'api' not in project.data['urls']:
        logging.info("No API URLs found")
        return True
    
    project_dir = Path(output_dir) / (project.project_id or create_safe_filename(project.url))
    api_urls = project.data['urls']['api']
    
    results = []
    
    # Process configured endpoints
    if config.scrape_comments and 'comments' in api_urls:
        # results.append(process_api_endpoint(api_urls['comments'], 'comments', project_dir, config))
        results.append(process_api_endpoint(api_urls['comments'], 'comments', project_dir, config))
        
    
    if config.scrape_updates and 'updates' in api_urls:
        results.append(process_api_endpoint(api_urls['updates'], 'updates', project_dir, config))
    
    # if config.scrape_rewards and 'rewards' in api_urls:
    #     results.append(process_api_endpoint(api_urls['rewards'], 'rewards', project_dir, config))
    
    # Process creator data
    # creator_url = project.data.get('creator', {}).get('urls', {}).get('api', {}).get('user')
    # if creator_url:
    #     results.append(process_api_endpoint(creator_url, 'creator', project_dir, config))
    
    return all(results) if results else True


def process_single_url(url: str, driver: uc.Chrome, cache: Cache, config: ScrapingConfig) -> bool:
    """Process a single URL completely."""
    logging.info(f"Processing: {url}")
    
    try:
        # Scrape page
        title, html_content = scrape_page(driver, url, config)
        
        if not html_content:
            logging.error("Failed to get HTML content")
            cache.add_failure(url)
            return False
        
        # Extract project data
        project_data = extract_project_data(html_content)
        
        if not project_data:
            logging.warning("No project data found")
            # Save HTML for debugging
            debug_dir = Path(config.output_dir) / "debug" / create_safe_filename(url)
            ensure_dir(str(debug_dir))
            write_html(html_content, str(debug_dir / "page.html"))
            cache.add_failure(url)
            return False
        
        # Create project object
        project_id = str(project_data.get('id')) if project_data.get('id') else None
        project = ProjectData(
            project_id=project_id,
            url=url,
            title=title,
            data=project_data,
            html_content=html_content
        )
        
        # Save files and process APIs
        files_saved = save_project_files(project, config.output_dir)
        apis_processed = process_project_apis(project, config.output_dir, config)
        
        success = files_saved and apis_processed
        
        if success:
            cache.add_success(url, project_id)
            logging.info(f"Successfully processed: {url}")
        else:
            cache.add_failure(url)
            logging.error(f"Failed to process: {url}")
        
        return success
    
    except Exception as e:
        logging.error(f"Error processing {url}: {e}")
        cache.add_failure(url)
        return False


# =============================================================================
# MAIN SCRAPING FUNCTION
# =============================================================================

def scrape_kickstarter_projects(urls: List[str], config: Optional[ScrapingConfig] = None) -> ScrapingResult:
    """
    Main function to scrape Kickstarter project URLs.
    
    Args:
        urls: List of Kickstarter project URLs
        config: Optional configuration object
    
    Returns:
        ScrapingResult with statistics
    """
    # Setup
    config = config or ScrapingConfig()
    log_file = setup_logging(config)
    start_time = time.time()
    
    logging.info(f"Starting scraper with {len(urls)} URLs")
    
    # Validate URLs
    valid_urls = [url for url in urls if url and url.strip() and url.startswith('http')]
    if len(valid_urls) != len(urls):
        logging.warning(f"Filtered {len(urls) - len(valid_urls)} invalid URLs")
    
    if not valid_urls:
        logging.error("No valid URLs provided")
        return ScrapingResult(0, 0, 0, 0, 0.0, log_file)
    
    # Setup cache and filter
    cache = Cache(config)
    new_urls = [url for url in valid_urls if not cache.is_processed(url)]
    cached_count = len(valid_urls) - len(new_urls)
    
    logging.info(f"URLs: {len(valid_urls)} total, {len(new_urls)} new, {cached_count} cached")
    
    if not new_urls:
        logging.info("All URLs already processed")
        duration = time.time() - start_time
        return ScrapingResult(len(valid_urls), 0, 0, cached_count, duration, log_file)
    
    # Ensure output directory
    ensure_dir(config.output_dir)
    
    # Process URLs
    successful = 0
    failed = 0
    
    try:
        with WebDriverManager(config) as driver:
            for i, url in enumerate(tqdm(new_urls, desc="Scraping URLs")):
                logging.info(f"\n--- Processing {i+1}/{len(new_urls)} ---")
                
                if process_single_url(url, driver, cache, config):
                    successful += 1
                else:
                    failed += 1
                
                # Save cache periodically
                if (i + 1) % 5 == 0:
                    cache.save()
                
                # Rate limiting
                if i < len(new_urls) - 1:
                    time.sleep(config.request_delay)
    
    except Exception as e:
        logging.error(f"Scraping error: {e}")
    
    finally:
        cache.save()
    
    # Final results
    duration = time.time() - start_time
    result = ScrapingResult(
        total_urls=len(valid_urls),
        successful=successful,
        failed=failed,
        cached=cached_count,
        duration=duration,
        log_file=log_file
    )
    
    logging.info(f"\n{'='*50}")
    logging.info(f"SCRAPING COMPLETED")
    logging.info(f"Total: {result.total_urls}, Success: {result.successful}, "
                f"Failed: {result.failed}, Cached: {result.cached}")
    logging.info(f"Duration: {result.duration:.1f}s")
    logging.info(f"{'='*50}")
    
    return result


# =============================================================================
# LEGACY COMPATIBILITY
# =============================================================================

def get_projects(urls: List[str], base_output_path: str = 'projects', 
                cache_file: str = 'cache.json') -> Dict[str, List[str]]:
    """Legacy function for backward compatibility."""
    config = ScrapingConfig(output_dir=base_output_path, cache_file=cache_file)
    result = scrape_kickstarter_projects(urls, config)
    
    cache = Cache(config)
    return {
        'urls': cache.data['urls'],
        'project_ids': cache.data['project_ids']
    }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example configuration
    config = ScrapingConfig(
        output_dir=r"../data/kickstarter_projects",
        request_delay=0.5,
        headless=False,
        debug_mode=False,
        scrape_comments=True,
        scrape_updates=True,
        max_retries=3
    )
    
    # Example URLs
    test_urls = [
        "https://www.kickstarter.com/projects/ankermake/eufymake-e1-the-first-personal-3d-textured-uv-printer",
        "https://www.kickstarter.com/projects/pixnlovegames/nightmare-busters-rebirth-0"
    ]

    # final_df = pd.read_csv(r'/mnt/c/Users/giahu/OneDrive - Michigan State University/Projects/KickstarterRes/scraper/data/out/kickstarter_final_data_fixed.csv')
    # project_urls = final_df['project_url'].tolist()
    
    
    # Run scraper
    result = scrape_kickstarter_projects(test_urls, config)
    print(f"Scraping completed: {result.successful}/{result.total_urls} successful")
