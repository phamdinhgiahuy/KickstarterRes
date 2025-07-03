from __future__ import annotations
import os
import glob
import logging
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
from functools import partial
import concurrent.futures
import time

# Import your existing extraction function
from step1_get_api_data import setup_logging as api_setup_logging

# Import other necessary components
import json
from bs4 import BeautifulSoup
from datetime import datetime


# Import or define your models - this is missing and will cause errors
# For example:


from typing import Any, List, Optional

from pydantic import BaseModel, Field


class PhotoProject(BaseModel):
    key: Optional[str] = None
    full: Optional[str] = None
    ed: Optional[str] = None
    med: Optional[str] = None
    little: Optional[str] = None
    small: Optional[str] = None
    thumb: Optional[str] = None
    field_1024x576: Optional[str] = Field(None, alias="1024x576")
    field_1536x864: Optional[str] = Field(None, alias="1536x864")


class WebProject(BaseModel):
    discover: Optional[str] = None
    location: Optional[str] = None


class ApiProject(BaseModel):
    nearby_projects: Optional[str] = None


class UrlsProject(BaseModel):
    web: Optional[WebProject] = None
    api: Optional[ApiProject] = None


class LocationProject(BaseModel):
    id: Optional[int] = None
    name: Optional[str] = None
    slug: Optional[str] = None
    short_name: Optional[str] = None
    displayable_name: Optional[str] = None
    localized_name: Optional[str] = None
    country: Optional[str] = None
    state: Optional[str] = None
    type: Optional[str] = None
    is_root: Optional[bool] = None
    expanded_country: Optional[str] = None
    urls: Optional[UrlsProject] = None


class Web1Project(BaseModel):
    discover: Optional[str] = None


class Urls1Project(BaseModel):
    web: Optional[Web1Project] = None


class CategoryProject(BaseModel):
    id: Optional[int] = None
    name: Optional[str] = None
    analytics_name: Optional[str] = None
    slug: Optional[str] = None
    position: Optional[int] = None
    parent_id: Optional[int] = None
    parent_name: Optional[str] = None
    color: Optional[int] = None
    urls: Optional[Urls1Project] = None


class VideoProject(BaseModel):
    id: Optional[int] = None
    status: Optional[str] = None
    hls: Optional[str] = None
    hls_type: Optional[str] = None
    high: Optional[str] = None
    high_type: Optional[str] = None
    base: Optional[str] = None
    base_type: Optional[str] = None
    tracks: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    frame: Optional[str] = None


class ImageUrlsProject(BaseModel):
    default: Optional[str] = None
    baseball_card: Optional[str] = None


class FeatureImageAttributes(BaseModel):
    image_urls: Optional[ImageUrlsProject] = None


class ProfileProject(BaseModel):
    id: Optional[int] = None
    project_id: Optional[int] = None
    state: Optional[str] = None
    state_changed_at: Optional[int] = None
    name: Optional[Any] = None
    blurb: Optional[Any] = None
    background_color: Optional[Any] = None
    text_color: Optional[Any] = None
    link_background_color: Optional[Any] = None
    link_text_color: Optional[Any] = None
    link_text: Optional[Any] = None
    link_url: Optional[Any] = None
    show_feature_image: Optional[bool] = None
    background_image_opacity: Optional[float] = None
    should_show_feature_image_section: Optional[bool] = None
    feature_image_attributes: Optional[FeatureImageAttributes] = None


class Web2Project(BaseModel):
    project: Optional[str] = None
    rewards: Optional[str] = None
    pledge_redemption: Optional[str] = None
    project_short: Optional[str] = None
    updates: Optional[str] = None


class Api1Project(BaseModel):
    project: Optional[str] = None
    comments: Optional[str] = None
    updates: Optional[str] = None


class Urls2Project(BaseModel):
    web: Optional[Web2Project] = None
    api: Optional[Api1Project] = None


class ItemProject(BaseModel):
    id: Optional[int] = None
    project_id: Optional[int] = None
    name: Optional[str] = None
    amount: Optional[int] = None
    taxable: Optional[bool] = None
    edit_path: Optional[str] = None


class ModelProject(BaseModel):
    id: Optional[int] = None
    photo: Optional[PhotoProject] = None
    name: Optional[str] = None
    blurb: Optional[str] = None
    goal: Optional[float] = None
    pledged: Optional[float] = None
    state: Optional[str] = None
    slug: Optional[str] = None
    disable_communication: Optional[bool] = None
    country: Optional[str] = None
    country_displayable_name: Optional[str] = None
    currency: Optional[str] = None
    currency_symbol: Optional[str] = None
    currency_trailing_code: Optional[bool] = None
    deadline: Optional[int] = None
    state_changed_at: Optional[int] = None
    created_at: Optional[int] = None
    launched_at: Optional[int] = None
    is_in_post_campaign_pledging_phase: Optional[Any] = None
    staff_pick: Optional[bool] = None
    is_starrable: Optional[bool] = None
    backers_count: Optional[int] = None
    static_usd_rate: Optional[float] = None
    usd_pledged: Optional[str] = None
    converted_pledged_amount: Optional[int] = None
    fx_rate: Optional[float] = None
    usd_exchange_rate: Optional[float] = None
    current_currency: Optional[str] = None
    usd_type: Optional[str] = None
    location: Optional[LocationProject] = None
    category: Optional[CategoryProject] = None
    video: Optional[VideoProject] = None
    profile: Optional[ProfileProject] = None
    spotlight: Optional[bool] = None
    urls: Optional[Urls2Project] = None
    updated_at: Optional[int] = None
    failed_at: Optional[int] = None
    comments_count: Optional[int] = None
    updates_count: Optional[int] = None
    tags: Optional[List] = None
    add_ons: Optional[List] = None
    items: Optional[List[ItemProject]] = None
    prelaunch_activated: Optional[bool] = None
    display_prelaunch: Optional[bool] = None
    available_card_types: Optional[List[str]] = None
    supports_addons: Optional[bool] = None
    addons_pledge_url: Optional[str] = None


class AvatarCreator(BaseModel):
    thumb: Optional[str] = None
    small: Optional[str] = None
    medium: Optional[str] = None
    large: Optional[str] = None


class WebCreator(BaseModel):
    user: Optional[str] = None


class ApiCreator(BaseModel):
    user: Optional[str] = None
    created_projects: Optional[str] = None


class UrlsCreator(BaseModel):
    web: Optional[WebCreator] = None
    api: Optional[ApiCreator] = None


class ModelCreator(BaseModel):
    id: Optional[int] = None
    name: Optional[str] = None
    slug: Optional[str] = None
    is_registered: Optional[Any] = None
    is_email_verified: Optional[Any] = None
    chosen_currency: Optional[Any] = None
    is_superbacker: Optional[Any] = None
    has_admin_message_badge: Optional[bool] = None
    ppo_has_action: Optional[bool] = None
    backing_action_count: Optional[int] = None
    avatar: Optional[AvatarCreator] = None
    is_ksr_admin: Optional[bool] = None
    urls: Optional[UrlsCreator] = None
    backed_projects: Optional[int] = None
    join_date: Optional[str] = None
    location: Optional[Any] = None
    biography: Optional[str] = None
    backed_projects_count: Optional[int] = None
    created_projects_count: Optional[int] = None
    is_admin: Optional[bool] = None
    updated_at: Optional[int] = None
    created_at: Optional[int] = None


# Now define your setup_logging function
def setup_logging(log_level=logging.INFO):
    """Configure logging with timestamp-based filename"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"kickstarter_data_{timestamp}.log"

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=log_filename,
        filemode="w",
    )

    # Add console handler to also log to console
    console = logging.StreamHandler()
    console.setLevel(log_level)
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)

    logging.info(f"Logging configured. Log file: {log_filename}")
    return log_filename


# Add a test function to verify the extraction process works correctly
def test_extraction(project_dir, project_id):
    """Test the extraction process for a single project"""
    print(f"Testing extraction for project {project_id}...")
    result = extract_one_campaign_data(project_dir, project_id)

    if result:
        print(f"✓ Successfully extracted data for project {project_id}")
        print(f"  Found {len(result)} fields of data")
        # Print a few key fields for verification
        for field in ["project_id", "project_name", "project_goal", "project_state"]:
            if field in result:
                print(f"  {field}: {result[field]}")
            else:
                print(f"  {field}: Not found")
        return True
    else:
        print(f"✗ Failed to extract data for project {project_id}")
        return False


# Add a verification function to the main script
def verify_environment():
    """Verify that the environment is properly set up for parallel processing"""
    print("Verifying parallel processing environment...")

    # Check Python version
    import sys

    print(f"Python version: {sys.version}")
    if sys.version_info < (3, 6):
        print("⚠️ Warning: Python 3.6+ recommended for this script")

    # Check CPU count
    cpu_count = mp.cpu_count()
    print(f"Available CPU cores: {cpu_count}")

    # Check if project directory exists
    project_dir = "kickstarter_data"
    if os.path.exists(project_dir):
        project_count = len(
            [d for d in glob.glob(os.path.join(project_dir, "*")) if os.path.isdir(d)]
        )
        print(f"Found {project_count} project directories in {project_dir}")
    else:
        print(f"⚠️ Warning: Project directory {project_dir} not found")

    # Test multiprocessing
    print("Testing multiprocessing...")
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
            future = executor.submit(lambda: "Multiprocessing test successful")
            result = future.result(timeout=5)
            print(f"✓ {result}")
    except Exception as e:
        print(f"⚠️ Multiprocessing test failed: {str(e)}")

    print("Verification complete!")


def load_project_files(project_dir):
    """Load all project files from the directory structure"""
    logging.info(f"Loading project files from {project_dir}")
    project_dirs = glob.glob(os.path.join(project_dir, "*"))
    project_files = {}

    for dir in project_dirs:
        project_id = os.path.basename(dir)
        project_files[project_id] = {}

        # Define expected file paths
        project_json_path = os.path.join(dir, "project.json")
        creator_json_path = os.path.join(dir, "creator.json")
        html_path = os.path.join(dir, "project_page.html")

        # Check if files exist
        project_files[project_id]["project_json"] = (
            project_json_path if os.path.exists(project_json_path) else None
        )
        project_files[project_id]["creator_json"] = (
            creator_json_path if os.path.exists(creator_json_path) else None
        )
        project_files[project_id]["project_html"] = (
            html_path if os.path.exists(html_path) else None
        )

        # Log which files were found
        if not project_files[project_id]["project_json"]:
            logging.warning(f"Project JSON not found for {project_id}")
        if not project_files[project_id]["creator_json"]:
            logging.warning(f"Creator JSON not found for {project_id}")
        if not project_files[project_id]["project_html"]:
            logging.warning(f"Project HTML not found for {project_id}")

    logging.info(f"Found {len(project_files)} projects")
    return project_files


def load_json_file(json_path):
    """Load and parse JSON file with error handling"""
    try:
        if not json_path or not os.path.exists(json_path):
            logging.warning(f"File not found: {json_path}")
            return None

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logging.debug(f"Successfully loaded JSON from {json_path}")
        return data
    except FileNotFoundError:
        logging.warning(f"File not found: {json_path}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from file: {json_path} - {str(e)}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error loading JSON file {json_path}: {str(e)}")
        return None


def make_soup(html_path):
    """Create BeautifulSoup object from HTML file"""
    try:
        if not html_path or not os.path.exists(html_path):
            logging.warning(f"HTML file not found: {html_path}")
            return None

        with open(html_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
        logging.debug(f"Successfully parsed HTML from {html_path}")
        return soup
    except FileNotFoundError:
        logging.warning(f"File not found: {html_path}")
        return None
    except Exception as e:
        logging.error(f"Error loading HTML file {html_path}: {str(e)}")
        return None


PROJECT_PREFIX = "https://www.kickstarter.com/projects/"


def pydantic_to_dict(project_model, creator_model):
    """Convert Pydantic models to dictionary, safely handling None values."""
    logging.debug("Converting Pydantic models to dictionary")

    # Helper function to safely get nested attributes
    def safe_get(obj, *attrs):
        """Get nested attribute safely, returning None if any attribute in the chain is None"""
        current = obj
        for attr in attrs:
            if current is None:
                return None
            try:
                current = getattr(current, attr)
            except AttributeError:
                return None
        return current

    # Create dictionary with safe attribute access
    pydantic_dict = {}
    # ---Campaign Data---
    pydantic_dict["project_id"] = safe_get(project_model, "id")
    pydantic_dict["project_name"] = safe_get(project_model, "name")
    pydantic_dict["project_blurb"] = safe_get(project_model, "blurb")
    pydantic_dict["project_url"] = safe_get(project_model, "urls", "web", "project")
    pydantic_dict["project_photo"] = safe_get(project_model, "photo", "full")
    pydantic_dict["project_goal"] = safe_get(project_model, "goal")
    pydantic_dict["project_pledged"] = safe_get(project_model, "pledged")
    pydantic_dict["project_state"] = safe_get(project_model, "state")
    pydantic_dict["project_disable_communication"] = safe_get(
        project_model, "disable_communication"
    )
    pydantic_dict["project_country"] = safe_get(project_model, "country")
    pydantic_dict["project_country_displayable_name"] = safe_get(
        project_model, "country_displayable_name"
    )
    pydantic_dict["project_currency"] = safe_get(project_model, "currency")
    pydantic_dict["project_currency_symbol"] = safe_get(
        project_model, "currency_symbol"
    )
    pydantic_dict["project_currency_trailing_code"] = safe_get(
        project_model, "currency_trailing_code"
    )
    pydantic_dict["project_deadline_at"] = safe_get(project_model, "deadline")
    pydantic_dict["project_state_changed_at"] = safe_get(
        project_model, "state_changed_at"
    )
    pydantic_dict["project_created_at"] = safe_get(project_model, "created_at")
    pydantic_dict["project_launched_at"] = safe_get(project_model, "launched_at")
    pydantic_dict["project_is_in_post_campaign_pledging_phase"] = safe_get(
        project_model, "is_in_post_campaign_pledging_phase"
    )
    pydantic_dict["project_staff_pick"] = safe_get(project_model, "staff_pick")
    pydantic_dict["project_is_starrable"] = safe_get(project_model, "is_starrable")
    pydantic_dict["project_backers_count"] = safe_get(project_model, "backers_count")
    pydantic_dict["project_static_usd_rate"] = safe_get(
        project_model, "static_usd_rate"
    )
    pydantic_dict["project_usd_pledged"] = safe_get(project_model, "usd_pledged")
    pydantic_dict["project_converted_pledged_amount"] = safe_get(
        project_model, "converted_pledged_amount"
    )
    pydantic_dict["project_fx_rate"] = safe_get(project_model, "fx_rate")
    pydantic_dict["project_usd_exchange_rate"] = safe_get(
        project_model, "usd_exchange_rate"
    )
    pydantic_dict["project_current_currency"] = safe_get(
        project_model, "current_currency"
    )
    pydantic_dict["project_usd_type"] = safe_get(project_model, "usd_type")

    # Location fields with safe access
    pydantic_dict["project_location_id"] = safe_get(project_model, "location", "id")
    pydantic_dict["project_location_name"] = safe_get(
        project_model, "location", "short_name"
    )
    pydantic_dict["project_location_state"] = safe_get(
        project_model, "location", "state"
    )
    pydantic_dict["project_location_country"] = safe_get(
        project_model, "location", "country"
    )
    pydantic_dict["project_location_type"] = safe_get(project_model, "location", "type")

    # Category fields with safe access
    pydantic_dict["project_category_id"] = safe_get(project_model, "category", "id")
    pydantic_dict["project_category_name"] = safe_get(project_model, "category", "name")
    pydantic_dict["project_parent_category_id"] = safe_get(
        project_model, "category", "parent_id"
    )
    pydantic_dict["project_parent_category_name"] = safe_get(
        project_model, "category", "parent_name"
    )

    # Video fields with safe access
    pydantic_dict["project_video_high"] = safe_get(project_model, "video", "high")
    pydantic_dict["project_video_hls"] = safe_get(project_model, "video", "hls")

    pydantic_dict["project_is_spotlight"] = safe_get(project_model, "spotlight")
    pydantic_dict["project_updated_at"] = safe_get(project_model, "updated_at")
    pydantic_dict["project_failed_at"] = safe_get(project_model, "failed_at")
    pydantic_dict["project_comments_count"] = safe_get(project_model, "comments_count")
    pydantic_dict["project_updates_count"] = safe_get(project_model, "updates_count")

    # Special handling for list/collection fields
    pydantic_dict["project_tags"] = safe_get(project_model, "tags")
    pydantic_dict["project_add_ons"] = safe_get(project_model, "add_ons")
    pydantic_dict["project_available_card_types"] = safe_get(
        project_model, "available_card_types"
    )

    pydantic_dict["project_prelaunch_activated"] = safe_get(
        project_model, "prelaunch_activated"
    )
    pydantic_dict["project_display_prelaunch"] = safe_get(
        project_model, "display_prelaunch"
    )
    pydantic_dict["project_supports_addons"] = safe_get(
        project_model, "supports_addons"
    )
    pydantic_dict["project_addons_pledge_url"] = safe_get(
        project_model, "addons_pledge_url"
    )

    # Creator fields with safe access
    if creator_model is None:
        logging.warning("Creator model is None, skipping creator fields")
    else:
        pydantic_dict["project_creator_id"] = safe_get(creator_model, "id")
        pydantic_dict["project_creator_name"] = safe_get(creator_model, "name")
        pydantic_dict["project_creator_url"] = safe_get(
            creator_model, "urls", "web", "user"
        )
        # pydantic_dict['project_creator_is_registered'] = safe_get(creator_model, 'is_registered')
        # pydantic_dict['project_creator_is_email_verified'] = safe_get(creator_model, 'is_email_verified')
        # pydantic_dict['project_creator_chosen_currency'] = safe_get(creator_model, 'chosen_currency')
        # pydantic_dict['project_creator_is_superbacker'] = safe_get(creator_model, 'is_superbacker')
        pydantic_dict["project_creator_backing_action_count"] = safe_get(
            creator_model, "backing_action_count"
        )
        pydantic_dict["project_creator_avatar"] = safe_get(
            creator_model, "avatar", "large"
        )
        pydantic_dict["project_creator_backed_projects"] = safe_get(
            creator_model, "backed_projects"
        )
        pydantic_dict["project_creator_join_date"] = safe_get(
            creator_model, "join_date"
        )
        pydantic_dict["project_creator_location"] = safe_get(creator_model, "location")
        pydantic_dict["project_creator_biography"] = safe_get(
            creator_model, "biography"
        )
        pydantic_dict["project_creator_backed_projects_count"] = safe_get(
            creator_model, "backed_projects_count"
        )
        pydantic_dict["project_creator_created_projects_count"] = safe_get(
            creator_model, "created_projects_count"
        )
        pydantic_dict["project_creator_is_admin"] = safe_get(creator_model, "is_admin")
        pydantic_dict["project_creator_updated_at"] = safe_get(
            creator_model, "updated_at"
        )
        pydantic_dict["project_creator_created_at"] = safe_get(
            creator_model, "created_at"
        )

    return pydantic_dict


def extract_text_information(soup):
    text_info = {}

    # Extract content from key sections
    section_mappings = [
        ("story_content", "div.story-content"),
        ("risks_content", "#risks-and-challenges"),
        ("environment_commitments", "#environmentalCommitments"),
    ]

    for field_name, selector in section_mappings:
        section = soup.select_one(selector)
        if section:
            # Extract text from all elements, not just <p> tags
            # This is more comprehensive than just getting <p> tags
            text_info[field_name] = section.get_text(separator="\n\n", strip=True)

            # Count words
            word_count = len(text_info[field_name].split())
            text_info[f"{field_name}_word_count"] = word_count

            # Extract images if this is the story section
            if field_name == "story_content":
                images = section.find_all("img")
                image_urls = [img.get("src", "") for img in images if img.get("src")]
                text_info["story_image_urls"] = image_urls
                text_info["story_image_count"] = len(image_urls)

    return text_info


def extract_one_campaign_data(project_dir, project_id):
    """Extract all data for a single campaign without reloading all projects"""
    logging.info(f"Extracting data for project {project_id}")

    # Direct path access instead of loading all projects
    project_path = os.path.join(project_dir, project_id)

    # Check if project directory exists
    if not os.path.exists(project_path):
        logging.error(f"Project directory not found: {project_path}")
        return None

    # Direct file paths
    project_json_path = os.path.join(project_path, "project.json")
    creator_json_path = os.path.join(project_path, "creator.json")
    html_path = os.path.join(project_path, "project_page.html")

    # Load JSON files
    project_json = load_json_file(project_json_path)
    creator_json = load_json_file(creator_json_path)

    if not project_json:
        logging.error(f"Could not load project JSON for {project_id}")
        return None

    try:
        # Convert JSONs to Pydantic models
        project_model = ModelProject(**project_json)
        creator_model = ModelCreator(**creator_json) if creator_json else None

        # Convert models to dictionary
        final_dict = pydantic_to_dict(project_model, creator_model)

        # Extract text information
        soup = make_soup(html_path)
        if soup:
            text_info = extract_text_information(soup)
            final_dict.update(text_info)
        else:
            logging.warning(f"No HTML data extracted for project {project_id}")

        logging.info(f"Successfully extracted data for project {project_id}")
        return final_dict

    except Exception as e:
        logging.error(f"Error processing project {project_id}: {str(e)}", exc_info=True)
        return None


def process_project(project_dir, project_id):
    """
    Process a single project and return the extracted data
    This wrapper function handles exceptions for multiprocessing
    """
    try:
        # Import here to avoid multiprocessing issues

        result = extract_one_campaign_data(project_dir, project_id)
        return {"project_id": project_id, "success": result is not None, "data": result}
    except Exception as e:
        return {
            "project_id": project_id,
            "success": False,
            "error": str(e),
            "data": None,
        }


def parallel_process_projects(project_dir, max_workers=None, batch_size=100):
    """
    Process all projects in parallel using a process pool

    Args:
        project_dir: Base directory containing project folders
        max_workers: Maximum number of parallel processes (defaults to CPU count)
        batch_size: Number of projects to process in each batch

    Returns:
        pandas.DataFrame: Combined dataframe with all project data
    """
    # Setup logging
    log_file = setup_logging()
    logging.info(f"Starting parallel processing with max_workers={max_workers}")

    # Find all project directories
    project_ids = [
        os.path.basename(d)
        for d in glob.glob(os.path.join(project_dir, "*"))
        if os.path.isdir(d)
    ]
    total_projects = len(project_ids)
    logging.info(f"Found {total_projects} project directories to process")

    # Initialize counters
    successful_projects = 0
    failed_projects = 0
    start_time = time.time()

    # Process in batches to show progress and manage memory
    all_results = []

    # Create a partial function with the project_dir parameter fixed
    process_func = partial(process_project, project_dir)

    # Determine optimal number of workers if not specified
    if not max_workers:
        max_workers = min(mp.cpu_count(), 16)  # Limit to avoid overwhelming the system

    logging.info(f"Using {max_workers} worker processes")

    # Process projects in batches
    for i in range(0, total_projects, batch_size):
        batch = project_ids[i : i + batch_size]
        logging.info(
            f"Processing batch {i//batch_size + 1}/{(total_projects+batch_size-1)//batch_size}, "
            f"projects {i+1}-{min(i+batch_size, total_projects)}"
        )

        # Use ThreadPoolExecutor for I/O bound tasks, or ProcessPoolExecutor for CPU bound tasks
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:
            # Process the batch with progress bar
            results = list(
                tqdm(
                    executor.map(process_func, batch),
                    total=len(batch),
                    desc=f"Batch {i//batch_size + 1}",
                )
            )

            # Accumulate results
            all_results.extend(results)

            # Update counters
            for result in results:
                if result["success"]:
                    successful_projects += 1
                else:
                    failed_projects += 1
                    logging.error(
                        f"Failed to process project {result['project_id']}: {result.get('error', 'Unknown error')}"
                    )

        # Log progress after each batch
        elapsed = time.time() - start_time
        projects_per_second = (i + len(batch)) / elapsed if elapsed > 0 else 0
        estimated_total = (
            total_projects / projects_per_second if projects_per_second > 0 else 0
        )
        remaining = estimated_total - elapsed if estimated_total > 0 else 0

        logging.info(
            f"Processed {i + len(batch)}/{total_projects} projects "
            f"({successful_projects} successful, {failed_projects} failed)"
        )
        logging.info(
            f"Speed: {projects_per_second:.2f} projects/second, "
            f"Est. remaining time: {remaining/60:.1f} minutes"
        )

    # Convert successful results to a DataFrame
    logging.info("Converting results to DataFrame...")
    successful_data = [
        r["data"] for r in all_results if r["success"] and r["data"] is not None
    ]

    if not successful_data:
        logging.warning("No successful data was extracted!")
        return pd.DataFrame()

    # Create DataFrame once with all data
    combined_df = pd.DataFrame.from_records(successful_data)

    # Log completion
    total_time = time.time() - start_time
    logging.info(
        f"Processing complete in {total_time/60:.1f} minutes. "
        f"Successful: {successful_projects}, Failed: {failed_projects}"
    )
    logging.info(
        f"Final dataframe has {len(combined_df)} rows and {len(combined_df.columns)} columns"
    )
    logging.info(f"Speed improved from 12+ hours to {total_time/60:.1f} minutes!")

    return combined_df


def process_and_save_data(
    project_dir="kickstarter_data",
    output_dir="output_dataframe",
    max_workers=None,
    batch_size=100,
):
    """
    Process all projects, save the results to CSV and Excel, and return the DataFrame
    """
    # Make sure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process the data
    df = parallel_process_projects(project_dir, max_workers, batch_size)

    if df.empty:
        logging.warning("No data to save!")
        return df

    # Generate timestamp for filenames
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    # Save to CSV and Excel
    csv_path = os.path.join(output_dir, f"kickstarter_data_parallel_{timestamp}.csv")
    excel_path = os.path.join(output_dir, f"kickstarter_data_parallel_{timestamp}.xlsx")

    logging.info(f"Saving data to {csv_path}...")
    df.to_csv(csv_path, index=False)

    logging.info(f"Saving data to {excel_path}...")
    df.to_excel(excel_path, index=False)

    logging.info("Data saved successfully!")
    return df


if __name__ == "__main__":
    # First verify the environment
    verify_environment()

    # Parse command line arguments
    import argparse

    parser = argparse.ArgumentParser(
        description="Process Kickstarter project data in parallel"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run a test on a single project before processing",
    )
    parser.add_argument(
        "--run", action="store_true", help="Run the full parallel processing"
    )
    parser.add_argument(
        "--dir", default="kickstarter_data", help="Directory containing project data"
    )
    parser.add_argument(
        "--output", default="output_dataframe", help="Directory to save output files"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: CPU count)",
    )
    parser.add_argument(
        "--batch", type=int, default=1000, help="Batch size for processing"
    )

    args = parser.parse_args()

    project_dir = args.dir

    # Find available project directories
    project_ids = [
        os.path.basename(d)
        for d in glob.glob(os.path.join(project_dir, "*"))
        if os.path.isdir(d)
    ]

    if not project_ids:
        print(f"No project directories found in {project_dir}. Please check your data.")
        exit(1)

    if args.test:
        # Test with the first project
        test_id = project_ids[0]
        if test_extraction(project_dir, test_id):
            print("\nTest successful! Ready to run the full parallel processing.")
        else:
            print(
                "\nTest failed. Please fix the extraction issues before running parallel processing."
            )
            exit(1)

    if args.run:
        print(f"Starting parallel processing of {len(project_ids)} projects...")
        result_df = process_and_save_data(
            project_dir=project_dir,
            output_dir=args.output,
            max_workers=args.workers,
            batch_size=args.batch,
        )
        print(f"Processing complete! DataFrame contains {len(result_df)} rows.")

    if not args.test and not args.run:
        print(
            "No action specified. Use --test to run a test or --run to process all data."
        )
        print(
            "Example: python parallel_process.py --test --run --dir kickstarter_data --workers 4"
        )
