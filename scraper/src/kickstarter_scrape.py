from __future__ import annotations
import pandas as pd
import json
import glob
import os
import re
import time
from bs4 import BeautifulSoup
from tqdm import tqdm
from datetime import datetime
import undetected_chromedriver as uc
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
import urllib.parse
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Actions(BaseModel):
    display_convert_amount: bool = Field(..., alias="displayConvertAmount")


class Location(BaseModel):
    displayable_name: str = Field(..., alias="displayableName")


class LaunchedProjects(BaseModel):
    total_count: int = Field(..., alias="totalCount")


class Website(BaseModel):
    url: str
    domain: str


class Creator(BaseModel):
    id: str
    name: str
    image_url: str = Field(..., alias="imageUrl")
    url: str
    last_login: int = Field(..., alias="lastLogin")
    biography: str
    is_facebook_connected: bool = Field(..., alias="isFacebookConnected")
    allows_follows: bool = Field(..., alias="allowsFollows")
    backings_count: int = Field(..., alias="backingsCount")
    location: Location
    launched_projects: LaunchedProjects = Field(..., alias="launchedProjects")
    websites: List[Website]


class Hls(BaseModel):
    src: str
    type: str


class High(BaseModel):
    src: str
    type: str


class Base(BaseModel):
    src: str
    type: str


class VideoSources(BaseModel):
    hls: Hls
    high: High
    base: Base


class Tracks(BaseModel):
    nodes: List


class Video(BaseModel):
    video_sources: VideoSources = Field(..., alias="videoSources")
    preview_image_url: str = Field(..., alias="previewImageUrl")
    tracks: Tracks


class ParentCategory(BaseModel):
    id: str
    name: str


class Category(BaseModel):
    name: str
    url: str
    parent_category: ParentCategory = Field(..., alias="parentCategory")


class Location1(BaseModel):
    displayable_name: str = Field(..., alias="displayableName")
    discover_url: str = Field(..., alias="discoverUrl")


class Goal(BaseModel):
    currency: str
    symbol: str
    amount: str


class Pledged(BaseModel):
    currency: str
    symbol: str
    amount: str


class Node(BaseModel):
    name: str
    image_url: str = Field(..., alias="imageUrl")
    url: str


class Edge(BaseModel):
    node: Node
    title: str


class Collaborators(BaseModel):
    edges: List[Edge]


class Data(BaseModel):
    id: str


class Node1(BaseModel):
    type: str
    timestamp: int
    data: Data


class Edge1(BaseModel):
    node: Node1


class Timeline(BaseModel):
    edges: List[Edge1]


class Project(BaseModel):
    id: str
    pid: int
    name: str
    image_url: str = Field(..., alias="imageUrl")
    verified_identity: str = Field(..., alias="verifiedIdentity")
    currency: str
    actions: Actions
    creator: Creator
    description: str
    video: Video
    is_project_we_love: bool = Field(..., alias="isProjectWeLove")
    category: Category
    location: Location1
    curated_collection: Dict[str, Any] = Field(..., alias="curatedCollection")
    is_sharing_project_budget: bool = Field(..., alias="isSharingProjectBudget")
    is_forward_fund_tagged: bool = Field(..., alias="isForwardFundTagged")
    backers_count: int = Field(..., alias="backersCount")
    percent_funded: int = Field(..., alias="percentFunded")
    goal: Goal
    pledged: Pledged
    deadline_at: int = Field(..., alias="deadlineAt")
    duration: int
    url: str
    state_changed_at: int = Field(..., alias="stateChangedAt")
    stats_interval: int = Field(..., alias="statsInterval")
    comments_count: int = Field(..., alias="commentsCount")
    state: str
    canceled_at: Any = Field(..., alias="canceledAt")
    slug: str
    is_launched: bool = Field(..., alias="isLaunched")
    is_watchable: bool = Field(..., alias="isWatchable")
    is_watched: bool = Field(..., alias="isWatched")
    user_was_removed: bool = Field(..., alias="userWasRemoved")
    project_short_link: str = Field(..., alias="projectShortLink")
    email_short_link: str = Field(..., alias="emailShortLink")
    collaborators: Collaborators
    timeline: Timeline
    is_in_post_campaign_pledging_phase: bool = Field(
        ..., alias="isInPostCampaignPledgingPhase"
    )


class Model(BaseModel):
    project: Optional[Project] = None


def get_htmls(urls, output_path="output_html"):
    """
    Get HTML content from a list of URLs using Selenium and save them to files.
    """
    # Get matching ChromeDriver
    driver_path = ChromeDriverManager().install()

    # Initialize undetected Chrome with the specific driver
    driver = uc.Chrome(driver_executable_path=driver_path)

    output_dir = "output_html"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        for url in urls:
            base_file_name = url_to_filename(url)
            print(base_file_name)
            driver.get(url)
            print(driver.title)
            html_path = os.path.join(output_path, f"{base_file_name}.html")
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(driver.page_source)

    except Exception as e:
        print(f"Error getting HTML: {e}")
    finally:
        # Using try-except to suppress the shutdown error
        try:
            driver.close()
            driver.quit()
        except Exception as e:
            print(f"Error closing Chrome driver: {e}")


def url_to_filename(url):
    """Convert URL to a valid filename"""
    # Replace special characters with underscores
    parsed = urllib.parse.urlparse(url)
    # Use the path without leading slash and replace / with _
    path = parsed.path.lstrip("/")
    filename = re.sub(r'[\\/:*?"<>|]', "_", path)
    # Add netloc (domain) at the beginning
    filename = f"{parsed.netloc}_{filename}"
    return filename


def extract_tab_information(soup):
    """Extract information from the campaign tabs using emoji selectors"""
    tab_info = {}

    # Extract tabs URLs and metadata
    tabs = [
        ("campaign", "#campaign-emoji"),
        ("rewards", "#rewards-emoji"),
        ("creator", "#creator-emoji"),
        ("faq", "#faq-emoji"),
        ("updates", "#updates-emoji"),
        ("comments", "#comments-emoji"),
        ("community", "#community-emoji"),
    ]

    prefix = "https://www.kickstarter.com"

    for tab_name, selector in tabs:
        tab_elem = soup.select_one(selector)
        if tab_elem:
            # Get tab URL
            if "href" in tab_elem.attrs:
                tab_info[f"{tab_name}_url"] = (
                    prefix + tab_elem["href"]
                    if not tab_elem["href"].startswith("http")
                    else tab_elem["href"]
                )

            # Get special data attributes
            if tab_name == "faq" and "emoji-data" in tab_elem.attrs:
                tab_info["faq_count"] = tab_elem["emoji-data"]

            if tab_name == "updates" and "emoji-data" in tab_elem.attrs:
                tab_info["updates_count"] = tab_elem["emoji-data"]

            if tab_name == "comments" and "data-comments-count" in tab_elem.attrs:
                tab_info["comments_count"] = tab_elem["data-comments-count"]

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
            tab_info[field_name] = section.get_text(separator="\n\n", strip=True)

            # Count words
            word_count = len(tab_info[field_name].split())
            tab_info[f"{field_name}_word_count"] = word_count

            # Extract images if this is the story section
            if field_name == "story_content":
                images = section.find_all("img")
                image_urls = [img.get("src", "") for img in images if img.get("src")]
                tab_info["story_image_urls"] = image_urls
                tab_info["story_image_count"] = len(image_urls)

    return tab_info


def flatten_project_data(model_obj):
    """
    Flatten a Pydantic Model object into a dictionary with flattened keys
    for easier conversion to DataFrame
    """
    if (
        model_obj is None
        or not hasattr(model_obj, "project")
        or model_obj.project is None
    ):
        return {}

    # Start with top-level project attributes
    project = model_obj.project
    result = {
        "project_id": project.id,
        "project_pid": project.pid,
        "project_name": project.name,
        "project_image_url": project.image_url,
        "project_description": project.description,
        "project_url": project.url,
        "project_slug": project.slug,
        "project_state": project.state,
        "project_short_link": project.project_short_link,
        "email_short_link": project.email_short_link,
        # Stats and counts
        "backers_count": project.backers_count,
        "percent_funded": project.percent_funded,
        "comments_count": project.comments_count,
        # Dates and duration
        "deadline_at": project.deadline_at,
        "state_changed_at": project.state_changed_at,
        "duration": project.duration,
        "stats_interval": project.stats_interval,
        # Boolean flags
        "is_project_we_love": project.is_project_we_love,
        "is_sharing_project_budget": project.is_sharing_project_budget,
        "is_forward_fund_tagged": project.is_forward_fund_tagged,
        "is_launched": project.is_launched,
        "is_watchable": project.is_watchable,
        "is_watched": project.is_watched,
        "user_was_removed": project.user_was_removed,
        "is_in_post_campaign_pledging_phase": project.is_in_post_campaign_pledging_phase,
    }

    # Add funding details
    if hasattr(project, "goal") and project.goal:
        result.update(
            {
                "goal_currency": project.goal.currency,
                "goal_symbol": project.goal.symbol,
                "goal_amount": project.goal.amount,
            }
        )

    if hasattr(project, "pledged") and project.pledged:
        result.update(
            {
                "pledged_currency": project.pledged.currency,
                "pledged_symbol": project.pledged.symbol,
                "pledged_amount": project.pledged.amount,
            }
        )

    # Add creator details
    if hasattr(project, "creator") and project.creator:
        creator = project.creator
        result.update(
            {
                "creator_id": creator.id,
                "creator_name": creator.name,
                "creator_image_url": creator.image_url,
                "creator_url": creator.url,
                "creator_biography": creator.biography,
                "creator_last_login": creator.last_login,
                "creator_is_facebook_connected": creator.is_facebook_connected,
                "creator_allows_follows": creator.allows_follows,
                "creator_backings_count": creator.backings_count,
            }
        )

        # Add creator location
        if hasattr(creator, "location") and creator.location:
            result["creator_location"] = creator.location.displayable_name

        # Add creator stats
        if hasattr(creator, "launched_projects") and creator.launched_projects:
            result["creator_launched_projects_count"] = (
                creator.launched_projects.total_count
            )

        # Add creator websites (as a list)
        if hasattr(creator, "websites") and creator.websites:
            websites_list = [website.url for website in creator.websites]
            result["creator_websites"] = websites_list
            # Also include the first domain for easy filtering
            if websites_list:
                result["creator_primary_website_domain"] = creator.websites[0].domain

    # Add category details
    if hasattr(project, "category") and project.category:
        result.update(
            {
                "subcategory_name": project.category.name,
                "subcategory_url": project.category.url,
            }
        )

        if (
            hasattr(project.category, "parent_category")
            and project.category.parent_category
        ):
            result.update(
                {
                    "category_id": project.category.parent_category.id,
                    "category_name": project.category.parent_category.name,
                }
            )

    # Add project location
    if hasattr(project, "location") and project.location:
        result.update(
            {
                "project_location": project.location.displayable_name,
                "project_discover_url": project.location.discover_url,
            }
        )

    # Add video details (if exists)
    if hasattr(project, "video") and project.video:
        result["has_video"] = True
        result["video_preview_url"] = project.video.preview_image_url

        if hasattr(project.video, "video_sources") and project.video.video_sources:
            if (
                hasattr(project.video.video_sources, "base")
                and project.video.video_sources.base
            ):
                result["video_base_url"] = project.video.video_sources.base.src
            if (
                hasattr(project.video.video_sources, "high")
                and project.video.video_sources.high
            ):
                result["video_high_url"] = project.video.video_sources.high.src
            if (
                hasattr(project.video.video_sources, "hls")
                and project.video.video_sources.hls
            ):
                result["video_hls_url"] = project.video.video_sources.hls.src
    else:
        result["has_video"] = False

    # Add collaborator info (count and list)
    if (
        hasattr(project, "collaborators")
        and project.collaborators
        and hasattr(project.collaborators, "edges")
        and project.collaborators.edges
    ):
        collaborators = [edge.node.name for edge in project.collaborators.edges]
        result["collaborators_count"] = len(collaborators)
        result["collaborators"] = collaborators
        result["collaborators_profile_urls"] = [
            edge.node.url for edge in project.collaborators.edges
        ]
    else:
        result["collaborators_count"] = 0

    return result


def convert_dict_to_dataframe(data_dict_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert a list of dictionaries into a pandas DataFrame
    """
    if not data_dict_list or len(data_dict_list) == 0:
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(data_dict_list)

    # Convert timestamps to datetime
    timestamp_cols = [
        col
        for col in df.columns
        if "_at" in col and pd.api.types.is_numeric_dtype(df[col])
    ]
    for col in timestamp_cols:
        try:
            df[col] = pd.to_datetime(df[col], unit="s")
        except:
            pass

    return df


def process_project_data(api_data, html_soup):
    """Process project data from both API and HTML"""
    # Extract tab information from HTML
    tab_data = extract_tab_information(html_soup)

    try:
        # Parse API data with Pydantic model
        model = Model(**api_data)

        # Flatten model data
        flat_data = flatten_project_data(model)

    except (ImportError, Exception) as e:
        print(f"Error parsing with Pydantic model: {e}")
        print("Falling back to direct dictionary processing...")

        # Just use the API data directly as a dictionary
        flat_data = {}

        # Process project data if available
        if isinstance(api_data, dict) and "project" in api_data:
            project = api_data["project"]

            # Extract basic fields
            basic_fields = ["id", "pid", "name", "description", "url", "slug", "state"]
            for field in basic_fields:
                if field in project:
                    flat_data[f"project_{field}"] = project[field]

            # Extract other fields as appropriate
            if "backersCount" in project:
                flat_data["backers_count"] = project["backersCount"]
            if "percentFunded" in project:
                flat_data["percent_funded"] = project["percentFunded"]
            # Add more field extractions as needed...

    # Add HTML-extracted data directly to flattened data
    if tab_data:
        for key, value in tab_data.items():
            if key not in flat_data:
                flat_data[key] = value

    return flat_data


def process_one_campaign(html_path):
    """Process a single campaign HTML file and return a DataFrame with processing timestamp"""
    # Get current timestamp
    processing_time = datetime.now()

    try:
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
    except FileNotFoundError as e:
        print(f"Error opening file: {e}")
        return None
    except Exception as e:
        print(f"Error reading {html_path}: {e}")
        return None

    try:
        # Parse HTML
        soup = BeautifulSoup(html_content, "html.parser")
        div_data = soup.select_one("div[data-initial]")

        if not div_data or "data-initial" not in div_data.attrs:
            print(f"Cannot find data-initial in {html_path}")
            return None

        # Extract and parse JSON data
        api_data = json.loads(div_data["data-initial"])

        # Process project data
        processed_data = process_project_data(api_data, soup)

        # Add source file path for reference
        processed_data["source_file"] = html_path

        # Add processing timestamp
        processed_data["scrape_time"] = processing_time

        # Convert to DataFrame
        df = convert_dict_to_dataframe([processed_data])
        return df

    except json.JSONDecodeError as e:
        print(f"JSON parsing error in {html_path}: {e}")
        return None
    except Exception as e:
        print(f"Error processing {html_path}: {e}")
        return None


def process_multiple_campaigns(html_folder_path, output_csv=None, output_excel=None):
    """
    Process multiple campaign HTML files and combine into a single DataFrame
    with scrape timestamp

    Args:
        html_folder_path: Path to folder containing HTML files
        output_csv: Optional path to save the resulting DataFrame as CSV
        output_excel: Optional path to save the resulting DataFrame as Excel file

    Returns:
        Combined pandas DataFrame with all campaign data
    """
    # Record start time of the entire batch processing
    batch_start_time = datetime.now()

    # Find all HTML files in the folder
    html_files = glob.glob(os.path.join(html_folder_path, "*.html"))

    if not html_files:
        print(f"No HTML files found in {html_folder_path}")
        return pd.DataFrame()

    print(f"Found {len(html_files)} HTML files to process")

    # Process each file and collect the results
    dfs = []
    errors = 0

    # Use tqdm for progress bar
    for html_file in tqdm(html_files, desc="Processing campaigns"):
        df = process_one_campaign(html_file)
        if df is not None and not df.empty:
            dfs.append(df)
        else:
            errors += 1

    # If no valid data was processed
    if not dfs:
        print("No valid data was extracted from any file")
        return pd.DataFrame()

    # Combine all DataFrames
    combined_df = pd.concat(dfs, ignore_index=True)

    # Add batch processing time to metadata
    batch_end_time = datetime.now()
    processing_duration = batch_end_time - batch_start_time

    print(f"Successfully processed {len(dfs)} campaigns with {errors} errors")
    print(f"Combined DataFrame shape: {combined_df.shape}")
    print(f"Total processing time: {processing_duration}")

    # Add batch processing metadata
    # if not combined_df.empty:
    #     combined_df['batch_start_time'] = batch_start_time
    #     combined_df['batch_end_time'] = batch_end_time
    #     combined_df['batch_processing_seconds'] = processing_duration.total_seconds()

    # Save to CSV if output path is provided
    if output_csv:
        # Add timestamp to filename if not already specified
        if "{timestamp}" in output_csv:
            timestamp_str = batch_start_time.strftime("%Y%m%d_%H%M%S")
            output_csv = output_csv.format(timestamp=timestamp_str)

        combined_df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")

    # Save to Excel if output path is provided
    if output_excel:
        # Add timestamp to filename if not already specified
        if "{timestamp}" in output_excel:
            timestamp_str = batch_start_time.strftime("%Y%m%d_%H%M%S")
            output_excel = output_excel.format(timestamp=timestamp_str)

        combined_df.to_excel(output_excel, index=False)
        print(f"Results saved to {output_excel}")

    return combined_df


def main(urls):
    # Define the folder containing HTML files
    html_folder_path = "output_html"
    # create the folder if it doesn't exist
    if not os.path.exists(html_folder_path):
        os.makedirs(html_folder_path)
    # scrape the urls
    get_htmls(urls, html_folder_path)
    # Process the HTML files and save the results

    # Define output paths
    output_csv = "kickstarter_campaigns_{timestamp}.csv"
    output_excel = "kickstarter_campaigns_{timestamp}.xlsx"

    # Process multiple campaigns
    combined_df = process_multiple_campaigns(html_folder_path, output_csv, output_excel)

    # Print the final DataFrame shape
    print(f"Final DataFrame shape: {combined_df.shape}")


# TEST_URLS = [
#     "https://www.kickstarter.com/projects/heroforge/custom-dice?category_id=Q2F0ZWdvcnktMzQ%3D",
#     "https://www.kickstarter.com/projects/frogwares/the-sinking-city-2?category_id=Q2F0ZWdvcnktMzU%3D",
#     "https://www.kickstarter.com/projects/peak-design/roller-pro-carry-on-luggage-by-peak-design?category_id=Q2F0ZWdvcnktMjg%3D",
# ]


if __name__ == "__main__":
    demo_df = pd.read_csv("Kickstarter001.csv")
    urls = map(json.loads, demo_df.urls)
    urls = list(urls)
    project_urls = [url["web"]["project"] for url in urls]
    test_urls = project_urls[:100]
    main(test_urls)
