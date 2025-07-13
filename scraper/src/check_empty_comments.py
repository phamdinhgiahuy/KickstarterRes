"""Check for empty comments JSON API responses in kickstarter data. Delete the project folder if the comments are empty."""

import json
import os
import logging
from pathlib import Path
from typing import Set, Optional
from tqdm import tqdm
import shutil
import concurrent.futures


OUT_DIR = Path("../data/kickstarter_projects")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def check_project_for_empty_comments(project_dir: Path) -> Optional[str]:
    """
    Check a single project directory for empty comments.
    Returns the project ID if comments are empty, otherwise None.
    """
    if not project_dir.is_dir():
        return None

    comments_file = project_dir / "comments.json"
    if not comments_file.exists():
        logging.debug(f"No comments file for project: {project_dir.name}")
        return project_dir.name

    try:
        with open(comments_file, "r", encoding="utf-8") as f:
            # Check if file is empty before trying to load JSON
            if os.fstat(f.fileno()).st_size == 0:
                logging.info(f"Empty comments file for project: {project_dir.name}")
                return project_dir.name
            comments_data = json.load(f)
            # An empty list `[]` from the API means no comments.
            if not comments_data:
                logging.info(f"Empty comments data for project: {project_dir.name}")
                return project_dir.name
    except (json.JSONDecodeError, IOError) as e:
        logging.warning(
            f"Could not read/parse {comments_file}: {e}. Marking for deletion."
        )
        return project_dir.name

    return None


def get_empty_comments_projects(out_dir: Path, max_workers: int = None) -> Set[str]:
    """Get a set of project IDs with empty comments, using parallel processing."""
    empty_projects = set()
    project_dirs = list(out_dir.iterdir())

    # Use ThreadPoolExecutor for I/O-bound tasks like reading files.
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Use tqdm to show progress
        results = list(
            tqdm(
                executor.map(check_project_for_empty_comments, project_dirs),
                total=len(project_dirs),
                desc="Checking projects for empty comments",
            )
        )

    for project_id in results:
        if project_id:
            empty_projects.add(project_id)

    return empty_projects


def delete_empty_projects(out_dir: Path, empty_projects: Set[str]) -> None:
    """Delete project directories with empty comments."""
    if not empty_projects:
        logging.info("No projects with empty comments to delete.")
        return

    for project_id in tqdm(empty_projects, desc="Deleting projects"):
        project_dir = out_dir / project_id
        if project_dir.is_dir():
            try:
                shutil.rmtree(project_dir)
                logging.info(f"Deleted project directory: {project_dir}")
            except OSError as e:
                logging.error(f"Error deleting directory {project_dir}: {e}")


def main():
    """Main execution function."""
    logging.info(f"Scanning directory: {OUT_DIR}")
    empty_projects = get_empty_comments_projects(OUT_DIR)

    if not empty_projects:
        logging.info("No projects with empty comments found.")
        return

    logging.info(f"Found {len(empty_projects)} projects with empty comments.")

    # As a safeguard, you might want to ask for confirmation before deleting.
    # confirm = input(f"Proceed with deleting {len(empty_projects)} directories? (y/N): ")
    # if confirm.lower() != 'y':
    #     logging.info("Deletion aborted by user.")
    #     return

    delete_empty_projects(OUT_DIR, empty_projects)
    logging.info("Cleanup complete.")


if __name__ == "__main__":
    main()
