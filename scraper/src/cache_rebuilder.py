"""
Cache Rebuilder for Kickstarter Scraper

This script scans the output directory for already scraped projects and rebuilds
the cache file to continue scraping from where it left off.
"""

import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple
from datetime import datetime
import pandas as pd


class CacheRebuilder:
    """Rebuild cache from existing scraped project folders."""
    
    def __init__(self, output_dir: str = "../data/kickstarter_projects", 
                 cache_file: str = "cache.json", 
                 csv_file: str = None):
        """
        Initialize CacheRebuilder.
        
        Args:
            output_dir: Directory containing scraped project folders
            cache_file: Path to cache file to create/update
            csv_file: Optional CSV file with project URLs to cross-reference
        """
        self.output_dir = Path(output_dir)
        self.cache_file = cache_file
        self.csv_file = csv_file
        self.setup_logging()
    
    def setup_logging(self) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'cache_rebuild_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def scan_scraped_projects(self) -> Tuple[Set[str], Set[str], Dict[str, str]]:
        """
        Scan output directory for scraped projects.
        
        Returns:
            Tuple of (project_ids, urls, project_id_to_url_mapping)
        """
        project_ids = set()
        urls = set()
        project_mapping = {}
        
        if not self.output_dir.exists():
            self.logger.warning(f"Output directory {self.output_dir} does not exist")
            return project_ids, urls, project_mapping
        
        self.logger.info(f"Scanning {self.output_dir} for scraped projects...")
        
        # Get all subdirectories in output folder
        for project_dir in self.output_dir.iterdir():
            if not project_dir.is_dir():
                continue
            
            project_folder_name = project_dir.name
            metadata_file = project_dir / "metadata.json"
            
            # Try to get project info from metadata.json
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    project_id = metadata.get('project_id')
                    url = metadata.get('url')
                    
                    if project_id:
                        project_ids.add(str(project_id))
                    
                    if url:
                        urls.add(url)
                        if project_id:
                            project_mapping[str(project_id)] = url
                    
                    self.logger.debug(f"Found project: {project_id} -> {url}")
                    
                except Exception as e:
                    self.logger.warning(f"Error reading metadata from {metadata_file}: {e}")
                    # Fallback: use folder name as project ID if it's numeric
                    if project_folder_name.isdigit():
                        project_ids.add(project_folder_name)
            
            else:
                # No metadata file - try to infer from folder name
                if project_folder_name.isdigit():
                    project_ids.add(project_folder_name)
                    self.logger.debug(f"Inferred project ID from folder name: {project_folder_name}")
        
        self.logger.info(f"Found {len(project_ids)} project IDs and {len(urls)} URLs")
        return project_ids, urls, project_mapping
    
    def load_existing_cache(self) -> Dict[str, List[str]]:
        """Load existing cache file if it exists."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                
                self.logger.info(f"Loaded existing cache with {len(cache_data.get('urls', []))} URLs")
                return {
                    'urls': cache_data.get('urls', []),
                    'project_ids': cache_data.get('project_ids', []),
                    'failed_urls': cache_data.get('failed_urls', [])
                }
            except Exception as e:
                self.logger.warning(f"Error loading existing cache: {e}")
        
        return {'urls': [], 'project_ids': [], 'failed_urls': []}
    
    def cross_reference_with_csv(self, project_ids: Set[str]) -> Set[str]:
        """
        Cross-reference project IDs with CSV file to get corresponding URLs.
        
        Args:
            project_ids: Set of project IDs found in scraped folders
            
        Returns:
            Set of URLs corresponding to the project IDs
        """
        urls = set()
        
        if not self.csv_file or not os.path.exists(self.csv_file):
            self.logger.info("No CSV file provided or file doesn't exist")
            return urls
        
        try:
            self.logger.info(f"Cross-referencing with CSV file: {self.csv_file}")
            df = pd.read_csv(self.csv_file)
            
            # Look for project ID and URL columns
            id_columns = [col for col in df.columns if 'id' in col.lower() and 'project' in col.lower()]
            url_columns = [col for col in df.columns if 'url' in col.lower()]
            
            if not id_columns or not url_columns:
                self.logger.warning("Could not find project ID or URL columns in CSV")
                return urls
            
            id_col = id_columns[0]
            url_col = url_columns[0]
            
            self.logger.info(f"Using columns: {id_col} and {url_col}")
            
            # Convert project IDs to strings for comparison
            df[id_col] = df[id_col].astype(str)
            
            # Filter DataFrame for matching project IDs
            matched_df = df[df[id_col].isin(project_ids)]
            matched_urls = set(matched_df[url_col].dropna().tolist())
            
            self.logger.info(f"Matched {len(matched_urls)} URLs from CSV")
            return matched_urls
            
        except Exception as e:
            self.logger.error(f"Error processing CSV file: {e}")
            return urls
    
    def rebuild_cache(self) -> Dict[str, List[str]]:
        """
        Rebuild cache from scraped projects.
        
        Returns:
            Updated cache data
        """
        self.logger.info("Starting cache rebuild...")
        
        # Scan for scraped projects
        scraped_project_ids, scraped_urls, project_mapping = self.scan_scraped_projects()
        
        # Cross-reference with CSV if provided
        csv_urls = self.cross_reference_with_csv(scraped_project_ids)
        
        # Combine URLs from different sources
        all_urls = scraped_urls.union(csv_urls)
        
        # Load existing cache
        existing_cache = self.load_existing_cache()
        
        # Merge with existing cache data
        final_urls = list(set(existing_cache['urls'] + list(all_urls)))
        final_project_ids = list(set(existing_cache['project_ids'] + list(scraped_project_ids)))
        
        # Keep existing failed URLs
        failed_urls = existing_cache['failed_urls']
        
        cache_data = {
            'urls': final_urls,
            'project_ids': final_project_ids,
            'failed_urls': failed_urls,
            'rebuilt_at': datetime.now().isoformat(),
            'rebuild_stats': {
                'scraped_folders': len(scraped_project_ids),
                'scraped_urls': len(scraped_urls),
                'csv_urls': len(csv_urls),
                'total_urls': len(final_urls),
                'total_project_ids': len(final_project_ids)
            }
        }
        
        return cache_data
    
    def save_cache(self, cache_data: Dict) -> bool:
        """Save cache data to file."""
        try:
            # Create backup of existing cache
            if os.path.exists(self.cache_file):
                backup_file = f"{self.cache_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                os.rename(self.cache_file, backup_file)
                self.logger.info(f"Created backup: {backup_file}")
            
            # Save new cache
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Cache saved to {self.cache_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving cache: {e}")
            return False
    
    def generate_report(self, cache_data: Dict) -> str:
        """Generate a summary report."""
        stats = cache_data.get('rebuild_stats', {})
        
        report = f"""
=== CACHE REBUILD REPORT ===
Rebuild Time: {cache_data.get('rebuilt_at', 'Unknown')}
Output Directory: {self.output_dir}
Cache File: {self.cache_file}

STATISTICS:
- Scraped Project Folders: {stats.get('scraped_folders', 0)}
- URLs from Scraped Folders: {stats.get('scraped_urls', 0)}
- URLs from CSV Cross-reference: {stats.get('csv_urls', 0)}
- Total URLs in Cache: {stats.get('total_urls', 0)}
- Total Project IDs in Cache: {stats.get('total_project_ids', 0)}
- Failed URLs (preserved): {len(cache_data.get('failed_urls', []))}

CACHE CONTENTS:
- Successfully Processed URLs: {len(cache_data.get('urls', []))}
- Known Project IDs: {len(cache_data.get('project_ids', []))}
============================
        """
        
        return report.strip()
    
    def run(self) -> bool:
        """
        Run the complete cache rebuild process.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Rebuild cache
            cache_data = self.rebuild_cache()
            
            # Save cache
            if self.save_cache(cache_data):
                # Generate and display report
                report = self.generate_report(cache_data)
                print(report)
                self.logger.info("Cache rebuild completed successfully")
                return True
            else:
                self.logger.error("Failed to save cache")
                return False
                
        except Exception as e:
            self.logger.error(f"Cache rebuild failed: {e}")
            return False


def main():
    """Main function to run cache rebuilder."""
    
    # Configuration
    output_dir = "../data/kickstarter_projects"
    cache_file = "cache.json"
    csv_file = "../data/out/kickstarter_final_data_fixed.csv"  # Optional
    
    # Create rebuilder
    rebuilder = CacheRebuilder(
        output_dir=output_dir,
        cache_file=cache_file,
        csv_file=csv_file
    )
    
    # Run rebuild
    success = rebuilder.run()
    
    if success:
        print("\n✅ Cache rebuild completed successfully!")
        print(f"📁 Output directory: {output_dir}")
        print(f"💾 Cache file: {cache_file}")
        print("\nYou can now continue scraping with the updated cache.")
    else:
        print("\n❌ Cache rebuild failed. Check the logs for details.")


if __name__ == "__main__":
    main()
