import os
import json
from datetime import datetime
from typing import Dict, List, Set, Any, Optional
import logging

# Configure logging
logger = logging.getLogger(__name__)

class URLCache:
    def __init__(self, cache_file_path: str = "data/url_cache.json"):
        """Initialize the URL cache manager.
        
        Args:
            cache_file_path: Path to the cache file
        """
        self.cache_file_path = cache_file_path
        self.processed_urls: Set[str] = set()
        self.completed_ids: Set[str] = set()
        self.load_cache()
    
    def load_cache(self) -> None:
        """Load the cache from disk if it exists."""
        if os.path.exists(self.cache_file_path):
            try:
                with open(self.cache_file_path, 'r', encoding='utf-8') as file:
                    cache_data = json.load(file)
                    self.processed_urls = set(cache_data.get("processed_urls", []))
                    self.completed_ids = set(cache_data.get("completed_ids", []))
                logger.info(f"Loaded cache with {len(self.processed_urls)} processed URLs and {len(self.completed_ids)} completed IDs")
            except Exception as e:
                logger.error(f"Error loading cache file: {e}")
                # Initialize empty cache if file exists but can't be loaded
                self.processed_urls = set()
                self.completed_ids = set()
        else:
            logger.info("No cache file found, starting with empty cache")
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.cache_file_path), exist_ok=True)
    
    def save_cache(self) -> None:
        """Save the current cache to disk."""
        cache_data = {
            "processed_urls": list(self.processed_urls),
            "completed_ids": list(self.completed_ids),
            "last_updated": datetime.now().isoformat()
        }
        try:
            with open(self.cache_file_path, 'w', encoding='utf-8') as file:
                json.dump(cache_data, file, indent=2)
            logger.info(f"Cache saved with {len(self.processed_urls)} processed URLs and {len(self.completed_ids)} completed IDs")
        except Exception as e:
            logger.error(f"Error saving cache file: {e}")
    
    def is_url_processed(self, url: str) -> bool:
        """Check if a URL has already been processed.
        
        Args:
            url: The URL to check
            
        Returns:
            bool: True if the URL has been processed, False otherwise
        """
        return url in self.processed_urls
    
    def is_id_completed(self, id_str: str) -> bool:
        """Check if an ID has been completely processed.
        
        Args:
            id_str: The ID to check
            
        Returns:
            bool: True if the ID has been processed, False otherwise
        """
        return id_str in self.completed_ids
    
    def mark_url_processed(self, url: str) -> None:
        """Mark a URL as processed.
        
        Args:
            url: The URL to mark as processed
        """
        self.processed_urls.add(url)
    
    def mark_id_completed(self, id_str: str) -> None:
        """Mark an ID as completely processed.
        
        Args:
            id_str: The ID to mark as completed
        """
        self.completed_ids.add(id_str)
    
    def get_unprocessed_urls(self, id_str: str, urls: List[str]) -> List[str]:
        """Get the list of URLs for the given ID that haven't been processed yet.
        
        Args:
            id_str: The ID to check
            urls: The list of all URLs for this ID
            
        Returns:
            List[str]: The list of unprocessed URLs
        """
        if self.is_id_completed(id_str):
            return []
        
        return [url for url in urls if not self.is_url_processed(url)]
