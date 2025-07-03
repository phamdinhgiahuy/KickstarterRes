#!/usr/bin/env python3
"""
Quick script to rebuild cache from existing scraped projects.
Run this before starting a new scraping session to continue from where you left off.
"""

import sys
import os
from pathlib import Path

# Add the src directory to path so we can import cache_rebuilder
sys.path.append(str(Path(__file__).parent))

from cache_rebuilder import CacheRebuilder

def quick_rebuild():
    """Quick cache rebuild with default settings."""
    print("🔄 Starting cache rebuild...")
    
    # Default paths (adjust these if your setup is different)
    output_dir = "../data/kickstarter_projects"
    cache_file = "cache.json"
    csv_file = "../data/out/kickstarter_final_data_fixed.csv"
    
    # Check if paths exist
    if not os.path.exists(output_dir):
        print(f"❌ Output directory not found: {output_dir}")
        print("Please check the path and try again.")
        return False
    
    print(f"📁 Scanning output directory: {output_dir}")
    print(f"💾 Cache file: {cache_file}")
    
    # Create and run rebuilder
    rebuilder = CacheRebuilder(
        output_dir=output_dir,
        cache_file=cache_file,
        csv_file=csv_file if os.path.exists(csv_file) else None
    )
    
    success = rebuilder.run()
    
    if success:
        print("\n✅ Cache rebuild completed!")
        print("You can now run your scraper to continue from where you left off.")
    else:
        print("\n❌ Cache rebuild failed. Check the logs for details.")
    
    return success

if __name__ == "__main__":
    quick_rebuild()
