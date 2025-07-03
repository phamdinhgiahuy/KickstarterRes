#!/usr/bin/env python3
"""
Cache verification and status checker for Kickstarter scraper.
Shows current cache status and progress statistics.
"""

import json
import os
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


def load_cache(cache_file: str = "cache.json") -> Optional[Dict]:
    """Load cache file."""
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading cache: {e}")
    return None


def load_csv_data(csv_file: str) -> Optional[pd.DataFrame]:
    """Load CSV file with project data."""
    if os.path.exists(csv_file):
        try:
            return pd.read_csv(csv_file)
        except Exception as e:
            print(f"Error loading CSV: {e}")
    return None


def count_scraped_projects(output_dir: str) -> int:
    """Count number of scraped project folders."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return 0
    
    count = 0
    for item in output_path.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            count += 1
    
    return count


def analyze_cache_status(output_dir: str = "../data/kickstarter_projects",
                        cache_file: str = "cache.json",
                        csv_file: str = "../data/out/kickstarter_final_data_fixed.csv"):
    """Analyze and display cache status."""
    
    print("="*60)
    print("KICKSTARTER SCRAPER - CACHE STATUS")
    print("="*60)
    
    # Load cache
    cache_data = load_cache(cache_file)
    if cache_data:
        print(f"📄 Cache file: {cache_file} ✅")
        print(f"   - URLs processed: {len(cache_data.get('urls', []))}")
        print(f"   - Project IDs: {len(cache_data.get('project_ids', []))}")
        print(f"   - Failed URLs: {len(cache_data.get('failed_urls', []))}")
        
        if 'rebuilt_at' in cache_data:
            print(f"   - Last rebuilt: {cache_data['rebuilt_at']}")
        
        if 'rebuild_stats' in cache_data:
            stats = cache_data['rebuild_stats']
            print(f"   - Scraped folders: {stats.get('scraped_folders', 'N/A')}")
    else:
        print(f"📄 Cache file: {cache_file} ❌ (not found)")
    
    print()
    
    # Count scraped projects
    scraped_count = count_scraped_projects(output_dir)
    print(f"📁 Output directory: {output_dir}")
    print(f"   - Scraped project folders: {scraped_count}")
    
    print()
    
    # Load CSV data
    df = load_csv_data(csv_file)
    if df is not None:
        print(f"📊 CSV file: {csv_file} ✅")
        total_projects = len(df)
        print(f"   - Total projects in CSV: {total_projects}")
        
        # Calculate progress
        if cache_data and scraped_count > 0:
            processed_urls = len(cache_data.get('urls', []))
            progress_percent = (processed_urls / total_projects) * 100 if total_projects > 0 else 0
            
            print(f"\n📈 PROGRESS STATISTICS:")
            print(f"   - Total projects: {total_projects}")
            print(f"   - Processed: {processed_urls}")
            print(f"   - Remaining: {total_projects - processed_urls}")
            print(f"   - Progress: {progress_percent:.1f}%")
            
            # Progress bar
            bar_length = 50
            filled_length = int(bar_length * progress_percent / 100)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            print(f"   - [{bar}] {progress_percent:.1f}%")
    else:
        print(f"📊 CSV file: {csv_file} ❌ (not found)")
    
    print()
    
    # Recommendations
    print("💡 RECOMMENDATIONS:")
    
    if not cache_data:
        print("   - Run 'python rebuild_cache.py' to create initial cache")
    elif scraped_count > len(cache_data.get('urls', [])):
        print("   - Run 'python rebuild_cache.py' to update cache with recent projects")
    elif cache_data and df is not None:
        remaining = len(df) - len(cache_data.get('urls', []))
        if remaining > 0:
            print(f"   - Ready to continue scraping {remaining} remaining projects")
        else:
            print("   - All projects appear to be processed! ✅")
    
    print("\n" + "="*60)


def show_recent_projects(output_dir: str = "../data/kickstarter_projects", 
                        limit: int = 10):
    """Show most recently scraped projects."""
    output_path = Path(output_dir)
    if not output_path.exists():
        print(f"Output directory {output_dir} not found")
        return
    
    print(f"\n📋 RECENT PROJECTS (last {limit}):")
    print("-" * 40)
    
    # Get all project directories with their modification times
    projects = []
    for item in output_path.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            try:
                mtime = item.stat().st_mtime
                projects.append((item.name, datetime.fromtimestamp(mtime)))
            except:
                pass
    
    # Sort by modification time (newest first)
    projects.sort(key=lambda x: x[1], reverse=True)
    
    # Show recent projects
    for i, (project_name, mod_time) in enumerate(projects[:limit]):
        print(f"{i+1:2d}. {project_name} ({mod_time.strftime('%Y-%m-%d %H:%M')})")
    
    if not projects:
        print("   No projects found")


if __name__ == "__main__":
    # Default paths - adjust if needed
    OUTPUT_DIR = "../data/kickstarter_projects"
    CACHE_FILE = "cache.json"
    CSV_FILE = "../data/out/kickstarter_final_data_fixed.csv"
    
    # Show cache status
    analyze_cache_status(OUTPUT_DIR, CACHE_FILE, CSV_FILE)
    
    # Show recent projects
    show_recent_projects(OUTPUT_DIR)
