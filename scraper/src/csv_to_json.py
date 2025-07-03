import pandas as pd
import json
import os

# --- Provided Paths ---
source_csv_path = r"/mnt/c/Users/giahu/OneDrive - Michigan State University/Projects/KickstarterRes/scraper/data/out/kickstarter_final_data_fixed.csv"
out_json_path = r"/mnt/c/Users/giahu/OneDrive - Michigan State University/Projects/KickstarterRes/scraper/data/out/kickstarter_final_data_fixed.json"

# --- Main Logic ---

# It's good practice to use a try-except block to handle potential errors like the file not being found.
try:
    # 1. Read the CSV file into a pandas DataFrame
    # A DataFrame is a 2D table-like data structure.
    print(f"Reading data from '{source_csv_path}'...")
    df = pd.read_csv(source_csv_path)

    # 2. Convert the DataFrame to a JSON file using the .to_json() method
    #    orient='records': This formats the JSON as a list of dictionaries,
    #                      where each dictionary represents a row from the CSV.
    #                      This is a very common and useful JSON structure.
    #    indent=4:         This makes the output JSON file human-readable by
    #                      adding indentation.
    print(f"Converting data and saving to '{out_json_path}'...")
    df.to_json(out_json_path, orient="records", indent=4)

    print("\n✅ Conversion successful!")
    print(f"JSON file saved at: {out_json_path}")

except FileNotFoundError:
    print(f"❌ ERROR: The source file was not found at '{source_csv_path}'")
except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")



