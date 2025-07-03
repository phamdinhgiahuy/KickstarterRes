import pandas as pd
import json
import os
import re
import glob

df = pd.read_csv(r'parallel_results\updated_stories_parallel_20250602_033500.csv')

# df_merged = df[df['state'].isin(['successful', 'failed'])]
df_merged = df
notnull_story_df = df_merged[~(df_merged['story_content'].isnull() | (df_merged['story_content'] == ''))]
notnull_story_df

# save the dataframe to a CSV file
notnull_story_df.to_csv(r'parallel_results\updated_stories_parallel_20250602_033500_notnull.csv', index=False)


def clean_illegal_characters(text):
    """Remove illegal characters that Excel cannot handle"""
    if pd.isna(text) or not isinstance(text, str):
        return text
    
    # Remove control characters (0-31 except tab, newline, carriage return)
    # and other problematic characters
    illegal_chars = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]')
    return illegal_chars.sub('', text)

def save_to_excel(df, filename):
    # Clean the dataframe before saving
    df_clean = df.copy()
    
    # Apply cleaning to all string columns
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].apply(clean_illegal_characters)
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        df_clean.to_excel(writer, index=False)

def save_to_excel_with_chunks(df, base_filename):
    # Create output directory if it doesn't exist
    output_dir = 'parallel_results'
    os.makedirs(output_dir, exist_ok=True)

    # Split the dataframe into chunks of 20000 rows
    for i in range(0, df.shape[0], 20000):
        chunk = df.iloc[i:i + 20000]
        chunk_filename = os.path.join(output_dir, f"{base_filename}_part_{i // 20000 + 1}.xlsx")
        save_to_excel(chunk, chunk_filename)

# Save the notnull_story_df to Excel with chunks
save_to_excel_with_chunks(notnull_story_df, 'updated_stories_parallel_20250602_033500_notnull')
