import requests
import os
from mediawikiapi import MediaWikiAPI, PageError

import pandas as pd
from io import BytesIO

from tqdm import tqdm
import re

from typing import List, Dict, Set, Optional

def extract_unique_species_to_csv(
    base_url: str, 
    species_csv_output_path: str, 
    log_file_path: str
) -> None:
    """
    Extract unique species names from a series of Parquet files hosted at the given URL.
    Append species names to a CSV file as the files are processed and log the file URLs 
    that have been successfully downloaded and extracted.

    Not the most efficient sol'n -- but whatever.

    Args:
        base_url (str): Parquet files URLs
        species_csv_output_path (str): The path to the CSV file where unique species names will be saved.
        log_file_path (str): The path to the file where downloaded and processed file URLs will be logged.

    Returns:
        None: The function writes species names and logs to separate files.
    """
    data_url_ls = list(requests.get(base_url).json())
    unique_species: Set[str] = set()

    with open(species_csv_output_path, 'a') as species_file, open(log_file_path, 'a') as log_file:
        for data_url in data_url_ls:

            response = requests.get(data_url, stream=True)

            if response.status_code == 200:
                parquet_data = BytesIO(response.content)
                df = pd.read_parquet(parquet_data)
                species_from_file = set(df['name'].unique())
                for species in species_from_file.difference(unique_species):
                    species_file.write(f"{species}\n")

                unique_species.update(species_from_file)
                log_file.write(f"{data_url}\n")
                print(f"Downloaded and extracted: {data_url}.")
            else:
                print(f"Failed to download {data_url}. Status code: {response.status_code}")

    species_outputs = pd.read_csv(species_csv_output_path, header = None)
    species_outputs = species_outputs.drop_duplicates()
    species_outputs.to_csv(species_csv_output_path, index=False, header=False)

    print(f"Unique species written to {species_csv_output_path}")
    print(f"Processed files logged in {log_file_path}")

def load_processed_data(output_file_path: str, sections: List[str]) -> (pd.DataFrame, Set[str]):
    """
    Load already processed species data from the output file, if it exists.

    Args:
        output_file_path (str): Path to the output CSV file.
        sections (List[str]): List of sections to include as columns.

    Returns:
        (pd.DataFrame, Set[str]): DataFrame with processed data and a set of processed species.
    """
    if os.path.exists(output_file_path):
        processed_df = pd.read_csv(output_file_path, index_col=0)
        processed_species = set(processed_df.index)
    else:
        processed_df = pd.DataFrame(columns=['species'] + sections + ['header'])
        processed_species = set()

    return processed_df, processed_species

def process_species(species_name: str, sections: List[str]) -> Optional[Dict[str, str]]:
    """
    Process a species by fetching content from Wikipedia.

    Args:
        species_name (str): The name of the species to process.
        sections (List[str]): List of Wikipedia sections to fetch.

    Returns:
        Optional[Dict[str, str]]: Dictionary containing species information or None if an error occurs.
    """
    try:
        # Assuming ws.get_wikipedia_content is a function that retrieves Wikipedia content
        wiki_dict = get_wikipedia_content(species_name=species_name, sections=sections)
        return wiki_dict
    except PageError:
        return None
    except Exception:
        return None

def save_species_data(wiki_dict: Dict[str, str], species_name: str, output_file_path: str, all_columns: List[str]) -> None:
    """
    Save the processed species data to the output file.

    Args:
        wiki_dict (Dict[str, str]): Dictionary containing species data.
        species_name (str): The name of the species.
        output_file_path (str): Path to the output CSV file.
        all_columns (List[str]): List of all columns to ensure consistent data format.
    """
    wiki_dict['species'] = species_name
    species_df_temp = pd.DataFrame([wiki_dict])
    species_df_temp = species_df_temp.reindex(columns=all_columns)
    species_df_temp.to_csv(output_file_path, mode='a', header=not os.path.exists(output_file_path), index=False)

def get_wikipedia_content(species_name, sections):

    mediawikiapi = MediaWikiAPI()
    page = mediawikiapi.page(species_name.replace(' ', '_'))
    full_text = page.content
    sections_available = page.sections

    section_dict = {}
    for section in sections:
        section_match = re.search(rf'== {section} ==(.*?)==', full_text, re.IGNORECASE | re.DOTALL)

        # Check if the section exists and save its content
        if section_match:
            section_dict[section] = section_match.group(1).strip()  

    # Check if section_dict is still empty
    if not section_dict:
        section = 'header'
        before_header_match = re.search(r'^(.*?)\n==', full_text, re.DOTALL)
        
        # Check if a match is found and save the content
        if before_header_match:
            section_dict[section] = before_header_match.group(1).strip()  # Remove leading/trailing whitespace
    
    section_dict['sections_available'] = sections_available

    return section_dict

def build_wiki_data(file_path: str, output_file_path: str, sections: List[str]) -> None:
    """
    Load, process, and save species data by fetching content from Wikipedia.

    Args:
        file_path (str): Path to the input CSV file with species names.
        output_file_path (str): Path to the output CSV file.
        sections (List[str]): List of Wikipedia sections to retrieve for each species.
    """
    # Load species data
    species_df = pd.read_csv(file_path, header=None)

    # Load processed data
    processed_df, processed_species = load_processed_data(output_file_path, sections)
    total_species = len(species_df)
    
    # Standard column names
    all_columns = ['species'] + sections + ['header']

    # Process each species
    for i in tqdm(range(total_species)):
        species_name = species_df.loc[i].values[0]

        # Skip if species is already processed
        if species_name in processed_species:
            continue

        # Process the species and save if successful
        wiki_dict = process_species(species_name, sections)
        if wiki_dict:
            save_species_data(wiki_dict, species_name, output_file_path, all_columns)
            processed_species.add(species_name)

    print("Processing complete!")
