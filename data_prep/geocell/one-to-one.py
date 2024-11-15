import pandas as pd
from sklearn.model_selection import train_test_split
from gbif_utils import build_geodataframe

# data output paths
train_data_path = '/data/cher/Sat2Habitat/data/crisp/train.csv'
val_data_path = '/data/cher/Sat2Habitat/data/crisp/val.csv'
train_data_10_path = '/data/cher/Sat2Habitat/data/crisp/train_10.csv'
val_data_10_path = '/data/cher/Sat2Habitat/data/crisp/val_10.csv'

## Load data
gbif_path = "/data/cher/Sat2Habitat/data/occurrence.txt"
species_wiki_path = "/data/cher/Sat2Habitat/data/species_wiki.csv"

occurrences = build_geodataframe(gbif_path)
species_wiki_df = pd.read_csv(species_wiki_path)

# Data cleaning
occurrences = occurrences.sort_values(by='stateProvince').reset_index(drop=True)
occurrences.index.name = 'key'
occurrences.reset_index(inplace=True)
occurrences.rename(columns={'decimalLatitude': 'lat', 'decimalLongitude': 'lon'}, inplace=True)

## Create the training and validation datasets
species_wiki_df.rename(columns={col: f"{col}_wiki" if col != 'species' else col for col in species_wiki_df.columns}, inplace=True)

habitat_info_w_wiki = occurrences.merge(species_wiki_df, on='species', how='left')
habitat_info_w_wiki = habitat_info_w_wiki[['key', 'species', 'occurrenceID','lat', 'lon','habitat','habitat_wiki', 'distribution and habitat_wiki', 'description_wiki', 'ecology_wiki', 'distribution_wiki', 'header_wiki']]

## Create train and val dataset csvs with text descriptions
train_data, val_data = train_test_split(habitat_info_w_wiki, test_size=0.1, random_state=42)

train_data.to_csv(train_data_path, index=False)
val_data.to_csv(val_data_path, index=False)

train_data.sample(frac=0.1, random_state=42).to_csv(train_data_10_path, index=False)
val_data.sample(frac=0.1, random_state=42).to_csv(val_data_10_path, index=False)