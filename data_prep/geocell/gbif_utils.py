import pandas as pd
from shapely.geometry import Point
import geopandas as gpd

def _clean_data(data):
    
    data = data[~data['stateProvince'].isna()]

    data['stateProvince'] = data['stateProvince'].replace({'New jersey' : 'New Jersey', 
                                                           'oklahoma' : 'Oklahoma', 
                                                           'wyoming' : 'Wyoming'})

    data = data[
        pd.to_numeric(data['decimalLatitude'], errors='coerce').notnull() &
        pd.to_numeric(data['decimalLongitude'], errors='coerce').notnull()
    ]
    data['decimalLatitude'] = data['decimalLatitude'].astype(float)
    data['decimalLongitude'] = data['decimalLongitude'].astype(float)

    data['geometry'] = data.apply(
        lambda row: Point(row['decimalLongitude'], row['decimalLatitude']), axis=1
    )
    return data

def build_geodataframe(gbif_path = "/data/cher/universe7/herbarium/data/MO-herbarium/occurrence.txt"):
    habitat_info = pd.read_csv(gbif_path, sep="\t", on_bad_lines='skip')
    habitat_info = habitat_info[['occurrenceID', 'habitat' , 'stateProvince', 'decimalLatitude', 'decimalLongitude']]

    data = habitat_info[~habitat_info['habitat'].isna()].copy()

    # Convert to GeoDataFrame
    data = _clean_data(data[['occurrenceID','stateProvince','decimalLatitude', 'decimalLongitude']].copy())
    data['geometry'] = data.apply(lambda row: Point(row['decimalLongitude'], row['decimalLatitude']), axis=1)
    gdf = gpd.GeoDataFrame(data, geometry='geometry')

    return gdf