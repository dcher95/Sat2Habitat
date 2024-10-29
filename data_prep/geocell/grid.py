## Inefficient method - e,n,u is faster
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import box
from joblib import Parallel, delayed
from tqdm import tqdm
from gbif_utils import build_geodataframe
from helper import _create_directories
import os

def _build_grid(bbox, cell_size_deg=0.01):
    # Calculate the number of rows and columns based on the degree size of each cell
    lat_range = bbox[3] - bbox[1]  # degrees of latitude
    lon_range = bbox[2] - bbox[0]  # degrees of longitude

    nrows = int(np.ceil(lat_range / cell_size_deg))
    ncols = int(np.ceil(lon_range / cell_size_deg))

    # Set up tasks for parallel processing
    tasks = [(i, j, cell_size_deg, bbox) for i in range(nrows) for j in range(ncols)]
    with Parallel(n_jobs=-1) as parallel:
        grid_polygons = list(parallel(delayed(_create_cell)(*task) for task in tasks))

    return grid_polygons

def _create_cell(i, j, cell_size_deg, bbox):
    # Calculate the corners of the cell
    min_lat = bbox[1] + i * cell_size_deg
    max_lat = bbox[1] + (i + 1) * cell_size_deg
    min_lon = bbox[0] + j * cell_size_deg
    max_lon = bbox[0] + (j + 1) * cell_size_deg
    
    # Return the cell as a tuple of coordinates or as a polygon as needed
    return box(min_lon, min_lat, max_lon, max_lat)

def _create_state_grid(usa_polygon, state, cell_size_deg=0.01):
    # Take bounds -> Make grid -> Overlay with multi-polygon boundary & prune.
    state_polygon = usa_polygon[usa_polygon['name'] == state]

    state_grid = _build_grid(bbox = state_polygon.total_bounds, cell_size_deg=0.01)
    state_grid_gdf = gpd.GeoDataFrame(geometry=state_grid)
    state_grid_gdf.set_crs(usa_polygon.crs, inplace=True)

    print('build state grid')

    grid_within_state = gpd.sjoin(state_grid_gdf, state_polygon, how='inner', predicate='intersects')
    
    # Create unique key for each grid cell
    grid_within_state = grid_within_state.reset_index(drop = True)
    grid_within_state['key'] = grid_within_state['name'].astype(str) + '_' + grid_within_state.index.astype(str)
    grid_within_state = grid_within_state[['key', 'geometry']]

    return grid_within_state[['key', 'geometry']]

def _prune_empty_cells(occurrence_gdf, grid_within_state, state):
    state_occurrence_gdf = occurrence_gdf[occurrence_gdf['stateProvince'] == state].copy()
    state_occurrence_gdf.set_crs(grid_within_state.crs, inplace=True)
    observations_in_cells = gpd.sjoin(state_occurrence_gdf, grid_within_state, how='inner', predicate='within')

    grid_w_observations = grid_within_state[grid_within_state['key'].isin(observations_in_cells['key'])]

    return observations_in_cells, grid_w_observations


if __name__ == '__main__':
    '''
    All bounding boxes within state are not necessary so removed
    '''

    # mimicking nature multi-view
    cell_size_deg = 0.01

    # output_bbox_poly=f'/data/cher/universe7/herbarium/data/geocell/grid_{dimension}m_{meters_per_pixel}ppixel_polygon.geojson'
    output_bbox_pt=f'/data/cher/universe7/herbarium/data/geocell/grid_{cell_size_deg}deg_pt.csv'
    output_csv=f'/data/cher/universe7/herbarium/data/geocell/grid_key_{cell_size_deg}deg.csv'

    _create_directories([output_bbox_pt, 
                        #  output_bbox_poly, 
                         output_csv])

    usa_polygon = gpd.read_file('/data/cher/universe7/herbarium/data/us-state-boundaries.geojson')[['name', 'geometry']]

    occurrence_gdf = build_geodataframe(gbif_path = "/data/cher/universe7/herbarium/data/MO-herbarium/occurrence.txt")
    occurrence_gdf.set_crs(usa_polygon.crs, inplace=True)

    # Just CA for testing
    occurrence_gdf = occurrence_gdf[occurrence_gdf['stateProvince'] == 'Virginia']

    for state in tqdm(occurrence_gdf['stateProvince'].unique(), desc="Processing States"):

        # Create grid for state within state polygon boundaries.
        grid_within_state = _create_state_grid(usa_polygon, state, cell_size_deg)
        print(state, ': created grid')

        # Find occurrences within cells. Prune empty grid cells.
        observations_in_cells, grid_w_observations = _prune_empty_cells(occurrence_gdf, grid_within_state, state)
        grid_w_observations['lon'], grid_w_observations['lat'] = grid_w_observations.centroid.x, grid_w_observations.centroid.y
        print(state, ': pruned grid')

        # Save {occurrenceID, grid key}, polygon, and centroids
        observations_in_cells[['occurrenceID', 'key']].to_csv(output_csv, mode='a', index=False)
        grid_w_observations[['key', 'lon', 'lat']].to_csv(output_bbox_pt, mode='a', index=False)

        # Calculate the number of cells with at least one observation
        num_cells_with_observations = observations_in_cells['index_right'].nunique()
        observations_per_cell = observations_in_cells.groupby('index_right').size().reset_index(name='observations_count').sort_values('observations_count', ascending = False)

        print(f"Occurrences: {observations_in_cells.shape[0]}\nSatellite images: {num_cells_with_observations}\nDescriptions per image {np.mean(observations_per_cell.observations_count)}")

    # print(f"Saved all grid polygons to {output_bbox_poly}")
    # TODO: Remove empty first line if exists
    print(f"Saved all grid centroids to {output_bbox_pt}")
    print(f"Saved occurrenceID and grid mapping to {output_csv}")