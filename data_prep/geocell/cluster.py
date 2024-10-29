import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from distance_utils import _generate_diameter_in_deg
from helper import (
    _check_max_distances,
    _get_points,
    _sample_point,
    _create_point_bbox,
    _create_cluster_bbox,
    _get_bbox_mask,
    _assign_new_cluster,
    _create_directories
)
from gbif_utils import build_geodataframe
import os
import geopandas as gpd
from tqdm import tqdm

# Function to cluster points using DBSCAN
def perform_dbscan(gdf, max_distance, eps):
    """Perform DBSCAN clustering on the given GeoDataFrame."""
    coords = np.array(list(zip(gdf['decimalLongitude'], gdf['decimalLatitude'])))
    db = DBSCAN(eps=eps, min_samples=1).fit(coords)
    return db.labels_


def refine_clusters(input_gdf, max_distance, num_clusters, state):
    """
    Refine clusters that exceed the max distance threshold. Sample randomly within cluster, 
    create bounding boxes, and repeat until no clusters exceed the max distance.
    """

    input_gdf = input_gdf.set_index('occurrenceID')
    
    while True:
        max_distances = _check_max_distances(input_gdf)
        clusters_too_large = max_distances[max_distances['max_distance_meters'] > max_distance]['cluster'].values
        
        if len(clusters_too_large) == 0:
            break  # Exit loop if no clusters exceed the max distance

        gdf_too_large = input_gdf[input_gdf['cluster'].isin(clusters_too_large)].copy()

        for cluster in clusters_too_large:
            points = _get_points(gdf_too_large, cluster)
            sampled_point = _sample_point(points)
            bbox = _create_point_bbox(sampled_point, max_distance)
            mask = _get_bbox_mask(gdf_too_large, bbox)
            gdf_too_large = _assign_new_cluster(gdf_too_large, mask, num_clusters)
            num_clusters += 1

        # Update values only for values not already in a cluster.
        input_gdf.loc[input_gdf.index.isin(gdf_too_large.index), 'cluster'] = gdf_too_large['cluster'].values

    return input_gdf.reset_index()

def renumber_clusters(gdf, state):
    """Renumber clusters and create a state-specific cluster label."""
    gdf['cluster'] = [f"{state}_{cluster}" for cluster in pd.factorize(gdf['cluster'])[0]]
    return gdf

def geocell_clustering(gdf, state, max_distance):
    """Cluster points within a specified state and max distance."""
    
    # Perform initial DBSCAN clustering
    diameter_deg = _generate_diameter_in_deg(max_distance)

    labels = perform_dbscan(gdf, max_distance, eps = diameter_deg)
    gdf['cluster'] = labels
    num_clusters = labels.max()

    # Refine clusters until all meet the max distance threshold
    gdf = refine_clusters(gdf, diameter_deg, num_clusters, state)

    # Renumber clusters and create state-specific labels
    gdf = renumber_clusters(gdf, state)
    
    return gdf

if __name__ == '__main__':
    '''
    Bounding boxes polygon are not necessary outputs so commented out
    TODO: Change to 0.01 deg per latitude
    '''
    
    dimension = 256
    meters_per_pixel = 0.6
    dimension_distance = dimension * meters_per_pixel

    # output_bbox_poly=f'/data/cher/universe7/herbarium/data/geocell/clusters_{dimension}m_{meters_per_pixel}ppixel_polygon.geojson'
    output_bbox_pt=f'/data/cher/universe7/herbarium/data/geocell/clusters_{dimension}m_{meters_per_pixel}ppixel_centroid.csv'
    output_csv=f'/data/cher/universe7/herbarium/data/geocell/clusters_key_{dimension}m.csv'

    _create_directories([output_bbox_pt, 
                        #  output_bbox_poly, 
                         output_csv])
    
    ### Input file downloaded from gbif -- occurrences
    gdf = build_geodataframe(gbif_path = "/data/cher/universe7/herbarium/data/MO-herbarium/occurrence.txt")

    dimension_deg = _generate_diameter_in_deg(dimension_distance)

    # Process each state
    states = gdf['stateProvince'].unique()
    for state in tqdm(states, desc="Processing States"):
        print(state)
        gdf_s = gdf[gdf['stateProvince'] == state].copy()
        gdf_s = geocell_clustering(gdf_s, state, dimension_distance)

        # Save {occurrenceID, cluster} key
        gdf_s[['occurrenceID', 'cluster']].to_csv(output_csv, mode='a', index=False)

        # Create bounding boxes for each cluster
        bbox_gdf = _create_cluster_bbox(gdf_s, dimension_deg)
        bbox_gdf[['cluster', 'lon', 'lat']].to_csv(output_bbox_pt, mode='a', index=False)


        # if not os.path.exists(output_geojson):
        #     bbox_gdf.to_file(output_bbox_poly, driver="GeoJSON")
        # else:
        #     bbox_gdf.to_file(output_bbox_poly, driver="GeoJSON", mode='a')

    print(f"Saved all clusters centroids to {output_bbox_pt}")
    print(f"Saved occurrenceID and cluster mapping to {output_csv}")
