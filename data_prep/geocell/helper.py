from distance_utils import max_haversine_distance, max_distance
from shapely.geometry import box
import random
import pandas as pd
import geopandas as gpd
import os

def _check_max_haversine_distances(gdf):
    """Calculate the max distances for each cluster."""
    max_distances = gdf.reset_index(drop=True).groupby('cluster').apply(max_haversine_distance).reset_index()
    max_distances.columns = ['cluster', 'max_distance_meters']
    return max_distances

def _check_max_distances(gdf):
    """Calculate the max distances for each cluster."""
    max_distances = gdf.reset_index(drop=True).groupby('cluster').apply(max_distance).reset_index()
    max_distances.columns = ['cluster', 'max_distance_meters']
    return max_distances

def _get_points(gdf, cluster):
    """Retrieve all points belonging to a specific cluster."""
    return gdf[gdf['cluster'] == cluster]

def _sample_point(points):
    """Randomly sample 1 point from the cluster to serve as the center for the bounding box."""
    return points.sample(n=1).iloc[0]

def _create_point_bbox(point, max_distance):
    """Create a bounding box around a given point with a defined buffer size."""
    buffer_size = (max_distance / 111320) / 2  # Approx. degrees -- both ways
    lon, lat = point['decimalLongitude'], point['decimalLatitude']
    return box(lon - buffer_size, lat - buffer_size, lon + buffer_size, lat + buffer_size)

def _get_bbox_mask(gdf, bbox):
    """Create a mask to identify which points are within the bounding box."""
    minx, miny, maxx, maxy = bbox.bounds
    return (gdf['decimalLongitude'].between(minx, maxx) &
            gdf['decimalLatitude'].between(miny, maxy))

def _assign_new_cluster(gdf, mask, num_clusters):
    """Assign a new cluster ID to points within the bounding box."""
    gdf.loc[mask, 'cluster'] = num_clusters + 1
    return gdf

def _create_cluster_bbox(gdf_s, radius_diameter):
    """Create bounding box based on mean of clustered points"""
    bounding_boxes = []
    for label in gdf_s['cluster'].unique():
        cluster_points = gdf_s[gdf_s['cluster'] == label]
        minx, miny = cluster_points.geometry.x.min() - (radius_diameter / 2), cluster_points.geometry.y.min() - (radius_diameter / 2)
        maxx, maxy = cluster_points.geometry.x.max() + (radius_diameter /  2), cluster_points.geometry.y.max() + (radius_diameter / 2)
        bbox = box(minx, miny, maxx, maxy)
        centroid = bbox.centroid

        bounding_boxes.append({
            'cluster': label,
            'lat': centroid.y,
            'lon': centroid.x,
            'geometry': bbox,
        })
        
    # Convert bounding boxes to GeoDataFrame and save to GeoJSON iteratively
    return gpd.GeoDataFrame(bounding_boxes)

def _create_directories(files):

    for file in files:
        file_dir = os.path.dirname(file)

        if file_dir and not os.path.exists(file_dir):
            os.makedirs(file_dir)

        if file.lower().endswith(".csv"):
            pd.DataFrame().to_csv(file, index=False)