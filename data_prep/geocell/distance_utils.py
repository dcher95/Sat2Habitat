from scipy.spatial.distance import pdist
import numpy as np

# Function to calculate Haversine distance in meters
def haversine(coord1, coord2):
    R = 6371000  # Radius of the Earth in meters
    lat1, lon1 = np.radians(coord1)
    lat2, lon2 = np.radians(coord2)
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c

# Function to find max Haversine distance within each cluster
def max_haversine_distance(group):
    coords = group[['decimalLatitude', 'decimalLongitude']].to_numpy()
    if len(coords) > 1:
        # Calculate pairwise Haversine distances
        distances = pdist(coords, lambda u, v: haversine(u, v))
        max_dist = distances.max()
    else:
        max_dist = 0  # Only one point, no distance to calculate
    return max_dist

def max_distance(group):
    coords = group[['decimalLatitude', 'decimalLongitude']].to_numpy()
    if len(coords) > 1:
        # Calculate pairwise distances
        distances = pdist(coords, metric='euclidean')
        max_dist = distances.max()
    else:
        max_dist = 0  # Only one point, no distance to calculate
    return max_dist

def _generate_diameter_in_deg(dimension_distance):
    return round(dimension_distance / 111320, 6)