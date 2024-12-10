import pandas as pd
import math

def calculate_bounding_box(centroid_lon, centroid_lat, zoom_level, patch_size_pixels=256):
    """
    Calculate the bounding box for a centroid and zoom level.
    
    Args:
        centroid_lon (float): Longitude of the centroid.
        centroid_lat (float): Latitude of the centroid.
        zoom_level (int): The zoom level.
        patch_size_pixels (int): Size of the patch in pixels (default is 256x256).

    Returns:
        tuple: (min_lon, min_lat, max_lon, max_lat)
    """
    # Earth's circumference in meters
    earth_circumference = 40075017
    tile_size = 256  # Tile size in pixels
    origin_shift = earth_circumference / 2.0

    # Calculate resolution (meters per pixel)
    resolution = earth_circumference / (tile_size * (2 ** zoom_level))
    
    # Calculate patch size in meters
    patch_size_meters = patch_size_pixels * resolution
    
    # Convert centroid to Web Mercator meters
    mx = (centroid_lon * origin_shift) / 180.0
    my = math.log(math.tan((90 + centroid_lat) * math.pi / 360.0)) / (math.pi / 180.0)
    my = (my * origin_shift) / 180.0
    
    # Calculate bounding box in Web Mercator meters
    min_mx = mx - patch_size_meters / 2
    max_mx = mx + patch_size_meters / 2
    min_my = my - patch_size_meters / 2
    max_my = my + patch_size_meters / 2
    
    # Convert back to latitude/longitude
    min_lon = (min_mx / origin_shift) * 180.0
    max_lon = (max_mx / origin_shift) * 180.0
    min_lat = 180.0 / math.pi * (2 * math.atan(math.exp((min_my / origin_shift) * math.pi)) - math.pi / 2.0)
    max_lat = 180.0 / math.pi * (2 * math.atan(math.exp((max_my / origin_shift) * math.pi)) - math.pi / 2.0)
    
    return min_lon, min_lat, max_lon, max_lat

def get_patch_for_coordinate(lat, lon, min_lon, min_lat, patch_width_lon, patch_height_lat):
    # Determine the column (longitude-wise)
    col = int((lon - min_lon) // patch_width_lon)
    
    # Determine the row (latitude-wise)
    row = int((lat - min_lat) // patch_height_lat)
    
    return row, col

def assign_patches(data):
    # loop through dataset
    patches_dict = {}
    remaining_data = data.copy()

    while not remaining_data.empty:
        # Randomly select a row (centroid) for bounding box calculation
        row = remaining_data.sample(1)
        centroid_lon, centroid_lat, key = row['lon'].values[0], row['lat'].values[0], row['key'].values[0]

        patches_dict[key] = {}

        # get bounding box of row
        min_lon, min_lat, max_lon, max_lat = calculate_bounding_box(centroid_lon, centroid_lat, zoom_level = 18)

        # find other rows within that bounding box
        filtered_data = remaining_data[
            (remaining_data['lon'] >= min_lon) & (remaining_data['lon'] <= max_lon) &
            (remaining_data['lat'] >= min_lat) & (remaining_data['lat'] <= max_lat)
        ]

        if filtered_data.empty:
            continue

        #### get the image patches for those rows

        # Define the 3x3 grid patch size (in terms of latitude and longitude)
        delta_lon, delta_lat = max_lon - min_lon, max_lat - min_lat
        patch_width_lon, patch_height_lat = delta_lon / 3, delta_lat / 3

        patches = []
        for idx, row in filtered_data.iterrows():
            lat, lon, filtered_key = row['lat'], row['lon'], row['key']  
            patch_row, patch_col = get_patch_for_coordinate(lat, lon, min_lon, min_lat, patch_width_lon, patch_height_lat)
            patch = f'{patch_row}_{patch_col}'
            
            # Store the patch in the dictionary with 'key' as the dictionary key
            patches_dict[key][filtered_key] = patch

        # Remove the assigned rows from remaining_data to avoid reassignment
        remaining_data = remaining_data.drop(filtered_data.index)

    return patches_dict

# Convert the nested dictionary into a list of tuples (key, inner_key, patch)
def convert_nested_dict_to_df(patches_dict):
    df = []
    for key, inner_dict in patches_dict.items():
        for inner_key, patch in inner_dict.items():
            df.append((key, inner_key, patch))

    # Create the DataFrame
    df = pd.DataFrame(df, columns=['assigned_image', 'key', 'patch'])
    return df

if __name__ == "__main__":


    data_path = "/data/cher/Sat2Habitat/data/crisp-data-split/test_w_eco.csv"
    output_path = "/data/cher/Sat2Habitat/data/crisp-data-split/test_w_eco_w_patches.csv"

    data = pd.read_csv(data_path)

    patches_dict = assign_patches(data)
    df = convert_nested_dict_to_df(patches_dict)

    ### Checks ### 
    # Calculate the percentage of data assigned to a patch that is not the middle patch
    patch_not_middle = df[df['patch'] != (1, 1)].shape[0] / df.shape[0] * 100

    # Calculate the percentage of data assigned to a different image (inner_key != key)
    different_image = df[df['key'] != df['assigned_image']].shape[0] / df.shape[0] * 100

    # How many images do we actually need?
    unique_image = df['key'].unique().shape[0]

    print(f"Percentage of data assigned to a patch not the middle patch: {patch_not_middle:.2f}%")
    print(f"Percentage of data assigned to a different image: {different_image:.2f}%")
    print(f"Percentage of images necessary: {(unique_image / df.shape[0]):.2f}%")


    # Save the DataFrame to a CSV file
    data_w_patches = data.merge(df, on = 'key', how = 'left')
    data_w_patches.to_csv(output_path, index = False)