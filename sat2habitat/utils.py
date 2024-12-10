import geopandas as gpd
from shapely.geometry import Point

def get_county_polygon(lat, lon, counties):
    
    # Create a Point object from latitude and longitude
    point = Point(lon, lat)
    
    # Check if the point is inside any county polygon using GeoPandas spatial operation
    # First, ensure the coordinates are in the correct projection (usually WGS84, EPSG:4326)
    counties = counties.to_crs(epsg=4326)  # Assuming the shapefile is in a different CRS
    
    # Use spatial operations to find the county containing the point
    matching_county = counties[counties.contains(point)]
    
    # If a matching county is found, return it
    if not matching_county.empty:
        return matching_county
        # county_polygon = matching_county.iloc[0]['geometry']
        # county_geometry = county_polygon.geometry  # Get the boundary as a LineString
        # return county_geometry
    
    return None