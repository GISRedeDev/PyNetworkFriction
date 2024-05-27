import warnings

import geopandas as gpd
import pandas as pd


def calculate_straight_line_distances(src_dst_matrix: pd.DataFrame, crs: int) -> gpd.GeoSeries:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        from_ = gpd.GeoSeries(src_dst_matrix.from_centroid, crs=4326).to_crs(crs)
        to_ = gpd.GeoSeries(src_dst_matrix.to_centroid, crs=4326).to_crs(crs)
    return from_.distance(to_)


def calculate_routes_and_route_distances():
    pass


def get_route_geoms():
    # FIXME Should this be removed or only used in levels 1 and 2?
    pass


def calculate_route_to_incident_distances():
    # FIXME this will calculate the distance along network - only use for level 3
    # You also need to check with this keeps the ACLED ID so that the distance can be joined to the
    # distance matrix. Otherwise it's probs useless.
    pass
