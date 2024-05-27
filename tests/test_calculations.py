import warnings
import geopandas as gpd

from net_friction.calculations import (
    calculate_straight_line_distances
)
from net_friction.data_preparation import make_graph, get_acled_data_from_csv

def test_calculate_straight_line_distances(make_src_dst_matrix):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        make_src_dst_matrix["straight_line_distance"] = calculate_straight_line_distances(make_src_dst_matrix, 6383)
        from_gdf = gpd.GeoDataFrame(make_src_dst_matrix, geometry="from_centroid", crs=4326).to_crs(6383)
        to_gdf = gpd.GeoDataFrame(make_src_dst_matrix, geometry="to_centroid", crs=4326).to_crs(6383)
    from_centroid = from_gdf.loc[0, "from_centroid"]
    to_centroid = to_gdf.loc[0, "to_centroid"]
    assert from_centroid.distance(to_centroid) == make_src_dst_matrix.loc[0, "straight_line_distance"]


def test_calcualte_routes_and_route_distances(make_src_dst_matrix, topology_fixed):
    net, edges = make_graph(topology_fixed, precompute_distance=500)
    acled_data = get_acled_data_from_csv(
        "tests/test_data/ACLED.csv", 6383
    ).set_index("event_id_cnty")
    net.set_pois(
            category="incidents",
            maxdist=5000,
            maxitems=100,
            x_col=acled_data.geometry.x,
            y_col=acled_data.geometry.y
        )
    pois_df = net.nearest_pois(
        distance=5000,
        category="incidents",
        num_pois=3,
        include_poi_ids=True)
    # SAVE NODES AND EDGES
    # SAVE ACLED AND MEASURE THE DISTANCES. TRY TO CHECK THE IDS.
    assert True
