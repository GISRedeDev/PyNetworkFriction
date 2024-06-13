import warnings

import geopandas as gpd
import pandas as pd

from net_friction.calculations import (
    calculate_routes_and_route_distances,
    calculate_straight_line_distances,
    get_distances_to_route_experimental,
    get_incidents_in_route,
    get_incidents_in_route_sjoin,
    get_pois_with_nodes,
    get_route_geoms_ids,
)
from net_friction.data_preparation import get_acled_data_from_csv, make_graph


def test_calculate_straight_line_distances(make_src_dst_matrix):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        make_src_dst_matrix["straight_line_distance"] = (
            calculate_straight_line_distances(make_src_dst_matrix, 6383)
        )
        from_gdf = gpd.GeoDataFrame(
            make_src_dst_matrix, geometry="from_centroid", crs=6383
        )
        to_gdf = gpd.GeoDataFrame(make_src_dst_matrix, geometry="to_centroid", crs=6383)
    from_centroid = from_gdf.loc[0, "from_centroid"]
    to_centroid = to_gdf.loc[0, "to_centroid"]
    assert (
        from_centroid.distance(to_centroid)
        == make_src_dst_matrix.loc[0, "straight_line_distance"]
    )


def test_calculate_routes_and_route_distances(make_src_dst_matrix, topology_fixed):
    net, edges = make_graph(topology_fixed, precompute_distance=500)
    shortest_path_nodes, shortest_path_lengths = calculate_routes_and_route_distances(
        net, make_src_dst_matrix
    )
    make_src_dst_matrix["shortest_path_nodes"] = shortest_path_nodes
    make_src_dst_matrix["shortest_path_lengths"] = shortest_path_lengths
    from_node = make_src_dst_matrix.loc[0, "from_nodeID"]
    to_node = make_src_dst_matrix.loc[0, "to_nodeID"]
    assert from_node == shortest_path_nodes[0][0]
    assert to_node == shortest_path_nodes[0][-1]


def test_get_route_geoms_ids(routes_with_distances, topology_fixed):
    _, edges = make_graph(topology_fixed, precompute_distance=500)
    route_df = routes_with_distances
    route_df = get_route_geoms_ids(route_df, edges)
    assert "edge_geometries_ids" in route_df.columns


def test_get_pois_with_nodes(make_src_dst_matrix, topology_fixed):
    net, edges = make_graph(topology_fixed, precompute_distance=500)
    acled_data = get_acled_data_from_csv("tests/test_data/ACLED.csv", 6383)
    pois_df = get_pois_with_nodes(acled_data, net)
    expected_node_id = make_src_dst_matrix.loc[0, "from_nodeID"]
    valid_route = pois_df.loc[
        pois_df["nodeID"] == expected_node_id, "poi_list"
    ].values.tolist()
    assert len(valid_route) > 0
    assert "nodeID" in pois_df.columns


def test_get_incidents_in_route(routes_with_distances, topology_fixed):
    net, edges = make_graph(topology_fixed, precompute_distance=500)
    route_df = get_route_geoms_ids(routes_with_distances, edges)
    acled_data = get_acled_data_from_csv("tests/test_data/ACLED.csv", 6383)
    pois_df = get_pois_with_nodes(acled_data, net, max_dist=5000)
    incident_list = []
    for row in route_df.itertuples():
        incident_list.append(get_incidents_in_route(row, pois_df, acled_data, edges))
    incident_df = pd.concat(incident_list)
    assert "from_pcode" in incident_df.columns
    assert "to_pcode" in incident_df.columns
    assert "distance_to_route" in incident_df.columns


def test_calculations_with_sjoin(get_preprocessed_data):
    df_matrix, acled, _, edges = get_preprocessed_data
    incidents_in_route = get_incidents_in_route_sjoin(df_matrix, edges, acled, 1000)
    incidents_in_route.to_csv("tests/test_data/incidents_in_route.csv", index=False)
    assert len(incidents_in_route) == 1


def test_get_distances_to_route_experimental(get_preprocessed_data):
    df_matrix, acled, _, edges = get_preprocessed_data
    incidents_in_routes = get_incidents_in_route_sjoin(df_matrix, edges, acled, 1000)
    results = []
    for (from_pcode, to_pcode), group_df in incidents_in_routes.set_index(
        ["from_pcode", "to_pcode"]
    ).groupby(level=[0, 1]):
        results.append(
            get_distances_to_route_experimental(group_df, df_matrix, acled, edges)
        )
    result_df = pd.concat(results)
    assert "distance_to_route" in result_df.columns
    assert result_df.loc[0, "distance_to_route"] > 0
