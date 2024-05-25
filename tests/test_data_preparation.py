import geopandas as gpd
import networkx as nx
import pandana as pdna
from shapely.geometry import LineString

from net_friction.data_preparation import (
    convert_pixels_to_points,
    fix_topology,
    get_weighted_centroid,
    make_graph,
)


def test_get_roads_data(get_test_roads_data_subset_and_projected):
    assert get_test_roads_data_subset_and_projected.crs.to_epsg() == 6383
    assert get_test_roads_data_subset_and_projected.columns.tolist() == [
        "osm_id",
        "fclass",
        "geometry",
    ]
    assert set(
        get_test_roads_data_subset_and_projected["fclass"].unique().tolist()
    ) == {"motorway", "trunk", "primary", "secondary", "tertiary"}


def test_fix_topology():
    data = {
        "id": [1, 2],
        "geometry": [LineString([(0, 0), (1, 1)]), LineString([(1, 1), (2, 2)])],
    }
    gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
    crs = 6383
    len_segments = 1000

    result = fix_topology(gdf, crs, len_segments)

    assert isinstance(result, gpd.GeoDataFrame)
    assert result.crs == "EPSG:6383"
    assert "length" in result.columns


def convert_pdna_to_nx(pdna_network, edges):
    G = nx.Graph()
    for node_id in pdna_network.node_ids:
        G.add_node(node_id)
    for edge in edges.itertuples():
        G.add_edge(edge.node_start, edge.node_end)
    return G


def test_make_graph(topology_fixed):
    net, edges = make_graph(topology_fixed, precompute_distance=500)
    assert isinstance(net, pdna.Network)
    assert isinstance(edges, gpd.GeoDataFrame)
    assert len(list(nx.connected_components(convert_pdna_to_nx(net, edges)))) == 1
    assert "length" in edges.columns
    assert "node_start" in edges.columns
    assert "node_end" in edges.columns


def test_convert_pixels_to_points():
    raster_path = "tests/test_data/ukr_ppp.tif"
    gdf = gpd.read_file("tests/test_data/UKR_TEST_BOUNDARIES.gpkg")
    gdf = gdf[gdf.admin_level == 2]
    first_row = gdf.iloc[0]
    result = convert_pixels_to_points(raster_path, first_row)
    assert isinstance(result, gpd.GeoDataFrame)
    assert result.crs == "EPSG:4326"
    assert -9999 not in result["Value"].unique()


def test_get_weighted_centroid():
    gdf = gpd.read_file("tests/test_data/UKR_TEST_BOUNDARIES.gpkg")
    raster_path = "tests/test_data/ukr_ppp.tif"
    gdf = gdf[gdf.admin_level == 2]
    result = get_weighted_centroid(gdf, raster_path)
    gdf_result = gpd.GeoDataFrame(result, columns=["geometry"], crs="EPSG:4326")
    joined = gpd.sjoin(gdf_result, gdf, predicate="within")
    assert len(joined) == len(gdf_result)
    assert len(gdf) == len(result)
