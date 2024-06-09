from pathlib import Path

import geopandas as gpd
import networkx as nx
import pandana as pdna
import pytest
from shapely.geometry import LineString

from net_friction.data_preparation import (
    convert_pixels_to_points,
    fix_topology,
    get_acled_data_from_csv,
    get_source_destination_points,
    get_weighted_centroid,
    make_graph,
    data_pre_processing
)
from net_friction.datatypes import WeightingMethod


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
    raster_path = Path("tests/test_data/ukr_ppp.tif")
    gdf = gpd.read_file("tests/test_data/UKR_TEST_BOUNDARIES.gpkg")
    gdf = gdf[gdf.admin_level == 2]
    first_row = gdf.iloc[0]
    result = convert_pixels_to_points(raster_path, first_row)
    assert isinstance(result, gpd.GeoDataFrame)
    assert result.crs == "EPSG:4326"
    assert -9999 not in result["Value"].unique()


def test_get_weighted_centroid():
    gdf = gpd.read_file("tests/test_data/UKR_TEST_BOUNDARIES.gpkg")
    raster_path = Path("tests/test_data/ukr_ppp.tif")
    gdf = gdf[gdf.admin_level == 2]
    result = get_weighted_centroid(gdf, raster_path)
    gdf_result = gpd.GeoDataFrame(result, columns=["geometry"], crs="EPSG:4326")
    joined = gpd.sjoin(gdf_result, gdf, predicate="within")
    assert len(joined) == len(gdf_result)
    assert len(gdf) == len(result)


@pytest.mark.parametrize(
    "weighting_method", [WeightingMethod.WEIGHTED, WeightingMethod.CENTROID]
)
def test_get_source_destination_points(topology_fixed, weighting_method):
    gdf = gpd.read_file("tests/test_data/UKR_TEST_BOUNDARIES.gpkg")
    raster_path = Path("tests/test_data/ukr_ppp.tif")
    net, edges = make_graph(topology_fixed, precompute_distance=500)
    gdf = gdf[gdf.admin_level == 2]
    result = get_source_destination_points(
        gdf, weighting_method, net, edges.crs.to_epsg(), raster_path
    )
    assert len(result) == len(gdf) * (len(gdf) - 1) / 2
    assert all(
        col in result.columns
        for col in [
            "from_pcode",
            "from_nodeID",
            "from_centroid",
            "to_pcode",
            "to_nodeID",
            "to_centroid",
        ]
    )


@pytest.mark.parametrize("outpath", [None, "tests/test_data/test.gpkg"])
def test_get_acled_data_from_csv(outpath):
    acled_data = get_acled_data_from_csv(
        "tests/test_data/ACLED.csv", 6383, outfile=outpath
    )
    expected_columns = [
        "event_id_cnty",
        "event_date",
        "year",
        "disorder_type",
        "event_type",
        "sub_event_type",
        "latitude",
        "longitude",
        "fatalities",
        "geometry",
    ]
    assert isinstance(acled_data, gpd.GeoDataFrame)
    assert acled_data.crs == "EPSG:6383"
    assert acled_data.columns.tolist() == expected_columns
    if outpath:
        assert Path(outpath).exists()
        Path(outpath).unlink()


def test_data_pre_processing():
    roads = "tests/test_data/ROADS_TEST.shp"
    crs = 6383
    raster = "tests/test_data/ukr_ppp.tif"
    admin_boundaries = "tests/test_data/UKR_TEST_BOUNDARIES.gpkg"
    acled_data = "tests/test_data/ACLED.csv"
    admin_level = 2
    buffer_distance = 1000
    centroids_file = "tests/test_data/data_prep/centroids.gpkg"
    edges_file = "tests/test_data/data_prep/edges.gpkg"
    acled_out_file = "tests/test_data/data_prep/acled.csv"
    data_pre_processing(
        roads,
        crs,
        raster,
        admin_boundaries,
        acled_data,
        admin_level,
        buffer_distance,
        centroids_file,
        edges_file,
        acled_out_file,
        WeightingMethod.WEIGHTED,
    )
    assert Path(centroids_file).exists()
    assert Path(edges_file).exists()
    assert Path(acled_out_file).exists()
    # Path(centroids_file).unlink()
    # Path(edges_file).unlink()
    # Path(acled_out_file).unlink()
