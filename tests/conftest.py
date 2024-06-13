from pathlib import Path

import geopandas as gpd
import pytest

from net_friction.calculations import (
    calculate_routes_and_route_distances,
    calculate_straight_line_distances,
)
from net_friction.data_preparation import (
    fix_topology,
    get_acled_data_from_csv,
    get_roads_data,
    get_route_geoms_ids,
    get_source_destination_points,
    make_graph,
)
from net_friction.datatypes import WeightingMethod

BASE_DATA = Path(__file__).resolve().parent.joinpath("test_data")


@pytest.fixture
def get_test_roads_data_subset_and_projected():
    yield get_roads_data(
        BASE_DATA.joinpath("ROADS_TEST.shp"),
        crs=6383,
        subset_fields=["osm_id", "fclass"],
        subset_categories=["motorway", "trunk", "primary", "secondary", "tertiary"],
    )


@pytest.fixture
def topology_fixed(get_test_roads_data_subset_and_projected):
    yield fix_topology(
        get_test_roads_data_subset_and_projected, 6383, len_segments=1000
    )


@pytest.fixture
def make_src_dst_matrix(topology_fixed):
    gdf = gpd.read_file("tests/test_data/UKR_TEST_BOUNDARIES.gpkg")
    raster_path = Path("tests/test_data/ukr_ppp.tif")
    net, edges = make_graph(topology_fixed, precompute_distance=500)
    gdf = gdf[gdf.admin_level == 2]
    result = get_source_destination_points(
        gdf, WeightingMethod.WEIGHTED, net, edges.crs.to_epsg(), raster_path
    )
    yield result


@pytest.fixture
def routes_with_distances(make_src_dst_matrix, topology_fixed):
    net, edges = make_graph(topology_fixed, precompute_distance=500)
    make_src_dst_matrix["straight_line_distance"] = calculate_straight_line_distances(
        make_src_dst_matrix, 6383
    )
    shortest_path_nodes, shortest_path_lengths = calculate_routes_and_route_distances(
        net, make_src_dst_matrix
    )
    make_src_dst_matrix["shortest_path_nodes"] = shortest_path_nodes
    make_src_dst_matrix["shortest_path_lengths"] = shortest_path_lengths
    yield make_src_dst_matrix


@pytest.fixture
def get_preprocessed_data():
    BASE = Path("tests/test_data/data_prep")
    roads = BASE.joinpath("edges.gpkg")
    centroids = BASE.joinpath("centroids.gpkg")
    acled_data = BASE.joinpath("acled.csv")
    raster = BASE.parent.joinpath("ukr_ppp.tif")
    admin_boundaries = BASE.parent.joinpath("UKR_TEST_BOUNDARIES.gpkg")
    crs = 6383

    roads = get_roads_data(roads, crs=6383)
    net, edges = make_graph(roads)
    boundaries = gpd.read_file(admin_boundaries)
    boundaries = boundaries[boundaries.admin_level == 2]
    df_matrix = get_source_destination_points(
        boundaries=boundaries,
        weighting_method=WeightingMethod.WEIGHTED,
        network=net,
        crs=crs,
        centroids_file=centroids,
        raster=Path(raster) if raster else None,
        admin_code_field="pcode",
    )
    df_matrix["straight_line_distance"] = calculate_straight_line_distances(
        df_matrix, crs
    )
    shortest_path_nodes, shortest_path_lengths = calculate_routes_and_route_distances(
        net, df_matrix
    )
    df_matrix["shortest_path_nodes"] = shortest_path_nodes
    df_matrix["shortest_path_lengths"] = shortest_path_lengths
    acled = get_acled_data_from_csv(acled_data, crs)
    df_matrix = get_route_geoms_ids(df_matrix.copy(), edges)
    yield df_matrix, acled, net, edges
