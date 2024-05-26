from pathlib import Path

import geopandas as gpd
import pytest

from net_friction.data_preparation import (
    fix_topology,
    get_roads_data,
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
    net, _ = make_graph(topology_fixed, precompute_distance=500)
    gdf = gdf[gdf.admin_level == 2]
    result = get_source_destination_points(
        gdf, WeightingMethod.WEIGHTED, net, raster_path
    )
    yield result
