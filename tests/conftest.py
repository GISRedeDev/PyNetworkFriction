from pathlib import Path

import pytest

from net_friction.data_preparation import fix_topology, get_roads_data

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
