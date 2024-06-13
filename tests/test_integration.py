from pathlib import Path

import pytest

from net_friction.table_production import process_data


@pytest.mark.integration
def test_integration():
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning)
    BASE = Path(__file__).resolve().parent.joinpath("test_data")
    OUTPUT = BASE.joinpath("integration_test_output")
    roads_data = BASE.joinpath("ROADS_TEST.shp")
    crs = 6383
    raster = BASE.joinpath("ukr_ppp.tif")
    admin_boundaries = BASE.joinpath("UKR_TEST_BOUNDARIES.gpkg")
    control_areas_dir = BASE.joinpath("control")
    acled_data = BASE.joinpath("ACLED.csv")
    start_data = "2024-04-01"
    end_date = "2024-04-30"
    distance_matrix = OUTPUT.joinpath("distance_matrix.csv")
    incidents_in_routes = OUTPUT.joinpath("incidents_in_routes.csv")
    incidents_in_routes_aggregated = OUTPUT.joinpath(
        "incidents_in_routes_aggregated.csv"
    )
    areas_of_control_matrix = OUTPUT.joinpath("areas_of_control_matrix.csv")
    admin_level = 3
    buffer_distance = 1000
    centroids_file = OUTPUT.joinpath("centroids.gpkg")
    roads_layer = None
    subset_fields = None
    subset_categories = None
    process_data(
        roads_data,
        crs,
        raster,
        admin_boundaries,
        control_areas_dir,
        acled_data,
        start_data,
        end_date,
        distance_matrix,
        incidents_in_routes,
        incidents_in_routes_aggregated,
        areas_of_control_matrix,
        admin_level,
        buffer_distance,
        centroids_file,
        roads_layer,
        fixed_topology=True,
        subset_fields=subset_fields,
        subset_categories=subset_categories,
    )
    assert distance_matrix.exists()
    assert incidents_in_routes.exists()
    assert areas_of_control_matrix.exists()
