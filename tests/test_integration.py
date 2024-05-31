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
    areas_of_control_matrix = OUTPUT.joinpath("areas_of_control_matrix.csv")
    admin_level = 3
    roads_layer = None
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
        areas_of_control_matrix,
        admin_level,
        roads_layer,
    )
    assert distance_matrix.exists()
    assert incidents_in_routes.exists()
    assert areas_of_control_matrix.exists()


@pytest.mark.integration
def test_dask_integration():
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning)
    BASE = Path(__file__).resolve().parent.joinpath("test_data")
    OUTPUT = BASE.joinpath("integration_test_dask_output")
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
    areas_of_control_matrix = OUTPUT.joinpath("areas_of_control_matrix.csv")
    admin_level = 3
    roads_layer = None
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
        areas_of_control_matrix,
        admin_level,
        roads_layer,
    )
    assert distance_matrix.exists()
    assert incidents_in_routes.exists()
    assert areas_of_control_matrix.exists()
