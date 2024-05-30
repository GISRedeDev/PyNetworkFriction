from pathlib import Path
import geopandas as gpd

from net_friction.areas_of_control_matrix import calculate_src_dst_areas_of_control
from net_friction.data_preparation import get_weighted_centroid


def test_calculate_src_dst_areas_of_control():
    boundaries_df = gpd.read_file("tests/test_data/UKR_TEST_BOUNDARIES.gpkg")
    boundaries_df = boundaries_df[boundaries_df.admin_level == 3]
    raster = "tests/test_data/ukr_ppp.tif"
    centroids_df = boundaries_df.copy()
    centroids_df["geometry"] = get_weighted_centroid(boundaries_df, raster
                                         )
    result = calculate_src_dst_areas_of_control(
        centroids_df,
        "2024-04-01",
        "2024-04-30",
        Path("tests/test_data/control"),
        6383,
    )
    expected_columns = [
        "from_pcode",
        "to_pcode",
        "src_occupied",
        "dst_occupied",
        "date",
    ]
    assert list(result.columns) == expected_columns
