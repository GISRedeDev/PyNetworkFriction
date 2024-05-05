import geopandas as gpd
from shapely.geometry import LineString
from net_friction.data_preparation import fix_topology


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
