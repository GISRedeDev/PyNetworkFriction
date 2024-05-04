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
