from pathlib import Path
from unittest.mock import Mock, patch

import geopandas as gpd
import networkx as nx
import pandana as pdna
import pytest
from shapely.geometry import LineString

from net_friction.data_preparation import (
    convert_pixels_to_points,
    data_pre_processing,
    fix_topology,
    get_acled_data_from_api,
    get_acled_data_from_csv,
    get_source_destination_points,
    get_weighted_centroid,
    make_graph,
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


MOCK_RESPONSE = """
    {"status":200,"success":true,"last_update":289,"count":122,"messages":[],"data":[{"event_id_cnty":"UKR142665",
    "event_date":"2024-01-01","year":"2024","time_precision":"1","disorder_type":"Political violence",
    "event_type":"Battles","sub_event_type":"Armed clash","actor1":"Military Forces of Russia (2000-)",
    "assoc_actor_1":"","inter1":"8","actor2":"Military Forces of Ukraine (2019-)","assoc_actor_2":"",
    "inter2":"1","interaction":"18","civilian_targeting":"","iso":"804","region":"Europe","country":"Ukraine",
    "admin1":"Donetsk","admin2":"Bakhmutskyi","admin3":"Bakhmutska","location":"Andriivka","latitude":"48.5008",
    "longitude":"37.9680","geo_precision":"2","source":"Ministry of Defence of Russia; Ministry of Defence of Ukraine",
    "source_scale":"Other-National","notes":"On 1 January 2024, Russian forces shelled and clashed with Ukrainian
    forces near Andriivka, Donetsk. According to Russian sources, up to 300 Ukrainian servicemen were killed near
    Bakhmut, Kurdiumivka, Andriivka, Bohdanivka, Hryhorivka, and Klischiivka. [Russian MoD reported 300 Ukrainian
    fatalities. Coded as 10 fatalities split across 6 events. 2 fatalities coded to this event].","fatalities":"2",
    "tags":"","timestamp":"1704833944"},{"event_id_cnty":"UKR142667","event_date":"2024-01-01","year":"2024",
    "time_precision":"1","disorder_type":"Political violence","event_type":"Explosions\\/Remote violence",
    "sub_event_type":"Shelling\\/artillery\\/missile attack","actor1":"Military Forces of Russia (2000-)",
    "assoc_actor_1":"","inter1":"8","actor2":"Civilians (Ukraine)","assoc_actor_2":"","inter2":"7","interaction":"78",
    "civilian_targeting":"Civilian targeting","iso":"804","region":"Europe","country":"Ukraine","admin1":"Donetsk",
    "admin2":"Pokrovskyi","admin3":"Avdiivska","location":"Avdiivka","latitude":"48.1394","longitude":"37.7497",
    "geo_precision":"1","source":"Suspilne Media","source_scale":"National","notes":"On 1 January 2024, Russian forces
    shelled Avdiivka, Donetsk. 1 civilian was killed.","fatalities":"1","tags":"","timestamp":"1704833944"},
    {"event_id_cnty":"UKR142668","event_date":"2024-01-01","year":"2024","time_precision":"1","disorder_type":
    "Political violence","event_type":"Explosions\\/Remote violence","sub_event_type":"Shelling\\/artillery\\/missile
      attack","actor1":"Military Forces of Russia (2000-)","assoc_actor_1":"","inter1":"8","actor2":"",
      "assoc_actor_2":"","inter2":"0","interaction":"80","civilian_targeting":"","iso":"804","region":"Europe",
      "country":"Ukraine","admin1":"Sumy","admin2":"Shostkynskyi","admin3":"Esmanska","location":"Bachivsk","latitude":
      "51.8450","longitude":"34.2775","geo_precision":"2","source":"Ministry of Defence of Ukraine","source_scale":
      "Other","notes":"On 1 January 2024, Russian forces shelled near Bachivsk, Sumy. Casualties unknown.","fatalities":
      "0","tags":"","timestamp":"1704833944"},{"event_id_cnty":"UKR142669","event_date":"2024-01-01","year":"2024",
      "time_precision":"1","disorder_type":"Political violence","event_type":"Battles","sub_event_type":"Armed clash",
      "actor1":"Military Forces of Ukraine (2019-)","assoc_actor_1":"","inter1":"1","actor2":"Military Forces of Russia
      (2000-)","assoc_actor_2":"Military Forces of Russia (2000-) Air Force","inter2":"8","interaction":"18",
      "civilian_targeting":"","iso":"804","region":"Europe","country":"Ukraine","admin1":"Donetsk","admin2":
      "Bakhmutskyi","admin3":"Bakhmutska","location":"Bakhmut","latitude":"48.5956","longitude":"37.9999",
      "geo_precision":"2","source":"Ministry of Defence of Russia","source_scale":"National","notes":"On 1 January 2024
      , Ukrainian forces assaulted Russian forces, who were supported by air units, near Bakhmut, Donetsk. According to
        Russian sources, up to 300 Ukrainian servicemen were killed near Bakhmut, Kurdiumivka, Andriivka, Bohdanivka,
        Hryhorivka, and Klischiivka. [Russian MoD reported 300 Ukrainian fatalities. Coded as 10 fatalities split
        across 6 events. 1 fatality coded to this event].","fatalities":"1","tags":"","timestamp":"1704833944"},
        {"event_id_cnty":"UKR142670","event_date":"2024-01-01","year":"2024","time_precision":"1","disorder_type":
        "Strategic developments","event_type":"Strategic developments","sub_event_type":"Disrupted weapons use",
        "actor1":"Military Forces of Russia (2000-)","assoc_actor_1":"","inter1":"8","actor2":"Military Forces of
        Ukraine (2019-) Air Force","assoc_actor_2":"","inter2":"1","interaction":"18","civilian_targeting":"","iso":
        "804","region":"Europe","country":"Ukraine","admin1":"Zaporizhia","admin2":"Polohivskyi","admin3":"Polohivska",
        "location":"Basan","latitude":"47.3773","longitude":"36.1811","geo_precision":"2","source":"Ministry of Defence
         of Russia","source_scale":"National","notes":"Interception: On 1 January 2024, Russian forces shot down a
         Ukrainian drone near Basan, Zaporizhia.","fatalities":"0","tags":"","timestamp":"1704833944"},{"event_id_cnty"
         :"UKR142671","event_date":"2024-01-01","year":"2024","time_precision":"1","disorder_type":"Political violence"
         ,"event_type":"Battles","sub_event_type":"Armed clash","actor1":"Military Forces of Russia (2000-)",
         "assoc_actor_1":"Military Forces of Russia (2000-) Air Force","inter1":"8","actor2":"Military Forces of
         Ukraine (2019-)","assoc_actor_2":"","inter2":"1","interaction":"18","civilian_targeting":"","iso":"804",
         "region":"Europe","country":"Ukraine","admin1":"Donetsk","admin2":"Pokrovskyi","admin3":"Ocheretynska",
         "location":"Berdychi","latitude":"48.1936","longitude":"37.6405","geo_precision":"2","source":"Institute for
          the Study of War; Ministry of Defence of Ukraine","source_scale":"Other","notes":"On 1 January 2024, Russian
          forces, supported by air units, clashed with and shelled Ukrainian forces near Berdychi, Donetsk. Casualties
          unknown.","fatalities":"0","tags":"","timestamp":"1704833944"},{"event_id_cnty":"UKR142672","event_date":
          "2024-01-01","year":"2024","time_precision":"1","disorder_type":"Political violence","event_type":
          "Explosions\\/Remote violence","sub_event_type":"Shelling\\/artillery\\/missile attack","actor1":"Military
          Forces of Russia (2000-)","assoc_actor_1":"","inter1":"8","actor2":"","assoc_actor_2":"","inter2":"0",
          "interaction":"80","civilian_targeting":"","iso":"804","region":"Europe","country":"Ukraine","admin1":
          "Kharkiv","admin2":"Kupianskyi","admin3":"Petpopavlivska","location":"Berestove","latitude":"49.5403",
          "longitude":"37.8941","geo_precision":"2","source":"Ministry of Defence of Ukraine","source_scale":"Other",
          "notes":"On 1 January 2024, Russian forces shelled near Berestove, Kharkiv. Casualties unknown.","fatalities"
          :"0","tags":"","timestamp":"1704833944"},{"event_id_cnty":"UKR142673","event_date":"2024-01-01","year":"2024",
          "time_precision":"1","disorder_type":"Political violence","event_type":"Explosions\\/Remote violence",
          "sub_event_type":"Shelling\\/artillery\\/missile attack","actor1":"Military Forces of Russia (2000-)",
          "assoc_actor_1":"","inter1":"8","actor2":"","assoc_actor_2":"","inter2":"0","interaction":"80",
          "civilian_targeting":
          "","iso":"804","region":"Europe","country":"Ukraine","admin1":"Luhansk","admin2":"Sievierodonetskyi","admin3"
          :"Lysychanska","location":"Bilohorivka","latitude":"48.9259","longitude":"38.2467","geo_precision":"2",
          "source":"Ministry of Defence of Ukraine","source_scale":"Other","notes":"On 1 January 2024, Russian forces
            shelled near Bilohorivka, Luhansk. Casualties unknown.","fatalities":"0","tags":"","timestamp":"1704833944"
            }],"filename":"2024-01-01-2024-01-01-Ukraine"}
    """.replace(
    "\n", ""
)


@patch("requests.get")
def test_get_acled_data_from_api(mock_get):
    mock_response = Mock()
    # mock_get.return_value = mock_response
    mock_response.text = MOCK_RESPONSE
    mock_response2 = Mock()
    mock_response2.text = """
        {"status":200,"success":true,"last_update":289,"count":122,"messages":[],"data":[], "filename":
        "2024-01-01-2024-01-01-Ukraine"}
    """
    mock_get.side_effect = [mock_response, mock_response2]
    gdf = get_acled_data_from_api(
        api_key="test_key",
        email="test_email",
        country="test_country",
        start_date="2024-01-01",
        end_date="2024-01-01",
        crs=4326,
        accept_acleddata_terms=True,
        outfile=None,
    )
    assert isinstance(gdf, gpd.GeoDataFrame)


def test_data_pre_processing():
    roads = "tests/test_data/ROADS_TEST.shp"
    crs = 6383
    raster = "tests/test_data/ukr_ppp.tif"
    admin_boundaries = "tests/test_data/UKR_TEST_BOUNDARIES.gpkg"
    admin_level = 2
    centroids_file = "tests/test_data/data_prep/centroids.gpkg"
    edges_file = "tests/test_data/data_prep/edges.gpkg"
    data_pre_processing(
        roads,
        crs,
        raster,
        admin_boundaries,
        admin_level,
        centroids_file,
        edges_file,
        WeightingMethod.WEIGHTED,
    )
    assert Path(centroids_file).exists()
    assert Path(edges_file).exists()
    # Path(centroids_file).unlink()
    # Path(edges_file).unlink()
    # Path(acled_out_file).unlink()
