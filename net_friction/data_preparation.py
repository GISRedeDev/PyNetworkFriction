import json
from itertools import combinations
from pathlib import Path

import geopandas as gpd  # type: ignore
import momepy
import networkx as nx
import numpy as np
import pandana as pdna
import pandas as pd  # type: ignore
import requests
import rioxarray
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union  # type: ignore
from xarray import DataArray

from .calculations import calculate_routes_and_route_distances, get_route_geoms_ids
from .datatypes import WeightingMethod


def get_roads_data(
    file_path: Path | str,
    layer: str | None = None,
    crs: int | None = None,
    subset_fields: list | None = None,
    subset_categories: list | None = None,
) -> gpd.GeoDataFrame:
    """Get roads data from a shapefile or geopackage file subset by fields and categories. NOTE categories fieldname
    must be 'fclass' as this is the default fieldname used by OSM.

    Args:
        file_path (Path | str): Path to shapefile or geopackage file for roads data.
        layer (str | None, optional): Layer name in geopackage. Defaults to None.
        crs (int | None, optional): Spatial reference system local to roads network . Defaults to None.
        subset_fields (list | None, optional): Fields to subset. Defaults to None.
        subset_categories (list | None, optional): Categories to subset. Defaults to None.

    Raises:
        ValueError: If file type is not supported.

    Returns:
        gpd.GeoDataFrame: Subset geodataframe of roads data
    """
    file_path = Path(file_path)
    if file_path.suffix == ".shp":
        roads = gpd.read_file(file_path)
    elif file_path.suffix == ".gpkg":
        roads = gpd.read_file(file_path, layer=layer)
    else:
        raise ValueError("File type not supported. Please use shapefile or geopackage.")

    if crs:
        roads = roads.to_crs(crs)
    if subset_fields:
        subset_fields.append("geometry")
        roads = roads[subset_fields]
    if subset_categories:
        roads = roads[roads["fclass"].isin(subset_categories)]
    return roads


def fix_topology(
    gdf: gpd.GeoDataFrame, crs: int, len_segments: int = 1000
) -> gpd.GeoDataFrame:
    """Crudely fixes topology of roads data by segmentizing the roads into smaller segments in an effort to fix
    disconnected segments of the network. This is not a perfect solution but can help in some cases. It is recommended
    that the user check the output to ensure it is suitable for their use case and adjust the network manually to
    fit their needs.

    Args:
        gdf (gpd.GeoDataFrame): Roads data
        crs (int): CRS of the roads data
        len_segments (int, optional): Length of segments to segmentize. Defaults to 1000.

    Returns:
       gpd.GeoDataFrame: Roads with more detailed segments
    """
    gdf = gdf.to_crs(f"EPSG:{crs}")
    merged = unary_union(gdf.geometry)
    geom = merged.segmentize(max_segment_length=len_segments)
    roads_multi = gpd.GeoDataFrame(
        data={"id": [1], "geometry": [geom]}, crs=f"EPSG:{crs}"
    )
    gdf_roads = roads_multi.explode(ignore_index=True)
    gdf_roads.crs = f"EPSG:{crs}"
    gdf_roads["length"] = gdf_roads.length
    return gdf_roads


def make_graph(
    gdf: gpd.GeoDataFrame, precompute_distance: int | None = None
) -> tuple[pdna.Network, gpd.GeoDataFrame]:
    """Makes a graph from a geodataframe of roads data and an edges geodataframe. This function will return the
    largest connected component of the graph and drop disconnected parts. NOTE users should check the output to
    ensure it is suitable for their use case and adjust the network manually to fit their needs.

    Args:
        gdf (gpd.GeoDataFrame): Roads data
        precompute_distance (int | None, optional): Precompute distance for network (see Pandana docs).
            Defaults to None.

    Returns:
        tuple[pdna.Network, gpd.GeoDataFrame]: Network graph and edges geodataframe
    """
    G_prep = momepy.gdf_to_nx(gdf, approach="primal")
    components = list(nx.connected_components(G_prep))
    largest_component = max(components, key=len)
    G = G_prep.subgraph(largest_component)

    nodes, edges, _ = momepy.nx_to_gdf(G, points=True, lines=True, spatial_weights=True)
    edges.crs = gdf.crs
    net = pdna.Network(
        nodes.geometry.x,
        nodes.geometry.y,
        edges.node_start,
        edges.node_end,
        edges[["length"]],
    )
    if precompute_distance:
        net.precompute(precompute_distance)
    return net, edges


def convert_pixels_to_points(
    raster: Path, polygon_list: list[Polygon]
) -> gpd.GeoDataFrame:
    """Converts raster pixels to points within a polygon

    Args:
        raster (Path): Contunuous raster data file path
        polygon_list (list[shapely.geometry.Polygon]): Boundary polygon to use to clip the raster

    Raises:
        ValueError: Raster must be in EPSG:4326

    Returns:
        gpd.GeoDataFrame: Points with values from the raster
    """
    raster_data = rioxarray.open_rasterio(raster)[0]
    assert isinstance(raster_data, DataArray)
    raster_data_clipped = raster_data.rio.clip(polygon_list)
    try:
        assert raster_data_clipped.rio.crs.to_string() == "EPSG:4326"
    except AssertionError:
        raise ValueError("Raster crs is not EPSG:4326")
    x_coords = raster_data_clipped.x.values
    y_coords = raster_data_clipped.y.values
    x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)
    x_flat = x_mesh.flatten()
    y_flat = y_mesh.flatten()
    values_flat = raster_data_clipped.values.flatten()
    gdf = gpd.GeoDataFrame(
        {"Value": values_flat},
        geometry=gpd.points_from_xy(x_flat, y_flat),
        crs=raster_data.rio.crs.to_string(),
    )
    return gdf[gdf.Value != raster_data_clipped.rio.nodata]


def adjust_weighted_centroid(polygon: gpd.GeoSeries, weighted_centroid: Point) -> Point:
    """Function to bring weighted centroid inside the polygon in cases where it is outside the polygon

    Args:
        polygon (gpd.GeoSeries): Boundary polygon in which centroid is being calculated
        weighted_centroid (Point): Weighted centroid

    Returns:
        Point: Adjusted centroid within polygon
    """
    line = LineString([polygon.geometry.representative_point(), weighted_centroid])
    intersection = line.intersection(polygon.geometry)
    if intersection.geom_type == "LineString":
        adjusted_centroid = intersection.interpolate(
            intersection.project(weighted_centroid)
        )
    else:
        adjusted_centroid = intersection.representative_point()
    return adjusted_centroid


def get_weighted_centroid(
    gdf: gpd.GeoDataFrame,
    raster: Path,
) -> gpd.GeoSeries:
    """Calculated centroid of polygons based on weighted raster data

    Args:
        gdf (gpd.GeoDataFrame): Boudary polygons
        raster (Path): Continuous raster data used for weighting

    Returns:
        gpd.GeoSeries: Weighted centroids' geometries
    """
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    centroids = []
    for polygon in gdf.itertuples():
        points = convert_pixels_to_points(raster, [polygon.geometry])
        weighted_x = np.average(points.geometry.x, weights=points.Value)
        weighted_y = np.average(points.geometry.y, weights=points.Value)
        if not polygon.geometry.contains(Point(weighted_x, weighted_y)):
            centroids.append(
                adjust_weighted_centroid(polygon, Point(weighted_x, weighted_y))
            )
        else:
            centroids.append(Point(weighted_x, weighted_y))
    return centroids


def get_source_destination_points(
    boundaries: gpd.GeoDataFrame,
    weighting_method: WeightingMethod,
    network: pdna.Network,
    crs: int,
    centroids_file: Path | str,
    raster: Path | None = None,
    admin_code_field: str = "pcode",
) -> pd.DataFrame:
    """Make source and destination points for shortest path calculations

    Args:
        boundaries (gpd.GeoDataFrame): Areas for which to calculate shortest paths' matrix
        weighting_method (WeightingMethod): Centroid or weighted centroid
        network (pdna.Network): Roads network
        crs (int): Local CRS
        centroids_file (Path | str): Centroids file path (input (if exists) and output)
        raster (Path | None, optional): Path to raster to use for weights. Defaults to None.
        admin_code_field (str, optional):Boundary field name in boundaries dataframe. Defaults to "pcode".

    Returns:
        pd.DataFrame: Dataframe with source and destination matrix and centroid geometries
    """
    if weighting_method is WeightingMethod.CENTROID:
        boundaries["geometry"] = boundaries.representative_point()
    elif weighting_method is WeightingMethod.WEIGHTED and raster is not None:
        if centroids_file and Path(centroids_file).exists():
            boundaries = gpd.read_file(centroids_file)
        else:
            boundaries["geometry"] = get_weighted_centroid(boundaries, raster)
            boundaries.to_file(centroids_file, driver="GPKG")
    boundaries.to_crs(f"EPSG:{crs}", inplace=True)
    centroids_df = boundaries[[admin_code_field, "geometry"]].copy()
    centroids_df["geometry"] = centroids_df["geometry"].apply(
        lambda geom: geom.centroid if geom.geom_type != "Point" else geom
    )
    centroids_df["nodeID"] = network.get_node_ids(
        centroids_df.geometry.x, centroids_df.geometry.y
    )
    row_combinations = list(
        combinations(centroids_df[["pcode", "nodeID", "geometry"]].values, 2)
    )
    df_matrix = pd.DataFrame(row_combinations, columns=["from", "to"])
    df_matrix[["from_pcode", "from_nodeID", "from_centroid"]] = pd.DataFrame(
        df_matrix["from"].tolist(), index=df_matrix.index
    )
    df_matrix[["to_pcode", "to_nodeID", "to_centroid"]] = pd.DataFrame(
        df_matrix["to"].tolist(), index=df_matrix.index
    )
    df_matrix = df_matrix.drop(columns=["from", "to"])
    return df_matrix


def make_incident_data(
    df: pd.DataFrame, crs: int, outfile: Path | str | None = None
) -> pd.DataFrame:
    """Subset incident data to only include relevant columns and convert to geodataframe

    Args:
        df (pd.DataFrame): Dataframe of incident data (Must include latitude and longitude columns)
        crs (int): Local crs
        outfile (Path | str | None, optional): Location in which to save output if required.
            Defaults to None.

    Returns:
        pd.DataFrame: Subset ACLED data
    """
    df["geometry"] = gpd.points_from_xy(df.longitude, df.latitude)
    gdf = gpd.GeoDataFrame(df, crs="EPSG:4326")
    gdf = gdf.to_crs(f"EPSG:{crs}")
    if outfile:
        gdf.to_file(Path(outfile), driver="GPKG")
    return gdf


def make_incident_data_from_raster(
    raster: Path | str,
    roads: Path | str,
    buffer_distance: int,
    crs: int,
    incident_out_file: Path | str | None = None,
) -> gpd.GeoDataFrame:
    """Extracts incident data from raster data within buffer of roads, saves as csv (optional) and returns geodataframe

    Args:
        raster (Path | str): Incident raster in EPSG:4326
        roads (Path | str): Roads dataset in EPSG:4326 to be used as buffer
        buffer_distance (int): Buffer distance in meters in which to extract incidents
        crs (int): Spatial reference system
        incident_out_file (Path | str): Output location for incident data csv

    Returns:
        gpd.GeoDataFrame: Points representing pixels in buffer
    """
    gdf_roads = gpd.read_file(Path(roads)).to_crs(f"EPSG:{crs}")
    gdf_roads["geometry"] = gdf_roads.buffer(buffer_distance)
    buffer_polygon = gdf_roads.to_crs(4326).unary_union
    incident_data = convert_pixels_to_points(Path(raster), [buffer_polygon])
    if incident_out_file:
        df = incident_data.copy()
        df["latitude"] = df.geometry.y
        df["longitude"] = df.geometry.x
        df.drop(columns=["geometry"], inplace=True)
        df.to_csv(incident_out_file, index=False)
    return incident_data.to_crs(f"EPSG:{crs}")


def data_pre_processing(
    roads_data: Path | str,
    crs: int,
    admin_boundaries: Path | str,
    admin_level: int,
    centroids_file: Path | str,
    edges_file: Path | str,
    weight_method: WeightingMethod = WeightingMethod.WEIGHTED,
    raster: Path | str | None = None,
    subset_fields: list | None = None,
    subset_categories: list | None = None,
) -> None:
    """Helper function to preprocess data for shortest path calculations. These outputs can then be used in
    subsequent calculations.

    Args:
        roads_data (Path | str): Roads shapefile or geopackage file
        crs (int): Local CRS
        admin_boundaries (Path | str): Admin boundaries shapefile or geopackage file. This file should contain all
            admin levels and the admin level should be specified in the admin_level argument.
        admin_level (int): Admin level to subset from admin boundaries file named 'admin_level'.
        centroids_file (Path | str): Centroids file path (input (if exists) and output)
        edges_file (Path | str): Path to save edges file
        weight_method (WeightingMethod, optional): Weighting method to use. Defaults to WeightingMethod.WEIGHTED.
        raster (Path | str): Path to raster data if weighted centroid method is used
        subset_fields (list | None, optional): Roads field names to subset. Defaults to None.
        subset_categories (list | None, optional): Roads categories to subset. Defaults to None.
    """
    roads = get_roads_data(
        roads_data,
        crs=crs,
        subset_fields=subset_fields,
        subset_categories=subset_categories,
    )
    roads = fix_topology(roads, crs)
    net, edges = make_graph(roads)
    admin_boundaries_gdf = gpd.read_file(admin_boundaries)
    admin_boundaries_gdf = admin_boundaries_gdf[
        admin_boundaries_gdf["admin_level"] == admin_level
    ]
    source_dest_points = get_source_destination_points(
        admin_boundaries_gdf,
        weight_method,
        net,
        crs,
        centroids_file,
        Path(raster) if raster else None,
    )
    shortest_path_nodes, _ = calculate_routes_and_route_distances(
        net, source_dest_points
    )
    source_dest_points["shortest_path_nodes"] = shortest_path_nodes
    source_dest_points = get_route_geoms_ids(source_dest_points.copy(), edges)
    edge_ids = source_dest_points.explode("edge_geometries_ids")[
        "edge_geometries_ids"
    ].unique()
    edges = edges[edges.index.isin(edge_ids)]
    edges.to_file(edges_file, driver="GPKG")


def subset_incident_data_in_buffer(
    edges: gpd.GeoDataFrame,
    incident_data: Path | str,
    incident_out_file: Path | str,
    buffer_distance: int,
    crs: int,
    is_acled: bool = True,
    index_col: str = "event_id_cnty",
) -> gpd.GeoDataFrame:
    """Subset incidents to those within buffer of routes edges. If ACLED data is used, the function will subset
    columns to those required for analysis.

    Args:
        edges (gpd.GeoDataFrame): Edges dataframe extracted from full network
        incident_data (Path | str): Incident data
        incident_out_file (Path | str): Output location
        buffer_distance (int): Buffer distance in which to subset incidents
        crs (int): CRS
        is_acled (bool, optional): Is the incident data ACLED data. Defaults to True. If True, ACLED columns will
            be subset.
        index_col (str, optional): Index column. Defaults to "event_id_cnty".

    Returns:
        gpd.GeoDataFrame: Geodataframe of incidents within buffer
    """
    if is_acled:
        incident = get_acled_data_from_csv(Path(incident_data), crs)
    else:
        make_incident_data(pd.read_csv(Path(incident_data)), crs)
    incident_buffered = (
        incident.set_index(index_col).copy().buffer(buffer_distance).to_frame()
    )
    incident_join = incident_buffered.sjoin(edges, how="inner", predicate="intersects")
    incident = incident[incident[index_col].isin(incident_join.index)]
    incident[[x for x in incident.columns if x != "geometry"]].to_csv(
        Path(incident_out_file), index=False
    )
    return incident


def get_acled_data_from_csv(
    csv_path: Path | str,
    crs: int,
    outfile: Path | str | None = None,
) -> gpd.GeoDataFrame:
    """Read ACLED data from csv and subset columns

    Args:
        csv_path (Path | str): Path to csv
        crs (int): CRS
        outfile (Path | str | None, optional): Output location if required. Defaults to None.

    Returns:
        gpd.GeoDataFrame: Subset ACLED data
    """
    df = pd.read_csv(Path(csv_path))
    df = df[
        [
            "event_id_cnty",
            "event_date",
            "year",
            "disorder_type",
            "event_type",
            "sub_event_type",
            "latitude",
            "longitude",
            "fatalities",
        ]
    ].copy()
    return make_incident_data(df, crs, outfile=outfile)


def get_acled_data_from_api(
    api_key: str,
    email: str,
    country: str,
    start_date: str,
    end_date: str,
    crs: int,
    accept_acleddata_terms: bool,
    outfile: Path | str | None = None,
) -> gpd.GeoDataFrame:
    """Get ACLED data from the ACLED API in date range. This function will return a geodataframe of the data and
    requires an API key and email address to access the data. The data will be subset to columns required for analysis.

    Args:
        api_key (str): API key for ACLED
        email (str): Email used to access ACLED data
        country (str): Full country name (i.e. "Ukraine" not "UKR")
        start_date (str): Start date in format "YYYY-MM-DD"
        end_date (str): End date in format "YYYY-MM-DD"
        crs (int): Local CRS in which to project points
        accept_acleddata_terms (bool): Indicate acceptance of ACLED terms - See ACLED API documentation
        outfile (Path | str | None, optional): Location to save output if required. Defaults to None.

    Raises:
        ValueError: If no data is returned from the API

    Returns:
        gpd.GeoDataFrame: GeoDataFrame of ACLED point data in specified CRS
    """
    df_list = []
    page = 1
    while True:
        url = (
            f"https://api.acleddata.com/acled/read?terms={accept_acleddata_terms}"
            f"&key={api_key}"
            f"&email={email}"
            f"&country={country}"
            f"&event_date={start_date}|{end_date}&event_date_where=BETWEEN"
            f"&page={page}"
            f"&export_type=csv"
        )
        response = requests.get(url)
        if data := json.loads(response.text).get("data"):
            df_list.append(pd.DataFrame(data))
            page += 1
        else:
            break
    if df_list:
        df = pd.concat(df_list)
        df = df[
            [
                "event_id_cnty",
                "event_date",
                "year",
                "disorder_type",
                "event_type",
                "sub_event_type",
                "latitude",
                "longitude",
                "fatalities",
            ]
        ].copy()
        return make_incident_data(df, crs, outfile=outfile)
    raise ValueError("No data returned from ACLED API")


def fill_missing_routes(
    df_grouped: pd.DataFrame, df_distance: pd.DataFrame, date_start: str, date_end: str
) -> pd.DataFrame:
    """Insert source and destination points into dataframe for which there were no incidents in the date range

    Args:
        df_grouped (pd.DataFrame): Aggregated incidents dataframe
        df_distance (pd.DataFrame): Distances dataframe
        date_start (str): Start date in format "YYYY-MM-DD"
        date_end (str): End date in format "YYYY-MM-DD"

    Returns:
        pd.DataFrame: Dataframe with missing pairwise combinations filled with zeros
    """
    date_range = pd.date_range(start=date_start, end=date_end)
    dates_df = pd.DataFrame(date_range, columns=["event_date"])
    unique_routes_df = df_distance[["from_pcode", "to_pcode"]].drop_duplicates()
    all_combinations_df = pd.merge(
        dates_df.assign(key=1), unique_routes_df.assign(key=1), on="key"
    ).drop("key", axis=1)
    all_combinations_df["event_date"] = pd.to_datetime(
        all_combinations_df["event_date"]
    )
    df_grouped["event_date"] = pd.to_datetime(df_grouped["event_date"])
    full_df = pd.merge(
        all_combinations_df,
        df_grouped,
        on=["event_date", "from_pcode", "to_pcode"],
        how="left",
    )
    full_df["incident_count"] = full_df["incident_count"].fillna(0)
    full_df["total_fatalities"] = full_df["total_fatalities"].fillna(0)
    return full_df
