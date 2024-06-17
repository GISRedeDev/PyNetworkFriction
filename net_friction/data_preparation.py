from itertools import combinations
import json
from pathlib import Path

import geopandas as gpd  # type: ignore
import momepy
import networkx as nx
import numpy as np
import pandana as pdna
import pandas as pd  # type: ignore
import requests
import rioxarray
from shapely.geometry import LineString, Point
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


def fix_topology(gdf: gpd.GeoDataFrame, crs: int, len_segments: int = 1000):
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
    gdf: gpd.GeoDataFrame, precompute_distance: int = 5000
) -> tuple[pdna.Network, gpd.GeoDataFrame]:
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
    net.precompute(precompute_distance)
    return net, edges


def convert_pixels_to_points(raster: Path, polygon: gpd.GeoSeries) -> gpd.GeoDataFrame:
    raster_data = rioxarray.open_rasterio(raster)[0]
    assert isinstance(raster_data, DataArray)
    raster_data_clipped = raster_data.rio.clip([polygon.geometry])
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
    """This will be called if the weighted centroid is not within the polygon."""
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
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    centroids = []
    for polygon in gdf.itertuples():
        points = convert_pixels_to_points(raster, polygon)
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


def subset_acled_data(
    df: pd.DataFrame, crs: int, outfile: Path | str | None = None
) -> pd.DataFrame:
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
    df["geometry"] = gpd.points_from_xy(df.longitude, df.latitude)
    gdf = gpd.GeoDataFrame(df, crs="EPSG:4326")
    gdf = gdf.to_crs(f"EPSG:{crs}")
    if outfile:
        gdf.to_file(Path(outfile), layer="acled", driver="GPKG")
    return gdf


def data_pre_processing(
    roads_data: Path | str,
    crs: int,
    raster: Path | str,
    admin_boundaries: Path | str,
    acled_data: Path | str,
    admin_level: int,
    buffer_distance: int,
    centroids_file: Path | str,
    edges_file: Path | str,
    acled_out_file: Path | str,
    weight_method: WeightingMethod = WeightingMethod.WEIGHTED,
) -> None:
    """Preprocessing data for before main calculations will speed things up"""
    roads = get_roads_data(roads_data, crs=crs)
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
        Path(raster),
    )
    acled = get_acled_data_from_csv(acled_data, crs)
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
    acled_buffered = (
        acled.set_index("event_id_cnty").copy().buffer(buffer_distance).to_frame()
    )
    acled_join = acled_buffered.sjoin(edges, how="inner", predicate="intersects")
    acled = acled[acled["event_id_cnty"].isin(acled_join.index)]
    acled[[x for x in acled.columns if x != "geometry"]].to_csv(
        acled_out_file, index=False
    )


def get_acled_data_from_csv(
    csv_path: Path | str,
    crs: int,
    outfile: Path | str | None = None,
) -> gpd.GeoDataFrame:
    df = pd.read_csv(Path(csv_path))
    return subset_acled_data(df, crs, outfile=outfile)


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
    # TODO: Implement this function
    df_list = []
    page = 1
    while True:
        url = (
            f"https://api.acleddata.com/acled/read?terms={accept_acleddata_terms}"
            f"{api_key}"
            f"&email={email}"
            f"&country={country}"
            f"&event_date={start_date}/{end_date}"
            f"&page={page}"
            f"&export_type=csv"
        )
        response = requests.get(url)
        data = json.loads(response.text)["data"]
        if not data:
            break
        df_list.append(pd.DataFrame(data))
        page += 1
    df = pd.concat(df_list)
    return subset_acled_data(df, crs, outfile=outfile)
