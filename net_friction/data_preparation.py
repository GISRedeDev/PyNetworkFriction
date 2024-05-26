from itertools import combinations
from pathlib import Path

import geopandas as gpd  # type: ignore
import momepy
import networkx as nx
import numpy as np
import pandana as pdna
import pandas as pd  # type: ignore
import rioxarray
from shapely.geometry import Point
from shapely.ops import unary_union  # type: ignore
from xarray import DataArray

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
        roads = gpd.read_file(file_path, layer=layer)
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
    gdf_roads["length"] = gdf_roads.length
    return gdf_roads


def make_graph(gdf: gpd.GeoDataFrame, precompute_distance: int = 5000) -> pdna.Network:
    G_prep = momepy.gdf_to_nx(gdf, approach="primal")
    components = list(nx.connected_components(G_prep))
    largest_component = max(components, key=len)
    G = G_prep.subgraph(largest_component)

    nodes, edges, _ = momepy.nx_to_gdf(G, points=True, lines=True, spatial_weights=True)
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
        centroids.append(Point(weighted_x, weighted_y))
    return centroids


def get_source_destination_points(
    boundaries: gpd.GeoDataFrame,
    weighting_method: WeightingMethod,
    network: pdna.Network,
    raster: Path | None = None,
    admin_code_field: str = "pcode",
) -> pd.DataFrame:
    if weighting_method is WeightingMethod.CENTROID:
        boundaries["geometry"] = boundaries.representative_point()
    elif weighting_method is WeightingMethod.WEIGHTED and raster is not None:
        boundaries["geometry"] = get_weighted_centroid(boundaries, raster)
    centroids_df = boundaries[[admin_code_field, "geometry"]].copy()
    centroids_df["nodeID"] = network.get_node_ids(centroids_df.geometry.x, centroids_df.geometry.y)
    row_combinations = list(combinations(centroids_df[['pcode', 'nodeID', 'geometry']].values, 2))
    df_matrix = pd.DataFrame(row_combinations, columns=['from', 'to'])
    df_matrix[['from_pcode', 'from_nodeID', 'from_centroid']] = pd.DataFrame(
        df_matrix['from'].tolist(), index=df_matrix.index
        )
    df_matrix[['to_pcode', 'to_nodeID', 'to_centroid']] = pd.DataFrame(
        df_matrix['to'].tolist(), index=df_matrix.index
        )
    df_matrix = df_matrix.drop(columns=['from', 'to'])
    return df_matrix


# ------ ACLED ---------------------------------------------------------------------
# Access api and get the data for the dates required

# Save the data to a csv

# Convert to point dataset
