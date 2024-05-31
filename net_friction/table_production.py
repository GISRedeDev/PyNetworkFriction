from pathlib import Path

import dask.dataframe as dd
import dask_geopandas as dask_gpd
import geopandas as gpd
import pandas as pd
from dask import compute, delayed

from .areas_of_control_matrix import calculate_src_dst_areas_of_control
from .calculations import (
    calculate_routes_and_route_distances,
    calculate_straight_line_distances,
    get_incidents_in_route,
    get_pois_with_nodes,
    get_route_geoms_ids,
)
from .data_preparation import (
    fix_topology,
    get_acled_data_from_csv,
    get_roads_data,
    get_source_destination_points,
    get_weighted_centroid,
    make_graph,
)
from .datatypes import WeightingMethod


def process_data(
    roads_data: Path | str,
    crs: int,
    raster: Path | str,
    admin_boundaries: Path | str,
    control_areas_dir: Path | str,
    aceld_data: Path | str,
    date_start: str,
    date_end: str,
    distance_matrix: Path | str,
    incidents_in_routes: Path | str,
    areas_of_control_matrix: Path | str,
    admin_level: int,
    roads_layer: str | None = None,
) -> None:
    roads = get_roads_data(
        roads_data,
        layer=roads_layer if roads_layer else None,
        crs=crs,
        subset_fields=["osm_id", "fclass"],
        subset_categories=["motorway", "trunk", "primary", "secondary", "tertiary"],
    )
    topology = fix_topology(roads, crs, len_segments=1000)
    net, edges = make_graph(topology, precompute_distance=5000)
    boundaries = gpd.read_file(admin_boundaries)
    boundaries = boundaries[boundaries.admin_level == admin_level]
    df_matrix = get_source_destination_points(
        boundaries,
        WeightingMethod.WEIGHTED,
        net,
        crs,
        Path(raster) if raster else None,
    )
    df_matrix["straight_line_distance"] = calculate_straight_line_distances(
        df_matrix, crs
    )
    shortest_path_nodes, shortest_path_lengths = calculate_routes_and_route_distances(
        net, df_matrix
    )
    df_matrix["shortest_path_nodes"] = shortest_path_nodes
    df_matrix["shortest_path_lengths"] = shortest_path_lengths
    acled = get_acled_data_from_csv(aceld_data, crs)
    df_matrix = get_route_geoms_ids(df_matrix.copy(), edges)
    pois_df = get_pois_with_nodes(acled, net)
    incidents_in_routes_list = []
    for row in df_matrix.itertuples():
        incidents_in_routes_list.append(
            get_incidents_in_route(row, pois_df, acled.copy(), edges)
        )
    incidents_in_routes_df = pd.concat(incidents_in_routes_list)
    distances_df = df_matrix[
        ["from_pcode", "to_pcode", "straight_line_distance", "shortest_path_lengths"]
    ]
    distances_df.to_csv(distance_matrix, index=False)
    incidents_in_routes_df.to_csv(incidents_in_routes, index=False)
    del net, edges, acled, pois_df, incidents_in_routes_df
    centroids_df = boundaries.copy()
    try:
        assert centroids_df.crs == "EPSG:4326"
    except AssertionError:
        centroids_df.to_crs("EPSG:4326", inplace=True)
    centroids_df["geometry"] = get_weighted_centroid(boundaries, Path(raster))
    areas_of_controls_df = calculate_src_dst_areas_of_control(
        centroids_df,
        start_date=date_start,
        end_date=date_end,
        polygon_dir=control_areas_dir,
        crs=crs,
    )
    areas_of_controls_df.to_csv(areas_of_control_matrix, index=False)


def process_data_dask(
    roads_data: Path | str,
    crs: int,
    raster: Path | str,
    admin_boundaries: Path | str,
    control_areas_dir: Path | str,
    aceld_data: Path | str,
    date_start: str,
    date_end: str,
    distance_matrix: Path | str,
    incidents_in_routes: Path | str,
    areas_of_control_matrix: Path | str,
    admin_level: int,
    roads_layer: str | None = None,
) -> None:
    roads = get_roads_data(
        roads_data,
        layer=roads_layer if roads_layer else None,
        crs=crs,
        subset_fields=["osm_id", "fclass"],
        subset_categories=["motorway", "trunk", "primary", "secondary", "tertiary"],
    )
    topology = fix_topology(roads, crs, len_segments=1000)
    net, edges = make_graph(topology, precompute_distance=5000)
    boundaries = gpd.read_file(admin_boundaries)
    boundaries = boundaries[boundaries.admin_level == admin_level]
    df_matrix = get_source_destination_points(
        boundaries,
        WeightingMethod.WEIGHTED,
        net,
        crs,
        Path(raster) if raster else None,
    )
    df_matrix["straight_line_distance"] = calculate_straight_line_distances(
        df_matrix, crs
    )
    shortest_path_nodes, shortest_path_lengths = calculate_routes_and_route_distances(
        net, df_matrix
    )
    df_matrix["shortest_path_nodes"] = shortest_path_nodes
    df_matrix["shortest_path_lengths"] = shortest_path_lengths
    df_matrix = dd.from_pandas(df_matrix, npartitions=10)
    acled = get_acled_data_from_csv(aceld_data, crs)
    acled = dask_gpd.from_geopandas(acled, npartitions=10)
    df_matrix = get_route_geoms_ids(df_matrix.copy(), edges)
    pois_df = get_pois_with_nodes(acled, net)
    pois_df = dd.from_pandas(pois_df, npartitions=10)
    delayed_results = [
        delayed(get_incidents_in_route)(row, pois_df, acled, edges)
        for row in df_matrix.itertuples(index=False)
    ]
    incidents_in_routes_list = compute(*delayed_results)
    incidents_in_routes_df = pd.concat(incidents_in_routes_list).compute()
    distances_df = df_matrix[
        ["from_pcode", "to_pcode", "straight_line_distance", "shortest_path_lengths"]
    ].compute()
    distances_df.to_csv(distance_matrix, index=False)
    incidents_in_routes_df.to_csv(incidents_in_routes, index=False)
    del net, edges, acled, pois_df, incidents_in_routes_df
    centroids_df = boundaries.copy()
    try:
        assert centroids_df.crs == "EPSG:4326"
    except AssertionError:
        centroids_df.to_crs("EPSG:4326", inplace=True)
    centroids_df["geometry"] = get_weighted_centroid(boundaries, Path(raster))
    areas_of_controls_df = calculate_src_dst_areas_of_control(
        centroids_df,
        start_date=date_start,
        end_date=date_end,
        polygon_dir=control_areas_dir,
        crs=crs,
    )
    areas_of_controls_df.to_csv(areas_of_control_matrix, index=False)
