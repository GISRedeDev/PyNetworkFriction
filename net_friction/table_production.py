from pathlib import Path

import geopandas as gpd
import pandas as pd

from .areas_of_control_matrix import calculate_src_dst_areas_of_control
from .calculations import (
    calculate_routes_and_route_distances,
    calculate_straight_line_distances,
    get_distances_to_route,
    get_incidents_in_route_sjoin,
    get_route_geoms_ids,
)
from .data_preparation import (
    fill_missing_routes,
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
    incidents_in_routes_outfile: Path | str,
    incidents_in_routes_aggregated: Path | str,
    areas_of_control_matrix: Path | str,
    admin_level: int,
    buffer_distance: int,
    centroids_file: Path | str,
    roads_layer: str | None = None,
    fix_road_topology: bool = False,
    subset_fields: list[str] | None = None,
    subset_categories: list[str] | None = None,
) -> None:
    """Helper function to process data for the net friction calculation

    Args:
        roads_data (Path | str): Roads data
        crs (int): CRS
        raster (Path | str): Raster data used for calculating weighted centroids
        admin_boundaries (Path | str): Boundaries data
        control_areas_dir (Path | str): Path in which areas of control polygons are defined
        aceld_data (Path | str): ACLED data
        date_start (str): Start date in format 'YYYY-MM-DD'
        date_end (str): End date in format 'YYYY-MM-DD'
        distance_matrix (Path | str): Distance matrix output file
        incidents_in_routes_outfile (Path | str): Incidents in routes output file
        incidents_in_routes_aggregated (Path | str): Incidents in routes aggregated output file
        areas_of_control_matrix (Path | str): Areas of control matrix output file
        admin_level (int): Admin level
        buffer_distance (int): Buffer distance in meters
        centroids_file (Path | str): Centroids file output/input (if it exists)
        roads_layer (str | None, optional): Roads layer. Defaults to None.
        fix_road_topology (bool, optional): Is it necessary to fix road topology. Defaults to False.
        subset_fields (list[str] | None, optional): Subset roads fields. Defaults to None.
        subset_categories (list[str] | None, optional): Subset roads categories. Defaults to None.
    """
    roads = get_roads_data(
        roads_data,
        layer=roads_layer if roads_layer else None,
        crs=crs,
        subset_fields=subset_fields,  # ["osm_id", "fclass"],
        subset_categories=subset_categories,  # ["motorway", "trunk", "primary", "secondary", "tertiary"],
    )
    if fix_road_topology:
        roads = fix_topology(roads, crs, len_segments=1000)
    net, edges = make_graph(roads, precompute_distance=5000)
    boundaries = gpd.read_file(admin_boundaries)
    boundaries = boundaries[boundaries.admin_level == admin_level]
    df_matrix = get_source_destination_points(
        boundaries=boundaries,
        weighting_method=WeightingMethod.WEIGHTED,
        network=net,
        crs=crs,
        centroids_file=centroids_file,
        raster=Path(raster) if raster else None,
        admin_code_field="pcode",
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
    incidents_in_routes = get_incidents_in_route_sjoin(
        df_matrix, edges, acled, buffer_distance
    )
    incidents_in_routes_list = []
    for (from_pcode, to_pcode), group_df in incidents_in_routes.set_index(
        ["from_pcode", "to_pcode"]
    ).groupby(level=[0, 1]):
        incidents_in_routes_list.append(
            get_distances_to_route(group_df, df_matrix, edges)
        )
    incidents_in_routes_df = pd.concat(incidents_in_routes_list)
    distances_df = df_matrix[
        ["from_pcode", "to_pcode", "straight_line_distance", "shortest_path_lengths"]
    ]
    distances_df.to_csv(distance_matrix, index=False)
    incidents_in_routes_df = incidents_in_routes_df[
        incidents_in_routes_df.distance_to_route <= buffer_distance
    ]
    incidents_in_routes_df.to_csv(incidents_in_routes_outfile, index=False)
    df_grouped = (
        incidents_in_routes_df.groupby(["event_date", "from_pcode", "to_pcode"])
        .agg(
            incident_count=("event_id_cnty", "count"),
            total_fatalities=("fatalities", "sum"),
            mean_distance_to_route=("distance_to_route", "mean"),
        )
        .reset_index()
    )
    df_grouped_filled = fill_missing_routes(
        df_grouped, distances_df, date_start, date_end
    )
    df_grouped_filled.to_csv(incidents_in_routes_aggregated, index=False)
    del net, edges, acled, incidents_in_routes_df
    centroids_df = boundaries.copy()
    try:
        assert centroids_df.crs == "EPSG:4326"
    except AssertionError:
        centroids_df.to_crs("EPSG:4326", inplace=True)
    if Path(centroids_file).exists():
        centroids_df = gpd.read_file(centroids_file)
    else:
        centroids_df["geometry"] = get_weighted_centroid(boundaries, Path(raster))
        centroids_df.to_file(centroids_file, driver="GPKG")
    areas_of_controls_df = calculate_src_dst_areas_of_control(
        centroids_df,
        start_date=date_start,
        end_date=date_end,
        polygon_dir=control_areas_dir,
        crs=crs,
    )
    areas_of_controls_df.to_csv(areas_of_control_matrix, index=False)
