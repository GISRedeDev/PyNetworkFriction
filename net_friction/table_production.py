from pathlib import Path

import dask.dataframe as dd
import dask_geopandas as dask_gpd
import geopandas as gpd
import pandas as pd
from dask import compute, delayed
import numpy as np
import pandana as pdna

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


def save_network(
        net: pdna.Network,
        edges: gpd.GeoDataFrame,
        df_matrix: pd.DataFrame,
        outdir: Path,
        crs: int,
        admin_level: int,
) -> None:
    edges_file = outdir.joinpath(f"edges_{admin_level}.gpkg")
    nodes_file = outdir.joinpath(f"nodes_{admin_level}.gpkg")
    if not edges_file.exists() or not nodes_file.exists():
        edge_ids = df_matrix['edge_geometries_ids'].explode().unique()
        mask = np.isnan(pd.to_numeric(edge_ids, errors="coerce"))
        edge_ids = edge_ids[~mask]
        filtered_edges = edges.loc[edge_ids].to_crs("EPSG:4326")
        filtered_edges.to_file(outdir.joinpath(f"edges_{admin_level}.gpkg"), layer="geographic", driver="GPKG")
        if isinstance(df_matrix.loc[0, "shortest_path_nodes"], str):
            df_matrix['shortest_path_nodes'] = df_matrix['shortest_path_nodes'].apply(
                lambda x: [int(i) for i in x.replace("[", "").replace("]", "").split()]
            )
        nodes_df = net.nodes_df.loc[df_matrix['shortest_path_nodes'].explode().unique()]
        nodes_gdf = gpd.GeoDataFrame(nodes_df, geometry=gpd.points_from_xy(nodes_df.x, nodes_df.y), crs=f"EPSG:{crs}")
        nodes_gdf.to_crs("EPSG:4326", inplace=True)
        nodes_gdf.to_file(outdir.joinpath(f"nodes_{admin_level}.gpkg"), driver="GPKG")


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
    incidents_in_routes_aggregated: Path | str,
    areas_of_control_matrix: Path | str,
    admin_level: int,
    buffer_distance: int,
    centroids_file: Path | str,
    roads_layer: str | None = None,
    save_edges_and_nodes: bool = True,
) -> None:
    roads = get_roads_data(
        roads_data,
        layer=roads_layer if roads_layer else None,
        # crs=crs,
        # subset_fields=["osm_id", "fclass"],
        # subset_categories=["motorway", "trunk", "primary", "secondary", "tertiary"],
    )
    topology = fix_topology(roads, crs, len_segments=1000)
    net, edges = make_graph(topology, precompute_distance=5000)
    #save_network(net, edges, Path(roads_data).parent.joinpath("network.h5"))
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
    pois_df = get_pois_with_nodes(acled, net, max_dist=buffer_distance)
    if save_edges_and_nodes:
        save_network(net, edges, df_matrix, Path(centroids_file).parent, crs, admin_level)
    incidents_in_routes_list = []
    for row in df_matrix.itertuples():
        df_part = get_incidents_in_route(row, pois_df, acled.copy(), edges)
        if df_part is not None:
            incidents_in_routes_list.append(df_part)
    incidents_in_routes_list = [df.dropna(how='all', axis=1) for df in incidents_in_routes_list]
    incidents_in_routes_df = pd.concat(incidents_in_routes_list)
    distances_df = df_matrix[
        ["from_pcode", "to_pcode", "straight_line_distance", "shortest_path_lengths"]
    ]
    distances_df.to_csv(distance_matrix, index=False)
    incidents_in_routes_df = incidents_in_routes_df[incidents_in_routes_df.distance_to_route <= buffer_distance]
    incidents_in_routes_df.to_csv(incidents_in_routes, index=False)
    df_grouped = incidents_in_routes_df.groupby(
        ['event_date', 'from_pcode', 'to_pcode']).agg(
            incident_count=('event_id_cnty', 'count'),
            total_fatalities=('fatalities', 'sum'),
            mean_distance_to_route=('distance_to_route', 'mean')
    ).reset_index()
    df_grouped.to_csv(incidents_in_routes_aggregated, index=False)
    del net, edges, acled, pois_df, incidents_in_routes_df
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
    incidents_in_routes_aggregated: Path | str,
    areas_of_control_matrix: Path | str,
    admin_level: int,
    buffer_distance: int,
    centroids_file: Path | str,
    roads_layer: str | None = None,
    save_edges_and_nodes: bool = True,
) -> None:
    roads = get_roads_data(
        roads_data,
        layer=roads_layer if roads_layer else None,
        crs=crs,
        subset_fields=["osm_id", "fclass"],
        subset_categories=["motorway", "trunk", "primary", "secondary", "tertiary"],
    )
    print("got roads", admin_level)
    topology = fix_topology(roads, crs, len_segments=1000)
    print("fixed topology", admin_level)
    net, edges = make_graph(topology, precompute_distance=5000)
    print("got graph", admin_level)
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
    print("Got matrix", admin_level)
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
    pois_df = get_pois_with_nodes(acled, net, max_dist=buffer_distance)
    pois_df = dd.from_pandas(pois_df, npartitions=10)
    if save_edges_and_nodes:
        save_network(net, edges, df_matrix.compute(), Path(centroids_file).parent, crs, admin_level)
    delayed_results = [
        delayed(get_incidents_in_route)(row, pois_df, acled, edges)
        for row in df_matrix.itertuples(index=False)
    ]
    incidents_in_routes_list = compute(*delayed_results)
    incidents_in_routes_list = [x for x in incidents_in_routes_list if x is not None]
    incidents_in_routes_df = pd.concat(incidents_in_routes_list)
    print("Got incidents", admin_level)
    distances_df = df_matrix[
        ["from_pcode", "to_pcode", "straight_line_distance", "shortest_path_lengths"]
    ].compute()
    distances_df.to_csv(distance_matrix, index=False)
    incidents_in_routes_df = incidents_in_routes_df[incidents_in_routes_df.distance_to_route <= buffer_distance]
    incidents_in_routes_df.to_csv(incidents_in_routes, index=False)
    df_grouped = incidents_in_routes_df.groupby(
        ['event_date', 'from_pcode', 'to_pcode']).agg(
            incident_count=('event_id_cnty', 'count'),
            total_fatalities=('fatalities', 'sum'),
            mean_distance_to_route=('distance_to_route', 'mean')
    ).reset_index()
    df_grouped.to_csv(incidents_in_routes_aggregated, index=False)
    del net, edges, acled, pois_df, incidents_in_routes_df
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
    centroids_df["geometry"] = get_weighted_centroid(boundaries, Path(raster))
    areas_of_controls_df = calculate_src_dst_areas_of_control(
        centroids_df,
        start_date=date_start,
        end_date=date_end,
        polygon_dir=control_areas_dir,
        crs=crs,
    )
    areas_of_controls_df.to_csv(areas_of_control_matrix, index=False)

