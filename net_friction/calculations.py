import warnings
from typing import Tuple

import geopandas as gpd
import pandana as pdna
import pandas as pd
import shapely
from shapely.geometry import Point


def calculate_straight_line_distances(
    src_dst_matrix: pd.DataFrame, crs: int
) -> gpd.GeoSeries:
    """Calculates the straight line distance between two points in src_dst_matrix. This dataframe must include  columns
    'from_centroid' and 'to_centroid' which are the centroids of the source and destination polygons respectively in
    the local crs.

    Args:
        src_dst_matrix (pd.DataFrame): Table including columns 'from_centroid' and 'to_centroid' points
        crs (int): Spatial reference system

    Returns:
        gpd.GeoSeries: Distances between 'from_centroid' and 'to_centroid' in meters
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        from_ = gpd.GeoSeries(src_dst_matrix.from_centroid, crs=crs)
        to_ = gpd.GeoSeries(src_dst_matrix.to_centroid, crs=crs)
    return from_.distance(to_)


def calculate_routes_and_route_distances(
    net: pdna.Network,
    src_dst_matrix: pd.DataFrame,
    chunk_size: int = 1000,
) -> Tuple[list, list]:
    """Calculates the shortest path between source and destination nodes in src_dst_matrix using the network net.

    Args:
        net (pdna.Network): Road network
        src_dst_matrix (pd.DataFrame): Dataframe with columns 'from_nodeID' and 'to_nodeID' which are the source
            and destination nodes respectively.
        chunk_size (int, optional): Chunksize for large dataframes. Defaults to 1000.

    Returns:
        Tuple[list, list]: Route node ids and route distances
    """
    shortest_path_nodes = []
    shortest_path_lengths = []
    for i in range(0, len(src_dst_matrix), chunk_size):
        src_dst_chunk = src_dst_matrix.iloc[i : i + chunk_size]
        shortest_path_nodes.extend(
            net.shortest_paths(
                src_dst_chunk.from_nodeID,
                src_dst_chunk.to_nodeID,
            )
        )
        shortest_path_lengths.extend(
            net.shortest_path_lengths(
                src_dst_chunk.from_nodeID,
                src_dst_chunk.to_nodeID,
            )
        )
    return shortest_path_nodes, shortest_path_lengths


def make_edges_dict(edges: gpd.GeoDataFrame) -> dict[tuple[int, int], int]:
    """Creates a dictionary of edges with keys as tuples of node_start and node_end and values as the index of the edge

    Args:
        edges (gpd.GeoDataFrame): Network edges

    Returns:
        dict[tuple[int, int], int]: Dict of edges (start_node, end_node) -> edge_index (int)
    """
    edges_dict = {}
    for row in edges.itertuples():
        edges_dict[(row.node_start, row.node_end)] = row.Index
        edges_dict[(row.node_end, row.node_start)] = row.Index
    return edges_dict


def nodes_to_edges(row, edges_dict: dict) -> list[int]:
    """Converts a list of nodes to a list of edges' ids

    Args:
        row (_type_): Row containing shortest path nodes
        edges_dict (dict): Dictionary of edges (start and end nodes -> edge index)

    Returns:
        list[int]: List of edge ids
    """
    nodes = row.shortest_path_nodes
    return [edges_dict[(nodes[i - 1], nodes[i])] for i in range(1, len(nodes))]


def edge_geometries(
    row: pd.Series, edges: gpd.GeoDataFrame
) -> shapely.geometry.base.BaseGeometry:
    """Creates a shapely geometry from a row of a dataframe and a geodataframe of edges

    Args:
        row (pd.Series): Row containing edge geometries ids
        edges (gpd.GeoDataFrame): Geodataframe of edges

    Returns:
        shapely.geometry.base.BaseGeometry: Edge geometry multilinestring
    """
    gdf = edges.loc[row.edge_geometries_ids]
    geom = chunked_unary_union(gdf, chunk_size=10000)
    return geom


def chunked_unary_union(
    gdf: gpd.GeoDataFrame, chunk_size: int = 10000
) -> shapely.geometry.base.BaseGeometry:
    """Creates a unary union of a geodataframe in chunks

    Args:
        gdf (gpd.GeoDataFrame): Geodataframe to union
        chunk_size (int, optional): Size to chunk dataframe. Defaults to 10000.

    Returns:
        shapely.geometry.base.BaseGeometry: Union of geodataframe geometries
    """
    unions = []
    for i in range(0, len(gdf), chunk_size):
        chunk = gdf.iloc[i : i + chunk_size]
        unions.append(chunk.unary_union)
    return shapely.ops.unary_union(unions)


def get_route_geoms_ids(
    route_df: pd.DataFrame,
    edges: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Gets the geometries of the edges in a route dataframe

    Args:
        route_df (pd.DataFrame): Dataframe with route node geometry ids
        edges (gpd.GeoDataFrame): Geodataframe of edges

    Returns:
        pd.DataFrame: Dataframe with edge ids applied
    """
    edges_dict = make_edges_dict(edges)
    route_df["edge_geometries_ids"] = route_df.apply(
        nodes_to_edges, args=(edges_dict,), axis=1
    )
    return route_df


def get_incidents_in_route_sjoin(
    matrix: pd.DataFrame,
    edges: gpd.GeoDataFrame,
    incidents: gpd.GeoDataFrame,
    buffer: int,
    incident_id_col: str = "event_id_cnty",
) -> pd.DataFrame:
    """Subsets the incidents in a route using a spatial join

    Args:
        matrix (pd.DataFrame): Table with source and destination nodes
        edges (gpd.GeoDataFrame): Edge geometries
        incidents (gpd.GeoDataFrame): Incident dataframe
        buffer (int): Size of buffer in which to perform the spatial join (meters)
        incident_id_col (str): Column name of the incident id. Defaults to "event_id_cnty" for use with ACLED data

    Returns:
        pd.DataFrame: Incident dataframe subset to those within buffer of route
    """
    incidents_buffer = incidents.set_index(incident_id_col).buffer(buffer)
    incidents_join = incidents_buffer.to_frame().sjoin(
        edges, how="left", predicate="intersects"
    )
    routes = matrix[["from_pcode", "to_pcode", "edge_geometries_ids"]]
    routes = routes.explode("edge_geometries_ids")
    df_joined = incidents_join.reset_index().merge(
        routes, left_on="index_right", right_on="edge_geometries_ids", how="inner"
    )
    df_final = df_joined.drop_duplicates(
        subset=[incident_id_col, "from_pcode", "to_pcode"]
    )
    df_final = df_final[[incident_id_col, "from_pcode", "to_pcode"]].set_index(
        incident_id_col
    )
    incidents_in_route = incidents.set_index(incident_id_col).merge(
        df_final, left_index=True, right_index=True
    )
    return pd.DataFrame(incidents_in_route.reset_index())


def get_edge_geometries(
    edge_ids: list, edges: gpd.GeoDataFrame
) -> shapely.geometry.base.BaseGeometry:
    """Map edge ids to edge geometries

    Args:
        edge_ids (list): List of edge ids
        edges (gpd.GeoDataFrame): Edge geometries

    Returns:
        shapely.geometry.base.BaseGeometry: Edge geometries representing the edge ids
    """
    gdf = edges.loc[edge_ids]
    geom = chunked_unary_union(gdf, chunk_size=10000)
    return geom


def calculate_distance_to_route(row, matrix, edges) -> float:
    edge_ids = matrix.loc[
        (matrix.from_pcode == row.from_pcode) & (matrix.to_pcode == row.to_pcode),
        "edge_geometries_ids",
    ].values[0]
    edge_geom = get_edge_geometries(edge_ids, edges)
    return edge_geom.distance(Point(row.geometry))


def get_distances_to_route(
    incidents: pd.DataFrame,
    matrix: pd.DataFrame,
    edges: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Calculates the distance of incidents to the route

    Args:
        incidents (pd.DataFrame): Incident points dataframe
        matrix (pd.DataFrame): Source destination matrix
        edges (gpd.GeoDataFrame):Edge geometries

    Returns:
        pd.DataFrame: _description_
    """
    incidents = incidents.reset_index()
    edge_ids = matrix.loc[
        (matrix.from_pcode == incidents.from_pcode.iloc[0])
        & (matrix.to_pcode == incidents.to_pcode.iloc[0]),
        "edge_geometries_ids",
    ].values[0]
    edge_geom = get_edge_geometries(edge_ids, edges)
    incidents["distance_to_route"] = edge_geom.distance(incidents.geometry)
    return incidents
