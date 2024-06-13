import ast
import warnings
from typing import Tuple

import dask.dataframe as dd
import dask_geopandas as dg
import geopandas as gpd
import pandana as pdna
import pandas as pd
import shapely
from shapely.geometry import Point


def calculate_straight_line_distances(
    src_dst_matrix: pd.DataFrame, crs: int
) -> gpd.GeoSeries:
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
    edges_dict = {}
    for row in edges.itertuples():
        edges_dict[(row.node_start, row.node_end)] = row.Index
        edges_dict[(row.node_end, row.node_start)] = row.Index
    return edges_dict


def nodes_to_edges(row, edges_dict: dict) -> list[int]:
    nodes = row.shortest_path_nodes
    if isinstance(nodes, str):  # Dask bug
        nodes = [int(node) for node in nodes.replace("[", "").replace("]", "").split()]
    return [edges_dict[(nodes[i - 1], nodes[i])] for i in range(1, len(nodes))]


def edge_geometries(
    row: pd.Series, edges: gpd.GeoDataFrame
) -> shapely.geometry.base.BaseGeometry:
    gdf = edges.loc[row.edge_geometries_ids]
    geom = chunked_unary_union(gdf, chunk_size=10000)
    return geom


def chunked_unary_union(
    gdf: gpd.GeoDataFrame, chunk_size: int = 10000
) -> shapely.geometry.base.BaseGeometry:
    unions = []
    for i in range(0, len(gdf), chunk_size):
        chunk = gdf.iloc[i : i + chunk_size]
        unions.append(chunk.unary_union)
    return shapely.ops.unary_union(unions)


def get_route_geoms_ids(
    route_df: pd.DataFrame | dd.DataFrame,
    edges: gpd.GeoDataFrame | dg.GeoDataFrame,
) -> pd.DataFrame | dd.DataFrame:
    edges_dict = make_edges_dict(edges)
    if isinstance(route_df, dd.DataFrame):
        meta = pd.Series(dtype="object")
        route_df["edge_geometries_ids"] = route_df.map_partitions(
            lambda df: df.apply(nodes_to_edges, args=(edges_dict,), axis=1), meta=meta
        )
    else:
        route_df["edge_geometries_ids"] = route_df.apply(
            nodes_to_edges, args=(edges_dict,), axis=1
        )
    return route_df


# TODO: This function is not used. Should it be removed?
def get_pois_with_nodes(
    acled: gpd.GeoDataFrame, net: pdna.Network, max_dist: int = 1000
) -> pd.DataFrame:
    if isinstance(acled, dd.DataFrame) or isinstance(acled, dg.GeoDataFrame):
        acled = acled.compute()
    acled.set_index("event_id_cnty", inplace=True)
    max_items = 800 # TODO: Is this catching everything?
    num_pois = 800
    max_dist = max_dist * 5  # This is because in some instances, pois within route buffers are still far from nodes,
    # resulting in them not being counted
    net.set_pois(
        category="incidents",
        maxdist=max_dist,
        maxitems=max_items,
        x_col=acled.geometry.x,
        y_col=acled.geometry.y,
    )
    pois_df = net.nearest_pois(
        distance=max_dist, category="incidents", num_pois=num_pois, include_poi_ids=True
    )
    poi_cols = [col for col in pois_df.columns if isinstance(col, str) if "poi" in col]
    pois_df["poi_list"] = pois_df[poi_cols].apply(
        lambda row: row.dropna().tolist(), axis=1
    )
    pois_df["nodeID"] = pois_df.index.copy()
    pois_df = pois_df[["nodeID", "poi_list"]].set_index("nodeID")
    return pois_df.reset_index()


def get_incidents_in_route_sjoin(
        matrix: pd.DataFrame | dd.DataFrame,
        edges: gpd.GeoDataFrame | dg.GeoDataFrame,
        acled: gpd.GeoDataFrame | dg.GeoDataFrame,
        buffer: int,
) -> pd.DataFrame:
    acled_buffer = acled.set_index("event_id_cnty").buffer(buffer)
    acled_join = acled_buffer.to_frame().sjoin(edges, how="left", predicate="intersects")
    routes = matrix[["from_pcode", "to_pcode", "edge_geometries_ids"]]
    routes = routes.explode("edge_geometries_ids")
    df_joined = acled_join.reset_index().merge(
        routes, left_on='index_right', right_on='edge_geometries_ids', how='inner'
    )
    df_final = df_joined.drop_duplicates(subset=['event_id_cnty', 'from_pcode', 'to_pcode'])
    df_final = df_final[['event_id_cnty', 'from_pcode', 'to_pcode']].set_index('event_id_cnty')
    incidents_in_route = acled.set_index('event_id_cnty').merge(df_final, left_index=True, right_index=True)
    return pd.DataFrame(incidents_in_route.reset_index())


def get_edge_geometries(
    edge_ids: list, edges: gpd.GeoDataFrame
) -> shapely.geometry.base.BaseGeometry:
    gdf = edges.loc[edge_ids]
    geom = chunked_unary_union(gdf, chunk_size=10000)
    return geom


def calculate_distance_to_route(row, matrix, edges)-> float:
    edge_ids = matrix.loc[(
        matrix.from_pcode == row.from_pcode) & (matrix.to_pcode == row.to_pcode),
        "edge_geometries_ids"
    ].values[0]
    edge_geom = get_edge_geometries(edge_ids, edges)
    return edge_geom.distance(Point(row.geometry))


def get_distances_to_route_experimental(
        incidents: pd.DataFrame | dd.DataFrame,
        matrix: pd.DataFrame | dd.DataFrame,
        acled: gpd.GeoDataFrame,
        edges: gpd.GeoDataFrame,
) -> pd.DataFrame:
    acled = acled.reset_index()
    incidents = incidents.reset_index()
    edge_ids = matrix.loc[
        (matrix.from_pcode == incidents.from_pcode.iloc[0]) & (matrix.to_pcode == incidents.to_pcode.iloc[0]),
        "edge_geometries_ids"
    ].values[0]
    edge_geom = get_edge_geometries(edge_ids, edges)
    incidents["distance_to_route"] = edge_geom.distance(incidents.geometry)
    return incidents


def get_incidents_in_route(
    row: pd.Series,
    pois_df: pd.DataFrame,
    acled: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
) -> pd.DataFrame:
    acled = acled.reset_index()
    route_nodes = row.shortest_path_nodes
    if isinstance(route_nodes, str):
        route_nodes = [int(node) for node in route_nodes.replace("[", "").replace("]", "").split()]
    poi_nodes = pois_df[pois_df.nodeID.isin(route_nodes)]
    if isinstance(poi_nodes.iloc[0].poi_list, str):
        poi_nodes["poi_list"] = poi_nodes["poi_list"].apply(
            lambda x: ast.literal_eval(x)
        )
    acled_ids = poi_nodes.poi_list.explode().dropna().unique().tolist()
    if len(acled_ids) > 0:
        incidents_in_route = acled[acled.event_id_cnty.isin(acled_ids)].copy()
        if not incidents_in_route.empty:
            incidents_in_route["from_pcode"] = row.from_pcode
            incidents_in_route["to_pcode"] = row.to_pcode
            edge_geom = edge_geometries(row, edges)
            incidents_in_route["distance_to_route"] = edge_geom.distance(
                incidents_in_route.geometry
            )
            return incidents_in_route
    return pd.DataFrame(
        columns=[
            "event_id_cnty",
            "geometry",
            "from_pcode",
            "to_pcode",
            "distance_to_route",
        ]
    )
