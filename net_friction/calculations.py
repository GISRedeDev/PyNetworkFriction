import warnings
from typing import Tuple

import dask.dataframe as dd
import dask_geopandas as dg
import geopandas as gpd
import pandana as pdna
import pandas as pd
import shapely


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


def get_route_geoms(
    route_df: pd.DataFrame | dd.DataFrame,
    edges: gpd.GeoDataFrame | dg.GeoDataFrame,
) -> pd.DataFrame | dd.DataFrame:
    # FIXME What if these geoms were created on the fly when calculating distances?
    # and incidents in route?
    edges_dict = make_edges_dict(edges)
    if isinstance(route_df, dd.DataFrame):
        meta = pd.Series(dtype="object")
        route_df["edge_geometries_ids"] = route_df.map_partitions(
            lambda df: df.apply(nodes_to_edges, args=(edges_dict,), axis=1), meta=meta
        )
        route_df["edge_geometry"] = route_df.map_partitions(
            lambda df: df.apply(edge_geometries, args=(edges,), axis=1), meta=meta
        )
    else:
        route_df["edge_geometries_ids"] = route_df.apply(
            nodes_to_edges, args=(edges_dict,), axis=1
        )
        route_df["edge_geometry"] = route_df.apply(
            edge_geometries, args=(edges,), axis=1
        )
    route_df = route_df.drop(columns=["edge_geometries_ids"])
    return route_df


def get_pois_with_nodes(
    acled: gpd.GeoDataFrame, net: pdna.Network, max_dist: int = 1000
) -> pd.DataFrame:
    acled.set_index("event_id_cnty", inplace=True)
    max_items = int(max_dist / 10)
    num_pois = int(max_dist / 10)
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


def get_incidents_in_route(
    row: pd.Series,
    pois_df: pd.DataFrame,
    acled: gpd.GeoDataFrame,
) -> pd.DataFrame:
    acled.reset_index(inplace=True)
    route_nodes = row.shortest_path_nodes
    poi_nodes = pois_df[pois_df.nodeID.isin(route_nodes)]
    acled_ids = poi_nodes.poi_list.explode().dropna().unique().tolist()
    if len(acled_ids) > 0:
        incidents_in_route = acled[acled.event_id_cnty.isin(acled_ids)].copy()
        if not incidents_in_route.empty:
            incidents_in_route["from_pcode"] = row.from_pcode
            incidents_in_route["to_pcode"] = row.to_pcode
            incidents_in_route["distance_to_route"] = (
                incidents_in_route.geometry.distance(row.edge_geometry)
            )
            return incidents_in_route
    else:
        return pd.DataFrame(
            columns=[
                "event_id_cnty",
                "geometry",
                "from_pcode",
                "to_pcode",
                "distance_to_route",
            ]
        )
