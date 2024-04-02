"""Crude fix for the topology issue. This is a temporary fix and should be replaced with a more robust solution."""
from pathlib import Path
import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import unary_union, linemerge, split
from shapely.geometry import MultiLineString, LineString
import momepy
import pandas as pd
from shapely.ops import unary_union
import pandana as pdna
import numpy as np
from scipy.spatial import cKDTree
from itertools import combinations
import dask_geopandas as dg

def get_gdf(gpkg_path: Path | str , layer: str, crs: int) -> gpd.GeoDataFrame:
    return gpd.read_file(gpkg_path, layer=layer, crs=crs)


def fix_topology(gdf: gpd.GeoDataFrame, out_gpkg: Path | str, crs: int, layer: str = "roads", len_segments: int = 1000):
    merged = unary_union(gdf.geometry)
    geom = merged.segmentize(max_segment_length=len_segments)
    roads_multi = gpd.GeoDataFrame(data={"id": [1], "geometry": [geom]}, crs=f"EPSG:4326")
    gdf_roads = roads_multi.explode(ignore_index=True)
    gdf_roads = gdf_roads.to_crs(crs)
    gdf_roads["length"] = gdf_roads.length
    gdf_roads.to_file(out_gpkg, layer=layer)
    return gdf_roads


def make_graph(gdf: gpd.GeoDataFrame) -> nx.Graph:
    G = momepy.gdf_to_nx(gdf, approach="primal")
    nodes, edges, _ = momepy.nx_to_gdf(G, points=True, lines=True, spatial_weights=True)
    G = nx.MultiGraph()
    G.add_nodes_from(nodes.nodeID.unique().tolist())
    for index, row in edges.iterrows():
        G.add_edge(row.node_start, row.node_end, weight=row.length)
    print("GOT G GRAPH")
    net = pdna.Network(nodes.geometry.x, nodes.geometry.y, edges.node_start, edges.node_end, edges[["length"]])
    print("GOT NET GRAPH")
    net.precompute(5000)
    print("PRECOMPUTED")
    return G, nodes, edges, net


def make_centroids(gpkg: Path | str, crs: int, admin_level: int, layer: str | None = None) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(gpkg, layer=layer).to_crs(crs)
    gdf = gdf[gdf.admin_level == admin_level]
    gdf['centroid'] = gdf['geometry'].centroid
    gdf = gpd.GeoDataFrame(gdf[["pcode", "centroid"]], geometry="centroid")
    return gdf


def ckdnearest(gdf_centroids, gdf_nodes):
    """Returns nearest neighbours between 1 gdf and another
    Function derived from posting @ 
    https://gis.stackexchange.com/questions/222315/geopandas-find-nearest-point-in-other-dataframe
    with thanks to user JHuw
    """
    gdf_nodes = gdf_nodes.rename(columns={'geometry': 'geometry_dst'})
    try:
        nA = np.array(list(gdf_centroids.geometry.apply(lambda x: (x.x, x.y))))
        nB = np.array(list(gdf_nodes.geometry_dst.apply(lambda x: (x.x, x.y))))
        btree = cKDTree(nB)
        dist, idx = btree.query(nA, k=1)
        gdB_nearest = gdf_nodes.iloc[idx].reset_index(drop=True).nodeID
        gdf = pd.concat(
            [
                gdf_centroids.reset_index(drop=True),
                gdB_nearest
            ], 
            axis=1)
    except ValueError:
        gdf = gpd.GeoDataFrame({'y': [], 'x': [], 'band': [], 'spatial_ref': [], 'data': [], 'geometry': [], 'data_dst': [], 'geometry_dst': [], 'dist': []})
    return gdf[["pcode", "nodeID", "centroid"]]


def calc_euc_distance(row):
    src = row.from_centroid
    dst = row.to_centroid
    distance = src.distance(dst)
    return distance


def make_routes(df_nearest: gpd.GeoDataFrame, net: pdna.Network) -> gpd.GeoDataFrame:
    # FIXME Are we just going to drop disconnected nodes or find a way of connecting them?
    row_combinations = list(combinations(df_nearest[['pcode', 'nodeID', 'centroid']].values, 2))
    df = pd.DataFrame(row_combinations, columns=['from', 'to'])
    df[['from_pcode', 'from_nodeID', 'from_centroid']] = pd.DataFrame(df['from'].tolist(), index=df.index)
    df[['to_pcode', 'to_nodeID', 'to_centroid']] = pd.DataFrame(df['to'].tolist(), index=df.index)
    df = df.drop(columns=['from', 'to'])
    df["euclidean_dist"] = df.apply(calc_euc_distance, axis=1)
    # FIXME Can this part be parallelized or split?
    shortest_path_nodes = net.shortest_paths(df.from_nodeID, df.to_nodeID)
    df["shortest_path_nodes"] = shortest_path_nodes
    # FIXME Can this part be parallelized or split?
    df["shortest_path_length"] = net.shortest_path_lengths(df.from_nodeID, df.to_nodeID)
    df['list_length'] = df['shortest_path_nodes'].apply(len)
    #  G = momepy.gdf_to_nx(gdf_roads, approach="primal")

    # # Find the disconnected components
    # components = list(nx.connected_components(G))

    # # Print the number of disconnected components
    # print(f"Number of disconnected components: {len(components)}")

    # # Identify the largest component (main graph)
    # largest_component = max(components, key=len)

    # # Identify the nodes in the isolated parts of the graph
    # isolated_nodes = [node for component in components if component != largest_component for node in component]
    # print(f"Nodes in isolated parts of the graph: {isolated_nodes}")
    return df[df['list_length'] > 0]


def make_edges_dict(edges: gpd.GeoDataFrame) -> dict:
    edges_dict = {}
    for row in edges.itertuples():
        edges_dict[(row.node_start, row.node_end)] = row.Index
        edges_dict[(row.node_end, row.node_start)] = row.Index
    return edges_dict


def nodes_to_edges(row, edges_dict: dict):
    nodes = row.shortest_path_nodes
    return [edges_dict[(nodes[i-1], nodes[i])] for i in range(1, len(nodes))]


def edge_geometries(row: pd.Series, edges: gpd.GeoDataFrame):
    gdf = edges.loc[row.edge_geometries_ids]
    geom = gdf.geometry.unary_union
    return geom


def get_indcidents_df(incidents: Path, crs: int, save_points: bool = False) -> pd.DataFrame:
    incidents_df = pd.read_csv(incidents)
    incidents_df = incidents_df[[
        "event_id_cnty",
        "event_date",
        "year",
        "disorder_type",
        "event_type",
        "sub_event_type",
        "latitude",
        "longitude",
        "fatalities",
        "timestamp"
    ]]
    incidents_df["geometry"] = incidents_df.apply(lambda row: Point(row.longitude, row.latitude), axis=1)
    incidents_gdf = gpd.GeoDataFrame(incidents_df, geometry="geometry", crs=4326).to_crs(crs)
    if save_points:
        incidents_gdf.to_file(incidents.parent.joinpath(f"{incidents.stem}.gpkg"), layer="incidents", driver="GPKG")
    return incidents_gdf


def join_routes_incidents(df_routes: gpd.GeoDataFrame, incidents: gpd.GeoDataFrame, buffer: int = 1000) -> pd.DataFrame:
    d_incidents_gdf = dg.from_geopandas(incidents, npartitions=4)
    d_routes = dg.from_geopandas(df_routes, npartitions=4)
    d_incidents_gdf["geometry"] = d_incidents_gdf["geometry"].buffer(1000)
    gdf_joined = dg.sjoin(d_incidents_gdf, d_routes, how="inner", predicate="intersects")
    print("COMPUTING...")
    result = gdf_joined.compute()
    print("COMPUTED")
    print(result.head())
    agg_result = pd.DataFrame(result[[
        "from_pcode",
        "to_pcode",
        "euclidean_dist",
        "shortest_path_length",
        "event_id_cnty",
        "event_date",
        "year",
        "disorder_type",
        "event_type",
        "fatalities"
        ]])
    df_grouped = agg_result.groupby(['from_pcode', 'to_pcode', 'euclidean_dist', 'shortest_path_length', 'event_date', 'disorder_type']).agg(
        fatalities_sum=('fatalities', 'sum')
    ).reset_index()
    return df_grouped


def main():
    import time
    start_time = time.time()
    BASE = Path(__file__).resolve().parent.parent
    GPKG = BASE.joinpath("data/UKR_networks.gpkg")
    GPKG_ADMIN = BASE.joinpath("data/GEODATA.gpkg")
    INCIDENTS = BASE.joinpath("data/2022-02-01-2024-02-21-Europe-Ukraine.csv")
    layer = "roads"
    crs = 6383
    out_gpkg = BASE.joinpath("data/UKR_networks_fixed.gpkg")
    gdf = get_gdf(GPKG, layer, crs=4326)
    if not out_gpkg.exists():
        gdf_roads = fix_topology(gdf, out_gpkg, crs=crs)
    else:
        gdf_roads = gpd.read_file(out_gpkg, layer=layer, crs=crs)
    print("Topology fixed")
    G, nodes, edges, net = make_graph(gdf_roads)
    print("Graph created")
    centroids = make_centroids(GPKG_ADMIN, crs, 2, None)
    print("Centroids created")
    df_nearest = ckdnearest(centroids, nodes)
    del centroids
    df_routes = make_routes(df_nearest, net)
    del df_nearest, net, nodes
    print("Routes created")
    edges_dict = make_edges_dict(edges)
    print("Edges dictionary created")
    df_routes = df_routes.copy()
    df_routes.loc[:, 'edge_geometries_ids'] = df_routes.apply(nodes_to_edges, edges_dict=edges_dict, axis=1)
    df_routes.loc[:, "edge_geometries"] = df_routes.apply(edge_geometries, edges=edges, axis=1)
    del edges_dict
    print("Edge geometries created")
    incidents = get_indcidents_df(INCIDENTS, crs, save_points=True)
    print("Incidents loaded")
    df_routes = gpd.GeoDataFrame(
        df_routes[['from_pcode', 'to_pcode', 'euclidean_dist', "shortest_path_length", "edge_geometries"]],
        geometry="edge_geometries",
        crs=crs)
    print("CALCULATIING ROUTES AND INCIDENTS...")
    df_result = join_routes_incidents(df_routes, incidents)
    print("Routes and incidents joined ----> SAVING TO CSV...")
    df_result.to_csv(BASE.joinpath("data/routes_incidents_level_2.csv"), index=False)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    



if __name__ == "__main__":
    main()  






