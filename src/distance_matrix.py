from pathlib import Path
import geopandas as gpd
import pandas as pd
import networkx as nx
import momepy


BASE_DIR = Path(__file__).resolve().parent
GPKG = "GEODATA.gpkg"
NETWORKS = "UKR_Networks.gpkg"


def get_nodes_edges(roads):
    # create a networkx graph from the geodataframe
    G = momepy.gdf_to_nx(roads, approach="primal")
    # create a geodataframe of the nodes
    nodes_gdf, edges_gdf = momepy.nx_to_gdf(G, points=True, lines=True)
    return nodes_gdf, edges_gdf


def make_graph(nodes_df, edges_df):
    G = nx.Graph()
    G.add_nodes_from(nodes_df.nodeID.unique().tolist())
    for index, row in edges_df.iterrows():
        G.add_edge(row.node_start, row.node_end)
    #G = G.remove_nodes_from(list(nx.isolates(G)))
    print("------------------", G)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def main():
    gdf = gpd.read_file(GPKG, layer="ADM1").to_crs("EPSG:6383").set_index("ADM1_PCODE")
    #roads = gpd.read_file(NETWORKS, layer="UKR_Networks").to_crs("EPSG:6383")
    roads = gpd.read_file(NETWORKS, layer="roads").to_crs("EPSG:6383")
    #roads = gpd.read_file(r"C:\Users\dkerr\Documents\GISRede\OXFORD_UNI_WORK\UKR_data_playground\ukraine-latest-free.shp\gis_osm_roads_free_1.shp").to_crs("EPSG:6383")
    print("GOT ROADS")
    print(roads.memory_usage(deep=True).sum() / 1e9)
    # make a series of the centroids of the polygons in gdf
    centroids = gdf.centroid
    assert "UA18" in centroids.index
    nodes_df, edges_df = get_nodes_edges(roads)
    del roads
    G = make_graph(nodes_df, edges_df)
    matrix_road_distance = pd.DataFrame(data={"from": [], "to": [], "distance": []})
    matrix_euc_distance = pd.DataFrame(data={"from": [], "to": [], "distance": []})
    count = 0
    for index_src, src in centroids.items():
        for index_dst, dst in centroids.items():
            edges_in_path = []
            src_node = nodes_df.loc[nodes_df.distance(src).idxmin()].nodeID
            dst_node = nodes_df.loc[nodes_df.distance(dst).idxmin()].nodeID
            if src_node != dst_node:
                try:
                    path = nx.shortest_path(G, src_node, dst_node)
                    for i in range(len(path) - 1):
                        start_node = path[i]
                        end_node = path[i + 1]
                        edge = edges_df[
                            (edges_df['node_start'] == start_node) & (edges_df['node_end'] == end_node) |
                            (edges_df['node_end'] == start_node) & (edges_df['node_start'] == end_node)
                        ]
                        edges_in_path.append(edge)
                    gdf_route = gpd.GeoDataFrame(pd.concat(edges_in_path))
                    road_length = gdf_route.length.sum()
                    data_roads = {"from": [index_src], "to": [index_dst], "distance": [road_length]}
                    matrix_road_distance = pd.concat([matrix_road_distance, pd.DataFrame(data_roads)])

                except nx.exception.NetworkXNoPath:
                    print(f"No path between {index_src} and {index_dst}")
                    count += 1
                    data_roads = {"from": [index_src], "to": [index_dst], "distance": [src.distance(dst)]}
                    matrix_road_distance = pd.concat([matrix_road_distance, pd.DataFrame(data_roads)])
                data_distance = {"from": [index_src], "to": [index_dst], "distance": [src.distance(dst)]}
                matrix_euc_distance = pd.concat([matrix_euc_distance, pd.DataFrame(data_distance)])
    matrix_road_distance['sorted_tuple'] = matrix_road_distance.apply(lambda row: tuple(sorted([row['from'], row['to']])), axis=1)
    matrix_road_distance = matrix_road_distance.drop_duplicates(subset='sorted_tuple')
    matrix_road_distance = matrix_road_distance.drop(columns='sorted_tuple')
    matrix_euc_distance['sorted_tuple'] = matrix_euc_distance.apply(lambda row: tuple(sorted([row['from'], row['to']])), axis=1)
    matrix_euc_distance = matrix_euc_distance.drop_duplicates(subset='sorted_tuple')
    matrix_euc_distance = matrix_euc_distance.drop(columns='sorted_tuple')
    matrix_road_distance.to_csv(BASE_DIR / "road_distance.csv")
    matrix_euc_distance.to_csv(BASE_DIR / "euc_distance.csv")
    matrix_road_distance.to_parquet(BASE_DIR / "road_distance.parquet")
    matrix_euc_distance.to_parquet(BASE_DIR / "euc_distance.parquet")
    print(f"Number of pairs with no path: {count/2}")

if __name__ == "__main__":
    main()
