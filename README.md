# networkFriction

## Description
net-friction is a Python package that provides functionalities for geographical data processing and analysis. It is designed to work with geospatial data, specifically road networks and points of interest (POIs), and perform operations such as creating graphs from road data, generating centroids, creating routes, and finding incidents in routes.

## Installation
To install net-friction, use pip:

## Usage
Here is a basic example of how to use net-friction:

## Functions
```
make_graph(gdf_roads): Creates a graph from road data.
make_centroids(gpkg_admin, crs, admin_level): Generates centroids from administrative data.
make_routes(centroids, net): Creates routes using the centroids and the network.
get_pois_with_nodes(acled_gdf, net): Gets points of interest with nodes.
get_route_geoms(route_df, edges): Gets the geometries of the routes.
get_incidents_in_route(route_df, pois_df, acled_gdf): Finds incidents in the routes.
```
Contributing
Contributions are welcome. Please submit a pull request or create an issue to discuss the changes you want to make.

License
This project is licensed under the MIT License.