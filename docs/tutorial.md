# Tutorial and Example Workflow

## Input data
To generate the outputs as intended by the package, the following input data is required:
1. **Road network shapefile or geopackage** (OSM suggested). This can be the full dataset, or subset to the required routes' edges as will be demonstrated in data preparation. If repeated analysis is required, it is advisable to save these edges, and refer to these saved edges in this variable, rather than reading in the full dataset in each iteration - `roads`.

2. **Incident data** in csv format with `latitude` and `longitude` columns in point format with a CRS of `EPSG:4326` representing the incidents' locations. See below for functionality to access this data from the ACLED API - `incidents`.

3. **Areas of control directory** (if analysis being carried out). The data in this directory will be used to calculate areas controlled by local (0) or outside forces (1). See [GISRedeDev/AreasOfControl](https://github.com/GISRedeDev/AreasofControl) to download this data in the context of the Ukranian conflict. The files in this directory should be named in the format `occupied_<YYYY-MM-DD>.gpkg` for each day of the data extracted - `areas_of_control_dir`.

4. **Weighting raster** to be used to weight administrative areas' centroids to a continuous spatial variable to make source and destination points for respective areas indicative of the most densely populated locations, rather than their geometric centroids. Population density or degree of urbanisation rasters are a good indication for this variable. All rasters should be in CRS `EPSG:4326` - `raster`.

5. **Administrative boundaries geopackage or shapefile** to be used for the source/destination matrix in the network analysis (i.e. GADM). The attribute of this table must have a field name `admin_level` denoting the administrative level the rows' geometries represent. If doing analysis for multiple levels, these can all be placed within the same table - `admin_boundaries`.


## Data Preparation

### 1. Roads / Network Data (incl Source/Destination Matrix)
Roads data can be prepared by creating a weighted centroid object, opening the roads data and creating a graph and edges objects, along with a source/destination matrix dataframe.

```python
import geopandas as gpd
from net_friction import data_preparation as prep
from net_friction import datatypes as dt
from net_fricetion import calculations as calc

# Continuous raster for use in weighting
raster = "population.tif"
# Boundaries data (geodataframe) including an admin_level field and admin unit code field (i.e. pcode)
boundaries = gpd.read_file("boundaries.gpkg")
# Weighted centroid output file
centroids = "boundary_centroids.gpkg"
# Weighting method (CENTROID/WEIGHTED)
weighting_method = df.WEIGHTED

### Open OSM roads
subset_fields = ["osm_id", "fclass"]  # Fields in OSM data
subset_categories = ["motorway", "trunk", "primary", "secondary", "tertiary"]  # fclass in OSM data
roads_gdf = prep.get_roads_data(roads, crs, subset_fields, subset_categories)

# Crude topology fix
roads_gdf = prep.fix_topology(roads, src, len_segments=1000)

# Get network object and edges dataframe of full data
net, edges = prep.make_graph(roads)

# Create source/destination points weighted by raster
src_dst_points = prep.get_source_destination_points(
    boundaries,
    weighting_method,
    net,
    crs,
    centroids,
    raster
)

# Get shortest path nodes between source/destination pairs
shortest_path_nodes, shortest_path_lengths = calculate_routes_and_route_distances(
        net, source_dest_points
    )
source_dest_points["shortest_path_nodes"] = shortest_path_nodes

# Subset 'global' edges to edges between source and destination pairs
route_geom_ids = get_route_geoms_ids(source_dest_points.copy(), edges)
edge_ids = route_geom_ids.explode("edge_geometries_ids")["edge_geometries_ids"].unique()
edges_subset = egdes[edges.index.isin(edge_ids)]

# Save edges as future input for roads for improved performance
edges.to_file("edges.gpkg", driver="GPKG")
```
**NOTE** There is a helper function to carry out the above preprocessing `net_friction.data_preparation.data_pre_processing`

### 2. Incident data
Any incident data in `EPSG:4326` can be used in csv format but must have `latitude` and `longitude` in point WKT format as columns. These incident data can be converted to a GeoDataFrame using `netfriction.data_preparation.make_incident_data`. There is also a helper function `netfriction.data_preparation.get_acled_data_from_api` to access ACLED data from the API, but users are required to have a key and email address registered with [ACLED](https://acleddata.com/).

```python
import pandas as pd
from net_friction.data_preparation import get_acled_data_from_api, make_incident_data, make_incident_data_from_raster

# Incident data
csv = "incidents.csv"
crs = 27700
incident_df = pd.read_csv(csv)
incident_gdf = get_incident_data(incident_df, crs)

# ACLED DATA
key = "secret_key"
email = "user@github.com"
country = "United Kingdom"
start_date = "2024-01-31"
end_date = "2024-02-05"
crs = 27700
accept_acled_terms = True
incident_gdf = get_acled_data_from_api(
    key,
    email,
    country,
    start_date,
    end_date,
    crs,
    accept_acled_terms
)

# Raster data in WGS84
raster_path = "population.tif"
roads = "edges.gpkg"
buffer_distance = 1000
crs = 27700
incident_out_file = "population_within_1000m_of_routes.csv"
incident_gdf = make_incident_data_from_raster(
    raster=raster,
    roads=roads,
    buffer_distance=buffer_distance,
    crs=crs,
    incident_out_file=incident_out_file,
)

```

### 3. Subset incident data within proximity of roads
To discard incident data outside of your area of distance from the roads network, and thus improve performance, subsut the incident data.

```python
from net_friction.data_preparation import subset_incident_data_in_buffer

incidents_outfile = "incidents_subset_in_buffer.csv"
buffer_distance_in_meters = 1000

incidents_gdf = subset_incident_data_in_buffer(
    edges,
    csv,
    incidents_outfile,
    buffer_distance_in_meters,
    crs,
    is_acled=True,
    index_col="event_id_cnty"  # Unique ID field in incidents table
)
```


## Network Analysis
Once data has been prepared, the network analysis can be processed. Please note that if this analysis will be processed in multiple iterations or different periods, it is best to save some of the above outputs, and use these as inputs to the analysis. This will help to speed up the performance.

### 1. Calculate the distances
Distances along the network and in straight lines can be calculated between each source and destination pair.

```python
import pandas as pd

# Straight line distance
src_dst_points["straight_line_distance"] = calculate_straight_line_distances(
    src_dst_points, crs
)

# Network distances between points
shortest_path_nodes, shortest_path_lengths = calc.calculate_routes_and_route_distances(
        net, src_dst_points
    )
src_dst_points["shortest_path_nodes"] = shortest_path_nodes
src_dst_points["shortest_path_lengths"] = shortest_path_lengths

# Add source/destination pairwise combinations to each row in the incidents dataframe which lie
# within the buffer distance from the respective network routes (i.e. Get incidents in buffer
# from route)
incidents_in_routes = calc.get_incidents_in_route_sjoin(
        df_matrix, edges, acled, buffer_distance
    )

# Calculate the distance in meters for each incident to each route where it lies within the buffer
# distance
incidents_in_routes_list = []
    for (from_pcode, to_pcode), group_df in incidents_in_routes.set_index(
        ["from_pcode", "to_pcode"]
    ).groupby(level=[0, 1]):
        incidents_in_routes_list.append(
            calc.get_distances_to_route(group_df, df_matrix, edges)
        )
incidents_in_routes_df = pd.concat(incidents_in_routes_list)

# Aggregate incidents in route (Note that this is for ACLED data and the schema may differ with
# different tables)
df_grouped = (
    incidents_in_routes_df.groupby(["event_date", "from_pcode", "to_pcode"])
    .agg(
        incident_count=("event_id_cnty", "count"),
        total_fatalities=("fatalities", "sum"),
        mean_distance_to_route=("distance_to_route", "mean"),
    )
    .reset_index()
)

# Fill in rows (0 values) for routes that do not have incidents for specific days. This will only work
# for the ACLED schema. Please see source code (net_friction.data_preparation.fill_missing_routes) to
# see how your missing data can be filled in using different schemas 
df_grouped_filled = prep.fill_missing_routes(
    df_grouped, distances_df, date_start, date_end
)
```


## Calculate Areas of Control
Create a dataframe of pairwise combinations of source and destinations, indicating whether the route originates or ends with local occupation force (0) or foreign occupation force (1)

```python
import geopandas as gpd

from net_friction import areas_of_control_matrix as control

centroids_df = gpd.read_file(centroids)

areas_of_control_df = control.calculate_src_dst_areas_of_control(
    centroids_df,
    start_date=date_start,
    end_date=date_end,
    polygon_dir=control_areas_dir,
    crs=crs,
)
```





