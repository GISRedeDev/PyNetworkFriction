# Network Friction

`net-friction` was funded by the University of Oxford (insert dept/url) and was used in their population modelling work in Ukraine (url to project and dashboard).

## Description and suggested workflow
The purpose of the `net-friction` package is to provide functionality to analyse input point or raster 'incident' data relative to their proximity of network routes, and more specifically, their proximity to pairwise route combinations in a network. The functions in the package provide output for the following suggested workflow:

1. Subset an input road network dataset (OSM) to required classifications and carry out a crude topology to ensure all edges in the network are connected;
 - See `data_pre_processing` to preprocess and save network edges and source/destination centroids or
 - See `get_roads_data` and `fix_topology`
2. Calculate source/destination point dataset either as centroids of an input boundary dataset, or as weighted centroids from a boundary dataset and an input continuous raster dataset;
 - See `get_source_desitnation_points`
3. Calculate shortest routes from the road network between all pairwise combinations of the source/destination dataset;
4. From these networks, calculate the straight line and network route distances in the source/destination matrix;
5. Extract incident datasets (raster or point, i.e. ACLED) within an input proximity of the routes and aggregate the incidents along the respective pairwise routes;
6. Optionally infer source and destination areas of control in conflict scenarios using pre-defined polygons, for example [Ukraine/Russian areas of control](https://github.com/GISRedeDev/AreasofControl).

## Installation (requires Python >= 3.11)
To install net-friction, use pip:
- `pip install git+https://github.com/GISRedeDev/PyNetworkFriction.git`


## Contributing
Contributions are welcome. Please submit a pull request or create an issue to discuss the changes you want to make.

## License
This project is licensed under the MIT License.






