from pathlib import Path
from net_friction.table_production import process_data_dask, process_data

if __name__ == "__main__":
    import time
    start = time.time()
    BASE = Path(r"C:\Users\dkerr\Documents\GISRede\OXFORD_UNI_WORK\NET_FRICTION_DEBUGGING\data").resolve()
    roads = BASE.joinpath("roads", "gis_osm_roads_free_1.shp")
    #roads = BASE.joinpath("roads", "test_roads.gpkg")
    crs = 6383
    raster = BASE.joinpath("ukr_ppp_2020_1km_Aggregated.tif")
    admin_boundaries = BASE.joinpath("GEODATA.gpkg")
    control_areas_dir = BASE.joinpath("control")
    acled_data = BASE.joinpath("2022-02-01-2024-02-21-Europe-Ukraine.csv")
    date_start = "2022-02-01"
    date_end = "2024-02-21"
    #for admin_level in [1, 2, 3]:
    for admin_level in [1, 2]:
        distance_matrix = BASE.joinpath(f"L{admin_level}", "distances.csv")
        incidents_in_routes = BASE.joinpath(f"L{admin_level}", "incidents_in_routes.csv")
        incidents_in_routes_aggregated = BASE.joinpath(f"L{admin_level}", "incidents_in_routes_aggregated.csv")
        areas_of_control_matrix = BASE.joinpath(f"L{admin_level}", "areas_of_control.csv")
        buffer_distance = 1000
        centroids_file = BASE.joinpath(f"L{admin_level}", f"centroids_L{admin_level}.gpkg")
        process_data_dask(
            roads,
            crs,
            raster,
            admin_boundaries,
            control_areas_dir,
            acled_data,
            date_start,
            date_end,
            distance_matrix,
            incidents_in_routes,
            incidents_in_routes_aggregated,
            areas_of_control_matrix,
            admin_level,
            buffer_distance,
            centroids_file,
        )
        end = time.time()
        print(f"Time taken: {(end - start) / 60} minutes for level {admin_level}")
