import itertools
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

pd.set_option("future.no_silent_downcasting", True)


def calculate_src_dst_areas_of_control(
    centroids_df: gpd.GeoDataFrame,
    start_date: str,
    end_date: str,
    polygon_dir: Path | str,
    crs: int,
) -> pd.DataFrame:
    """Calculate source and destination areas of control for each pairwise combination in the centoids_df.

    Args:
        centroids_df (gpd.GeoDataFrame): Boundary centroids (source and destination points)
        start_date (str): Start date
        end_date (str): End date
        polygon_dir (Path | str): Location of the areas of control polygons saved as geopackage files saved in the
            format `occupied_{date}.gpkg`
        crs (int): CRS

    Returns:
        pd.DataFrame: Dataframe indicating source and destination occupied (1) or unoccupied (0) as indicated by areas
        of control polygons
    """
    dates = pd.date_range(start_date, end_date, freq="D").strftime("%Y-%m-%d").tolist()
    areas_of_control_df_list = []
    for index, day in enumerate(dates):
        gpkg_path = Path(polygon_dir).joinpath(f"occupied_{day}.gpkg")
        if gpkg_path.exists():
            polygon = gpd.read_file(
                Path(polygon_dir).joinpath(f"occupied_{day}.gpkg")
            ).loc[0, "geometry"]
            centroids_df["in_occupied_area"] = centroids_df.within(polygon)
        elif not gpkg_path.exists() and index == 0:
            centroids_df["in_occupied_area"] = False
        else:
            centroids_df["in_occupied_area"] = np.nan
        occupied_area_dict = centroids_df.set_index("pcode")[
            "in_occupied_area"
        ].to_dict()

        combinations = pd.DataFrame(
            list(itertools.product(centroids_df["pcode"], repeat=2)),
            columns=["from_pcode", "to_pcode"],
        )
        combinations["src_occupied"] = combinations["from_pcode"].map(
            occupied_area_dict
        )
        combinations["dst_occupied"] = combinations["to_pcode"].map(occupied_area_dict)
        combinations["date"] = day
        areas_of_control_df_list.append(combinations)
    areas_of_control_df = pd.concat(areas_of_control_df_list)
    areas_of_control_df = areas_of_control_df.ffill()
    return areas_of_control_df
