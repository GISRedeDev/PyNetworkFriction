import itertools
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

pd.set_option("future.no_silent_downcasting", True)


def calculate_src_dst_areas_of_control(
    src_dst_matrix: pd.DataFrame,
    start_date: str,
    end_date: str,
    polygon_dir: Path | str,
    crs: int,
) -> pd.DataFrame:
    areas_of_control_df = src_dst_matrix.copy()
    centroids_df = gpd.GeoDataFrame(
        src_dst_matrix[["from_pcode", "from_centroid"]],
        geometry="from_centroid",
        crs=crs,
    ).to_crs(4326)
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
        combinations = pd.DataFrame(
            list(itertools.product(centroids_df["from_pcode"], repeat=2)),
            columns=["from_pcode", "to_pcode"],
        )
        combinations["src_occupied"] = combinations["from_pcode"].map(
            centroids_df.set_index("from_pcode")["in_occupied_area"]
        )
        combinations["dst_occupied"] = combinations["to_pcode"].map(
            centroids_df.set_index("from_pcode")["in_occupied_area"]
        )
        combinations["date"] = day
        areas_of_control_df_list.append(combinations)
    areas_of_control_df = pd.concat(areas_of_control_df_list)
    areas_of_control_df = areas_of_control_df.ffill()
    return areas_of_control_df
