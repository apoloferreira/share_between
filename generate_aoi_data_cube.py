import os
import pickle
import sys
import subprocess
import warnings
import json
import time
import geopandas
import pandas as pd
import numpy as np
import geopandas as gpd
import shapely
from shapely.geometry import shape
from shapely.geometry import Polygon
import xarray as xr
import rioxarray
from rioxarray.exceptions import NoDataInBounds
import gc
import logging
import datetime

# Parameters
MAX_CLOUD_COVER_AOI = 0.3 # maximum 30% cloud coverage WTHIN area of interest (not the full tile)


# HELPERS
def get_logger(log_level):
    logger = logging.getLogger("processing")

    console_handler = logging.StreamHandler(sys.stdout)
    # include %(name)s to also include logger name
    console_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    console_handler.setLevel(log_level)

    logger.addHandler(console_handler)
    logger.setLevel(log_level)
    return logger


def s2_scene_id_to_cog_path(scene_id):
    parts = scene_id.split("_")
    s2_qualifier = "{}/{}/{}/{}/{}/{}".format(
        parts[1][0:2],
        parts[1][2],
        parts[1][3:5],
        parts[2][0:4],
        str(int(parts[2][4:6])),
        "_".join(parts)
    )
    return f"https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/{s2_qualifier}/"


def scene_id_to_datetime(scene_id):
    dt = pd.to_datetime(scene_id.split("_")[-3])
    return dt


def get_aoi_cloud_free_ratio(SCL_raster, aoi_gdf):
    # reproject to EPSG:4326
    kwargs = {"nodata": np.nan}
    SCL_raster = SCL_raster.rio.reproject("EPSG:4326", **kwargs)
    # clip to AOI
    SCL_raster_clipped = SCL.rio.clip(aoi_gdf.geometry.values, aoi_gdf.crs, drop=False, invert=True)
    # get cloud-free ratio
    SCL_mask_pixel_count = SCL_raster_clipped.SCL.data.size - np.count_nonzero(np.isnan(SCL_raster_clipped.SCL.data)) # get size of SCL mask in num pixels (excl. any nans)
    SCL_classes_cloud_free = [4,5,6] # see here: https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/scene-classification/
    SCL_cloud_free_pixel_count = np.isin(SCL_raster_clipped.SCL.data,SCL_classes_cloud_free).sum() #count pixels that are non-cloud class
    cloud_free_ratio = SCL_cloud_free_pixel_count/SCL_mask_pixel_count
    return cloud_free_ratio


# MAIN
if __name__ == "__main__":
    logger = get_logger(logging.DEBUG) #INFO

    logger.info("Starting processing")
    logger.debug(f"Argument List: {str(sys.argv)}")

    # set permissions on output path
    output_path = "/opt/ml/processing/output/"
    subprocess.check_call(["sudo", "chown", "-R", "sagemaker-user", output_path])

    # load geometry and construct gdf
    aoi_path = '/opt/ml/processing/input/aoi_meta/aoi_metadata.json'
    with open(file=aoi_path, encoding="utf-8") as file:
        aoi_metadata = json.load(file)

    polgyon = Polygon(aoi_metadata["coords"])
    aoi_gdf = gpd.GeoDataFrame(index=[0], crs=aoi_metadata["crs"], geometry=[polgyon])

    # load all the sentinel-2 metadata files
    s2_data_path = '/opt/ml/processing/input/sentinel2_meta/'
    s2_items = []
    for current_path, sub_dirs, files in os.walk(s2_data_path):
        for file in files:
            if file.endswith(".json"):
                full_file_path = os.path.join(s2_data_path, current_path, file)
                with open(file=full_file_path, mode='r', encoding='utf-8') as f:
                    s2_items.append(json.load(f))

    item_count_total = len(s2_items)
    item_count_current = 0
    elapsed_time_batch = 0
    logger.info("Received {} scenes to process".format(item_count_total))

    for item in s2_items:
        if item_count_current > 0 and item_count_current % 5 == 0:
            logger.info("Processed {}/{} scenes ({}s per scene)".format(
                item_count_current,
                item_count_total,
                round(elapsed_time_batch / item_count_current, 2)
            ))
        item_count_current += 1

        start = time.time()
        s2_scene_id = item["id"]
        logger.debug(f"Processing scene: {s2_scene_id}")

        s2_cog_prefix = s2_scene_id_to_cog_path(s2_scene_id)
        grid_id = s2_scene_id.split("_")[1]
        # time/date
        date = scene_id_to_datetime(s2_scene_id)
        # 10m bands
        blue_band_url = f"{s2_cog_prefix}/B02.tif"
        green_band_url = f"{s2_cog_prefix}/B03.tif"
        red_band_url = f"{s2_cog_prefix}/B04.tif"
        nir1_band_url = f"{s2_cog_prefix}/B08.tif"
        # 20m bands
        nir2_band_url = f"{s2_cog_prefix}/B8A.tif"
        swir1_band_url = f"{s2_cog_prefix}/B11.tif"
        swir2_band_url = f"{s2_cog_prefix}/B12.tif"
        scl_mask_url = f"{s2_cog_prefix}/SCL.tif"

        # read from S3
        # 10m bands
        B02 = rioxarray.open_rasterio(blue_band_url, masked=True, band_as_variable=True)
        B02 = B02.rename(name_dict={"band_1":"B02"})
        B03 = rioxarray.open_rasterio(green_band_url, masked=True, band_as_variable=True)
        B03 = B03.rename(name_dict={"band_1":"B03"})
        B04 = rioxarray.open_rasterio(red_band_url, masked=True, band_as_variable=True)
        B04 = B04.rename(name_dict={"band_1":"B04"})
        B08 = rioxarray.open_rasterio(nir1_band_url, masked=True, band_as_variable=True)
        B08 = B08.rename(name_dict={"band_1":"B08"})
        # 20m bands/masks
        B8A = rioxarray.open_rasterio(nir2_band_url, masked=True, band_as_variable=True)
        B8A = B8A.rename(name_dict={"band_1":"B8A"})
        B11 = rioxarray.open_rasterio(swir1_band_url, masked=True, band_as_variable=True)
        B11 = B11.rename(name_dict={"band_1":"B11"})
        B12 = rioxarray.open_rasterio(swir2_band_url, masked=True, band_as_variable=True)
        B12 = B12.rename(name_dict={"band_1":"B12"})
        SCL = rioxarray.open_rasterio(scl_mask_url, masked=True, band_as_variable=True)
        SCL = SCL.rename(name_dict={"band_1":"SCL"})

        # resample to 10m where needed (relatively compute intensive!)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            B8A = B8A.interp(x=B02["x"], y=B02["y"])
            B11 = B11.interp(x=B02["x"], y=B02["y"])
            B12 = B12.interp(x=B02["x"], y=B02["y"])
            SCL = SCL.interp(x=B02["x"], y=B02["y"])

        # merge bands
        band_arrays = [B02, B03, B04, B08, B8A, B11, B12, SCL]
        s2_cube = xr.merge(objects=band_arrays)
        del B02
        del B03
        del B04
        del B08
        del B8A
        del B11
        del B12
        del band_arrays
        gc.collect()

        # assign time dimension
        s2_cube = s2_cube.assign_coords(time=date) #call this 'time'
        # reproject to EPSG:4326
        kwargs = {"nodata": np.nan}
        s2_cube = s2_cube.rio.reproject("EPSG:4326", **kwargs)

        # check cloud-free ratio at aoi level
        cloud_free_ratio = get_aoi_cloud_free_ratio(SCL_raster=SCL, aoi_gdf=aoi_gdf)
        if (1-float(cloud_free_ratio)) > MAX_CLOUD_COVER_AOI: #skip if too cloudy for given geom
            logger.debug(f"AOI cloud cover ratio too high ({round(1-cloud_free_ratio,3)}), skipping scene {s2_scene_id}...")
            del cloud_free_ratio
        else: #continue if not too cloudy for given geom
            logger.debug(
                f"AOI cloud cover ratio below threshold ({round(1-cloud_free_ratio,3)}), processing scene {s2_scene_id}...")
            try:
                clipped = s2_cube.rio.clip(aoi_gdf.geometry.values, aoi_gdf.crs)
            except NoDataInBounds as e:
                logger.warning("Skipping {}: no data in bounds".format(s2_scene_id))
                continue

            # save to file
            file_name = "{}-{}.nc".format(aoi_metadata["name"],s2_scene_id)
            output_file_path = f"{output_path}{file_name}"

            clipped.to_netcdf(output_file_path)

            logger.debug(f"Written output:{output_file_path}")

            del clipped
            del cloud_free_ratio
            gc.collect()

        # explicit dereference to keep memory usage low
        del s2_cube
        del SCL
        gc.collect()

        elapsed_time = time.time() - start
        elapsed_time_batch += elapsed_time

        logger.debug("Processed scene {}: {}s".format(s2_scene_id, elapsed_time))
