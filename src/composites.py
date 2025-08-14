import datacube

import pystac_client
import pystac
import planetary_computer
from pystac_client.stac_api_io import StacApiIO
from urllib3 import Retry

import geopandas as gpd
from odc.geo.geom import BoundingBox
import odc.geo.xr

import pandas as pd
import numpy as np
import rioxarray as rxr
import xarray as xr
from odc.algo import mask_cleanup
from dea_tools.spatial import xr_rasterize

from skimage.measure import block_reduce
from numpy import mean, uint16
import xarray as xr


import requests
import xml.etree.ElementTree as ET
from collections import defaultdict

import odc.stac
from odc.stac import configure_rio
from datacube.utils.aws import configure_s3_access
from distributed import LocalCluster, Client

import logging
import datetime
import pytz
from pathlib import Path
import argparse
import os
import json
from typing import List, Tuple

import warnings
warnings.filterwarnings("ignore")


def mkdir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def get_sys_argv():
    parser = argparse.ArgumentParser(description="Parse required arguments for the analysis",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-j", "--json-file", help="Point to json file that contains required parameters", required=True)

    args = parser.parse_args()
    config = vars(args)
    return config


def setup_logger(logger_name, logger_path, logger_format):
    logger = logging.getLogger(logger_name)

    if not logger.handlers:  # Check if the logger has no handlers yet
        # Configure the root logger
        logging.basicConfig(filename=logger_path, level=logging.INFO, format=logger_format)

        # Create a file handler
        handler = logging.FileHandler(logger_path)
        handler.setFormatter(logging.Formatter(logger_format))

        # Add the file handler to the logger
        logger.addHandler(handler)

        # Set propagate to False in order to avoid double entries
        logger.propagate = False

    return logger


def generate_json_files(output_dir="../jsons"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Set start and end
    start_date = datetime.datetime(2020, 7, 1)
    end_date = datetime.datetime(2023, 6, 1)
    
    current_date = start_date
    while current_date <= end_date:
        year_month = current_date.strftime("%Y-%m")
        file_prefix = current_date.strftime("%Y%m")
        
        for i in range(1, 6):  # AoI1 to AoI5
            data = {
                "year_month": year_month,
                "AOI_path": f"../studyarea/AoI{i}.kml"
            }
            
            file_name = f"{file_prefix}_AoI{i}.json"
            file_path = os.path.join(output_dir, file_name)
            
            with open(file_path, "w") as f:
                json.dump(data, f, indent=4)
            
            print(f"Saved: {file_path}")
        
        # Move to the next month
        current_date += datetime.timedelta(days=32)
        current_date = current_date.replace(day=1)  # Ensure we are at the first of the month


def check_gri_refinement(items: List[pystac.Item]) -> Tuple[List[pystac.Item], pd.DataFrame]:
    """ Function to search whether the provided scene is refined via GRI or not.
    Regarding mis-registration (as observed in 2023 vs 2024):
    See: https://forum.step.esa.int/t/geometric-gri-refinement-in-sentinel-2-level-1c-early-images-below-pb-3-0/44024/2
    Finally, the activation of the geometric refining does not mean that the products will be always refined. 
    There are some cases (e.g. too many clouds) where the refining cannot be applied as it would not improve 
    and could degrade the geolocation of the products. This can be checked thanks to the metadata 
    Image_Data_Info/Geometric_Info/Image_Refining in the datastrip matadata file (DATASTRIP/*/MTD_DS.xml). In STAC: datastrip_metadata
    This parameter is equal to REFINED or NOT_REFINED.
    <Geometric_Info metadataLevel="Standard">
       <RGM>COMPUTED</RGM>
       <Image_Refining flag="REFINED">

    Args:
        items (List[pystac.Item]):List of pystac.item.Item from pystac_client.item_search.ItemSearch

    Returns:
        Tuple:
            - List[pystac.Item], List of pystac.Item objects with REFINED status and
            - pd.DataFrame, DataFrame with refinement status for each item
    """
    refined_items = []
    refinement_data = []

    # Loop through items and extract `Image_Refining` flag
    for item in items:
        datastrip_metadata_url = None
        for asset_key, asset_data in items[0].assets.items():
            if "datastrip-metadata" in asset_key.lower(): # and asset_data.href.endswith(".xml"):
                # datastrip_metadata_url = planetary_computer.sign(asset_data.href)
                datastrip_metadata_url = asset_data.href
                break

        if datastrip_metadata_url:
            # log.info(f"Processing: {datastrip_metadata_url}")
            # logging.info(f"Processing: {item.id}") #log.info(f"Processing: {datastrip_metadata_url}")

            # Fetch XML content
            xml_response = requests.get(datastrip_metadata_url)
            if xml_response.status_code == 200:
                root = ET.fromstring(xml_response.content)

                # Extract Image_Refining flag
                refining_element = root.find(".//Geometric_Info/Image_Refining")
                refining_flag = refining_element.get("flag") if refining_element is not None else "Not Found"

                logging.info(f"{item.id} -> {refining_flag}")

                # Store item and status in dataframe
                refinement_data.append({
                    "item_id": item.id,
                    "refinement_status": refining_flag
                })

                # Append to refined_items if the flag is 'REFINED'
                if refining_flag == "REFINED":
                    refined_items.append(item)
            else:
                # log.info(f"Failed to fetch XML: {xml_response.status_code}")
                refinement_data.append({
                    "item_id": item.id,
                    "refinement_status": "Fetch Failed"
                })
        else:
            # log.info("datastrip_metadata.xml not found in STAC item")
            refinement_data.append({
                "item_id": item.id,
                "refinement_status": "Metadata Not Found"
            })

    # Create a DataFrame
    df_refinement_status = pd.DataFrame(refinement_data)

    return refined_items, df_refinement_status




def generate_composite(year_month: str, tile: pd.Series):
    # Set up logger.
    log = setup_logger(logger_name='compgen_',
                                logger_path=f'../logs/compgen_{datetime.datetime.now(pytz.timezone("Europe/Athens")).strftime("%Y%m%dT%H%M%S")}.log', 
                                logger_format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
                                )
    
    tile_id = tile.tiles_id
    log.info('#######################################################################')
    log.info('Processing started')
    log.info(f'        Tile: {tile_id}')
    log.info(f'        Time: {year_month}')
    
    
    log.info('                          ')
    log.info('Establishing connection to datacube')
    dc = datacube.Datatcube(app='Composite generation', env='drought')
    
    log.info('                          ')
    log.info('Check if dataset already exists in the datacube')
    find_ds_in_sc = dc.find_datasets(
        **dict(
            product='composites',
            time=year_month,
            region_code=tile_id
        ),
        ensure_location=True
    )
        
    if find_ds_in_sc:
        msg = f"This composite already exists in {find_ds_in_sc[0].uri} | Continuing."
        log.warning(msg)
        return
    else:
        log.info("The composite requested will be computed")
    
    
    log.info('                          ')
    log.info('Retrieve tile geometry')
    minx, miny, maxx, maxy = tile.geometry.bounds
    log.info('Create the Bounding Box')
    aoi_bbox = BoundingBox.from_xy(
        (minx, maxx),
        (miny, maxy)
    )

    
    log.info('                          ')
    log.info('Connect to Planetary Computer STAC Catalog')
    stac_api_io = StacApiIO(max_retries=Retry(total=5, backoff_factor=5))
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
        stac_io=stac_api_io
    )
    
    log.info('Search the STAC Catalog')
    cloud_cover = 70
    search = catalog.search(
        collections=["sentinel-2-l2a"], #hls2-s30
        bbox=aoi_bbox,
        datetime='2024-07',
        limit=100,
        query={
            "eo:cloud_cover": {"lt":cloud_cover},
            "s2:nodata_pixel_percentage": {"lt":20},
        },
    )
    log.info('        Query parameters:')
    log.info(f'            url:          {search.url}')
    log.info(f'            client:       {search.client.id}')
    log.info(f'            collection:   {search._parameters['collections']}')
    log.info(f'            bbox:         {search._parameters['bbox']}')
    log.info(f'            time range:   {search._parameters['datetime']}')
    log.info(f'            cloud cover:  0% - {search._parameters['query']['eo:cloud_cover']['lt']}%')
    log.info(f'            nodata cover: 0% - {search._parameters['query']['s2:nodata_pixel_percentage']['lt']}%')
    
    log.info('                          ')
    log.info('Searching.................')
    items = search.item_collection()
    
    log.info(f'Query found {len(items)} items')
    
    log.info('Searching for GRI REFINED scenes:')
    refined_items, df_refinement_status = check_gri_refinement(items)
    
    log.info('                                 ')
    log.info(f'{len(refined_items)}/{len(df_refinement_status)} were refined by GRI.')
    
    refineflag = None
    if len(refined_items) == 0:
        msg = f"Tile {tile_id} | Time: {year_month}: All scenes are flagged NOT REFINED."
        log.warning(msg)
        log.warning(f"Tile {tile_id} | Time: {year_month}: The composite will be flagged with _NOREFINED.")
        refineflag = 'NOTREFINED'
        refined_items = items

    log.info('Looking for up to 18 cleanest images within time range')
    filtered_items = []
    mgrs_tiles = np.unique([i.properties['s2:mgrs_tile'] for i in refined_items])
    epsgs = np.unique([i.properties['proj:epsg'] for i in refined_items])
    
    for mgrstile in mgrs_tiles:
        item_mgrs_sorted = sorted([
            i for i in refined_items if i.properties['s2:mgrs_tile'] == str(mgrstile)
            ], key=lambda item: item.properties['eo:cloud_cover'])
        
        if len(item_mgrs_sorted) > 18:
            filtered_items.extend(item_mgrs_sorted[:18])
        else:
            filtered_items.extend(item_mgrs_sorted)
    log.info(f"Filtered cleanest scenes: Kept {len(filtered_items)} out of {len(refined_items)} items. ")
    log.info(f"        Included MGRS Tiles: {mgrs_tiles}")
    log.info(f"        Included EPSG codes: {epsgs}")

    log.info('Selected scenes:')
    for stacitem in filtered_items:
        log.info(f'        {stacitem.id}')
    
    log.info('                                 ')
    log.info('Initialize Dask cluster for parallelization')
    cluster = LocalCluster(
        n_workers=16, 
        threads_per_worker=1, 
        processes=False,
        # memory_limit='4GB', 
        # local_directory="/tmp/dask-worker-space",
        )
    client = Client(cluster)
    configure_rio(cloud_defaults=True, client=client) # For Planetary Computer
    log.info(f'The Dask client listens to {client.dashboard_link}')
    


    log.info(f'                                 ')
    log.info(f'Downstream STAC items from Planetary Computer')
    BANDS_R10m = ['B02', 'B03', 'B04', 'B08']
    BANDS_R20m = ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12', 'SCL']
    

    # Loading for each EPSG and Native RESOLUTION separatelly, then merging into a single xr.Dataset
    # to to a correct downsampling
    for EPSG in epsgs:
        for RESOLUTION in [10, 20]:
            if RESOLUTION==10:
                BANDS=BANDS_R10m
            else:
                BANDS=BANDS_R20m
            
            log.info('    Loading parameters:')
            log.info(f'        Bands: {BANDS}')
            log.info(f'        Spatial resolution: {RESOLUTION}')

            
            ds_cube = odc.stac.stac_load(
                filtered_items,
                bbox=aoi_bbox,
                bands=BANDS,
                chunks=dict(y=2048, x=2048),
                crs=f'EPSG:{EPSG}',  # {epsgs[0]}
                resolution=RESOLUTION,
                groupby='time', # if 'time' loads all items, retaining duplicates
                fail_on_error=True,
                # resampling={
                #     "*": RESAMPLING_ALGO,
                # },
            )
            
            if RESOLUTION==10:
                

            

            log.info(f'        Resampling algorithm: {RESAMPLING_ALGO}')                
            
    
    
    
    BANDS = ['B02', 'B03', 'B04', 'B05', 'B07', 'B08', 'B8A', 'SCL']
    EPSG = 'EPSG:3035'
    RESOLUTION = 20
    RESAMPLING_ALGO = "average"
    log.info('    Loading parameters:')
    log.info(f'        Bands: {BANDS}')
    log.info(f'        Spatial resolution: {RESOLUTION}')
    log.info(f'        Resampling algorithm: {RESAMPLING_ALGO}')
    
    ds_cube = odc.stac.stac_load(
        filtered_items,
        bbox=aoi_bbox,
        bands=BANDS,
        chunks=dict(y=2048, x=2048),
        crs=EPSG,  # {epsgs[0]}
        resolution=RESOLUTION,
        groupby='time', # if 'time' loads all items, retaining duplicates
        fail_on_error=True,
        resampling={
            "*": RESAMPLING_ALGO,
        },
    )
    
    log.info(f'Apply masks on clouds, shadows, thin cirrus, and snow/ice')
    # - 0: no data
    # - 1: saturated or defective
    # - 2: dark area pixels
    # - 3: cloud shadows
    # - 4: vegetation
    # - 5: bare soils
    # - 6: water
    # - 7: unclassified
    # - 8: cloud medium probability
    # - 9: cloud high probability
    # - 10: thin cirrus
    # - 11: snow or ice
    invalid_scl_values = [3, 7, 8, 9, 10, 11]
    log.info(f'Masking bits: {invalid_scl_values}')
    cloud_binary_mask = ds_cube.SCL.isin(invalid_scl_values)

    BANDS.remove('SCL')    
    ds_cube_cf = ds_cube[BANDS].where(~cloud_binary_mask, 0).astype('uint16')
    
    ds_cube_cf = ds_cube_cf.where(ds_cube_cf > 0, np.nan)
    try:
        log.info(f'Downloading bands and computing median composites: Tile {tile_id} | Time: {year_month}')
        median_composite = ds_cube_cf[BANDS].median(dim='time').astype('float32')
        median_composite = median_composite.compute()
        
        log.info('Convert to unsigned integer 16-bits')
        median_composite = median_composite.where(~median_composite.isnull(), 0).astype('uint16')

        for var in list(median_composite.data_vars):
            median_composite[var].attrs['nodata'] = 0

        # Convert to a multi-band DataArray
        da = xr.concat([median_composite[var] for var in median_composite.data_vars], dim="band")
        da = da.assign_coords(band=("band", list(range(1, len(median_composite.data_vars) + 1))))
        da.name = ''
        
        FOLDER_NAME = f"../composites/AOI{AOI_number}"
        if refineflag:
            DATASET_NAME = f"AOI{AOI_number}_1m_med_compo_{year_month.split('-')[0]}{year_month.split('-')[1]}_{refineflag}"
        else:
            DATASET_NAME = f"AOI{AOI_number}_1m_med_compo_{year_month.split('-')[0]}{year_month.split('-')[1]}"
        mkdir(FOLDER_NAME)
        
        log.info(f'Write composite to GeoTIFF -> {FOLDER_NAME}/{DATASET_NAME}.tif')
        da.rio.to_raster(f"{FOLDER_NAME}/{DATASET_NAME}.tif", 
                        driver="GTiff", 
                        compress="lzw",
                        nodata=0,
                        )
        
        log.info('Closing Dask client.')
        client.close()
        cluster.close()
    except Exception as exc:
        msg=f'Failed loading for : Tile {tile_id} | Time: {year_month}'
        log.error(msg)
        client.close()
        cluster.close()
        return


if __name__ == "__main__":
    # Run the function to create json files
    json_path = '../jsons'
    
    generate_json_files(output_dir=json_path)      
    
    # Check if the path is a folder or a file
    if os.path.isdir(json_path):
        # List all JSON files in the directory
        json_files = [os.path.join(json_path, f) for f in os.listdir(json_path) if f.endswith('.json')]
    else:
        # Single file case
        json_files = [json_path]
    
    # Loop through and process each JSON file
    for json_file in json_files:
        try:
            with open(json_file) as f:
                parameters_dict = json.load(f)

                year_month = parameters_dict['year_month']
                AOI_path = parameters_dict['AOI_path'] #'../anciliary/grid_v2.geojson'
                
                aoi = gpd.read_file(AOI_path).to_crs('EPSG:4326')

                for i, tile in aoi.iterrows():
                    generate_composite(year_month=year_month, tile=tile)
                
                print(f"Processed {json_file}")
        except Exception as e:
            print(f"Error processing {json_file}: {e}")