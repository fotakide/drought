'''
######################################################################
## ARISTOTLE UNIVERSITY OF THESSALONIKI
## PERSLAB
## REMOTE SENSING AND EARTH OBSERVATION TEAM
##
## DATE:             Aug-2025
## SCRIPT:           utils/utils.py
## AUTHOR:           Vangelis Fotakidis (fotakidis@topo.auth.gr)
##
## DESCRIPTION:      Utility module with general functions of features required in the pipelines
##
#######################################################################
'''

import os, json, datetime, argparse
import logging
from pathlib import Path
import geopandas as gpd
from shapely.geometry import mapping
from dateutil.relativedelta import relativedelta


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
        logging.basicConfig(
            filename=logger_path, 
            level=logging.INFO, 
            format=logger_format,
            encoding="utf-8"
            )

        # Create a file handler
        handler = logging.FileHandler(logger_path, encoding="utf-8", errors="strict")
        handler.setFormatter(logging.Formatter(logger_format))

        # Add the file handler to the logger
        logger.addHandler(handler)

        # Set propagate to False in order to avoid double entries
        logger.propagate = False

    return logger


def generate_geojson_files_for_composites(
    output_dir="../geojsons/compgen",
    tile_geojson_filepath="../anciliary/grid_v2.geojson",
    start_date=datetime.datetime(2020, 1, 1),
    end_date=datetime.datetime(2025, 9, 1),
):
    os.makedirs(output_dir, exist_ok=True)

    aoi = gpd.read_file(tile_geojson_filepath).to_crs("EPSG:4326")

    current_date = start_date
    while current_date <= end_date:
        year_month = current_date.strftime("%Y-%m")
        date_prefix = current_date.strftime("%Y%m")

        for _, tile in aoi.iterrows():
            tile_id = tile["tile_ids"]

            feature = {
                "type": "Feature",
                "properties": {
                    "year_month": year_month,
                    "tile_id": tile_id,
                },
                "geometry": mapping(tile.geometry),  # GeoJSON geometry
            }

            file_prefix = f"{date_prefix}_{tile_id}"
            file_name = f"compgen_{file_prefix}.geojson"
            file_path = os.path.join(output_dir, file_name)

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(feature, f, indent=2, ensure_ascii=False)

        current_date += relativedelta(months=1)


# def generate_json_files_for_composites(
#     output_dir="../jsons/compgen",
#     tile_geojson_filepath='../anciliary/grid_v2.geojson',
#     start_date: datetime.datetime=datetime.datetime(2020, 1, 1),
#     end_date: datetime.datetime=datetime.datetime(2025, 9, 1),
#     ):
    
#     os.makedirs(output_dir, exist_ok=True)
    
#     current_date = start_date
#     while current_date <= end_date:
#         year_month = current_date.strftime("%Y-%m")
#         file_prefix = current_date.strftime("%Y%m")
        
#         data = {
#             "year_month": year_month,
#             "tilegrid_path": tile_geojson_filepath
#         }
            
#         file_name = f"compgen_{file_prefix}.json"
#         file_path = os.path.join(output_dir, file_name)
        
#         with open(file_path, "w") as f:
#             json.dump(data, f, indent=4)
        
#         # print(f"Saved: {file_path}")
        
#         # Move to the next month
#         current_date += datetime.timedelta(days=32)
#         current_date = current_date.replace(day=1)  # Ensure we are at the first of the month
