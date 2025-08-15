import os
from pathlib import Path
import datetime
import json
import logging
import argparse


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


def generate_json_files_for_composites(output_dir="../jsons"):
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
