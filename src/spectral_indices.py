import geopandas as gpd

import rioxarray as rxr
from dea_tools.spatial import xr_rasterize

from distributed import LocalCluster, Client

import logging
import datetime
import pytz
from pathlib import Path
import argparse
import os
import json
import gc

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


def generate_json_files(output_dir="../jsons/sis"):
    os.makedirs(output_dir, exist_ok=True)
    
    start_date = datetime.datetime(2020, 7, 1)
    end_date = datetime.datetime(2023, 6, 1)
    
    current_date = start_date
    while current_date <= end_date:
        year_month = current_date.strftime("%Y%m")
        file_prefix = current_date.strftime("%Y%m")
        
        for i in range(1, 6):  # AoI1 to AoI5
            data = {
                "AOI_compo_path": f"../composites/AOI{i}/AOI{i}_1m_med_compo_{year_month}.tif"
            }
            
            file_name = f"{file_prefix}_AoI{i}.json"
            file_path = os.path.join(output_dir, file_name)
            
            with open(file_path, "w") as f:
                json.dump(data, f, indent=4)
            
            print(f"Saved: {file_path}")
        
        # Move to the next month
        current_date += datetime.timedelta(days=32)
        current_date = current_date.replace(day=1)  # Ensure we are at the first of the month


def compute_sis(AOI_compo_path: str):
    """_summary_

    Args:
        AOI_compo_path (str): _description_
    """
    # Set up logger.
    _log = setup_logger(logger_name='sisgen_',
                                logger_path=f'../logs/sisgen_{datetime.datetime.now(pytz.timezone("Europe/Athens")).strftime("%Y%m%dT%H%M%S")}.log', 
                                logger_format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
                                )
    
    _log.info(f'Processing {AOI_compo_path}')
    
    # Set up Dask Cluster for Parallelization
    # cluster = LocalCluster(n_workers=24, threads_per_worker=1, processes=False)
    # client = Client(cluster)
    # _log.info(client.dashboard_link)
    
    _log.info('Opening GeoTiff')
    try:
        ds = rxr.open_rasterio(AOI_compo_path) #, chunks=dict(x=1024, y=1024))
    except:
        AOI_compo_path = AOI_compo_path.replace('.tif', '_NOTREFINED.tif')
        _log.info(f'Opening GeoTiff: {AOI_compo_path}')
        ds = rxr.open_rasterio(AOI_compo_path) #, chunks=dict(x=1024, y=1024))
        
    AOI_compo_path = AOI_compo_path.replace('1m_med_compo', 'SIs_compo')
    AOI_compo_path = AOI_compo_path.replace('composites', 'indices')
    AOI = AOI_compo_path.split('/AOI')[1]
    
    mkdir(AOI_compo_path.split(f'/AOI{AOI}_')[0])
    
    if (os.path.isfile(AOI_compo_path)):
        msg = f"Indices have already been computed for composite and live in {AOI_compo_path} | Continuing."
        _log.warning(msg)
        return
    
    band_mapping = {
        1: 'B01', 2: 'B02', 3: 'B03', 4: 'B04',
        5: 'B05', 6: 'B06', 7: 'B07', 8: 'B08',
        9: 'B8A', 10: 'B09', 11: 'B11', 12: 'B12'
    }
    
    ds = ds.to_dataset(dim='band')
    ds = ds.rename(band_mapping)
    
    ds = ds.where(ds<=10000)
    
    _log.info('Computing indices')
    sis = ['NGRDI', 'NDWI', 'PSRI2', 'NDVI', 'NDVI705', 'NDII', 'EVI']
    ds['EVI'] = 2.5 * ((ds.B08 - ds.B04)/10000) / ((ds.B08/10000 + 6*ds.B04/10000 - 7.5*ds.B02/10000) + 1)
    ds['NGRDI'] = ((ds.B03 - ds.B04) / (ds.B03 + ds.B04)).astype('float32')
    ds['NDWI'] = ((ds.B08 - ds.B11) / (ds.B08 + ds.B11)).astype('float32')
    ds['PSRI2'] = ((ds.B05 - ds.B03) / ds.B07).astype('float32')
    ds['NDVI'] = ((ds.B08 - ds.B04) / (ds.B08 + ds.B04)).astype('float32')
    ds['NDVI705'] = ((ds.B06 - ds.B05) / (ds.B06 + ds.B05)).astype('float32')
    ds['NDII'] = ((ds.NDVI - ds.NDWI) / (ds.NDVI + ds.NDWI)).astype('float32')
    
    _log.info('Clip to typical value range')
    for si in sis:
        if si in ['NGRDI', 'NDWI', 'NDVI', 'NDVI705', 'NDII', 'EVI']:
            ds[si] = ds[si].where((ds[si]>=-1)&(ds[si]<=1))
        else:
            ds[si] = ds[si].where((ds[si]>=-1)&(ds[si]<=4))

    ds = ds[sis]
    
    _log.info('Rasterizing Carta Permeabilita 2019')
    carta_permeabilita_2019 = gpd.read_file('../ancilliary/carta_permeabilita_2019/Shapefile/carta_permeabilita_2019.shp')
    carta_permeabilita_2019 = carta_permeabilita_2019[carta_permeabilita_2019['Leg_TIPO'].notna()]
    carta_permeabilita_2019_da = xr_rasterize(gdf=carta_permeabilita_2019, da=ds, crs='EPSG:32632')
    
    _log.info('Masking dataset with Carta Permeabilita 2019')
    ds = ds.where(carta_permeabilita_2019_da) #.compute()
    ds = ds.odc.assign_crs(crs='EPSG:32632')
    
    _log.info('Writing spectral indices stacked Cloud Optimized GeoTiff')
    ds.rio.to_raster(AOI_compo_path,
                     driver="COG",
                     compress="deflate",  # optional, recommended
                     dtype="float32",     
                     tiled=True           # required for valid COG
                     )
    gc.collect()
    
    _log.info(f'Wrote it sucessfully -> {AOI_compo_path}')
    

if __name__ == "__main__":
    # Run the function to create json files
    generate_json_files()
    
    # args = get_sys_argv()
    # json_path = args['json_file']  # Accept either a single JSON file or a folder
    json_path = '../jsons/sis'
    
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

                AOI_compo_path = parameters_dict['AOI_compo_path']

                compute_sis(AOI_compo_path)
                
                print(f"Processed {json_file}")
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    print(f"Competed")