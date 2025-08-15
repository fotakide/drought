import datacube
from datacube.index.hl import Doc2Dataset
from eodatasets3 import serialise

import pandas as pd
import numpy as np

import xarray as xr
import rioxarray as rxr
from rioxarray.merge import merge_datasets
import odc.geo.xr

from odc.algo import mask_cleanup
from dea_tools.spatial import xr_rasterize

import geopandas as gpd
from odc.geo.geom import BoundingBox

import pystac_client
import odc.stac
from odc.stac import configure_rio
from datacube.utils.aws import configure_s3_access
import planetary_computer
from pystac_client.stac_api_io import StacApiIO
from urllib3 import Retry

from distributed import LocalCluster, Client

import os
import json
import datetime
import pytz
from pathlib import Path

from utils.downsample import s2_downsample_dataset_10m_to_20m
from utils.metadata import prepare_eo3_metadata_NAS
from utils.sentinel2 import check_gri_refinement, mask_with_scl
from utils.utils import mkdir, get_sys_argv, setup_logger

import warnings
warnings.filterwarnings("ignore")


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
    BANDS_R10m = ['B02', 'B03', 'B04']
    BANDS_R20m = ['B05', 'B07', 'B8A', 'SCL']


    # Loading for each EPSG and Native RESOLUTION separatelly, then merging into a single xr.Dataset
    # to to a correct downsampling
    try:
        processed_epsgs = []
        for EPSG in epsgs:
            processed_bands = []
            geobox = None
            log.info('Loading bands of diferent resolutions')
            for RESOLUTION in [20, 10]:
                if RESOLUTION==10:
                    BANDS=BANDS_R10m
                else:
                    BANDS=BANDS_R20m
                
                log.info('                       ')
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
                ).compute()
                
                if RESOLUTION==10:
                    log.info('        Downsample 10m bands to 20m by average 2x2 binning')
                    ds_cube = s2_downsample_dataset_10m_to_20m(ds_cube)
                    log.info(f'        Warp (reproject) 2x2 binned bands like native 20m bands: method={RESAMPLING_ALGO}')
                    ds_bands = ds_bands.odc.reproject(how=geobox, resampling=RESAMPLING_ALGO)
                elif RESOLUTION==20:
                    log.info('        Fix order of dimensions')
                    ds_bands = ds_bands[['time','y','x']+list(ds_bands.data_vars)]
                    geobox = ds_bands.odc.geobox
                
                log.info(f'        Append bands to the band-list of EPSG:{EPSG}')
                processed_bands.append(ds_bands)
            
            log.info('Merging bands in a single dataset')
            log.info('    Ensure all partial datasets are at 20m resolution')
            for dataset in processed_bands:
                dataset.odc.resolution == 20
            
            log.info('    Clip to the intersection of indexes')
            xmin = xmax = ymin = ymax = None
            for da in processed_bands:
                xmin = da.x.min().item() if xmin is None or da.x.min() > xmin else xmin
                xmax = da.x.max().item() if xmax is None or da.x.max() < xmax else xmax
                ymin = da.y.min().item() if ymin is None or da.y.min() > ymin else ymin
                ymax = da.y.max().item() if ymax is None or da.y.max() < ymax else ymax

            merge_bands = []
            log.info('    Make a list of bands to merge')
            for da in processed_bands:
                merge_bands.append(da.sel(x=slice(xmin, xmax), y=slice(ymax,ymin)))
            
            log.info('    Merge on intersection')
            ds_epsg = xr.merge(
                merge_bands,
                compat="no_conflicts",
                combine_attrs="drop_conflicts",
                join="inner"
            )
            
            log.info(f'    Apply masks on clouds, shadows, thin cirrus, and snow/ice')
            BANDS = ['B02', 'B03', 'B04', 'B05', 'B07', 'B08', 'B8A', 'SCL']
            ds_epsg_masked = mask_with_scl(ds_epsg, BANDS)
            
            RESAMPLING_ALGO = "bilinear"
            EPSG = '3035'
            log.info(f'    Reproject to Tiling Schema projection EPSG:{EPSG}')
            ds_epsg_masked = ds_epsg_masked.odc.reproject(how=f'EPSG:{EPSG}', resampling=RESAMPLING_ALGO)
            
            processed_epsgs.append(ds_epsg_masked)
               
        
        if len(processed_epsgs)>1:
            log.info(f'                          ')
            log.info(f'Mosaic datasets of different native UTM zones to a single dataset')
            ds_timeseries = merge_datasets(processed_epsgs)
        else:
            ds_timeseries = processed_epsgs[0]
        # reset bands list
        BANDS = ['B02', 'B03', 'B04', 'B05', 'B07', 'B08', 'B8A']
        
        print('Clip value range')
        chunks = {"time": 1, "y": 1024, "x": 1024}
        ds_timeseries = ds_timeseries.chunk(chunks)
        ds_timeseries = ds_timeseries.where(ds_timeseries > 0, np.nan)
        ds_timeseries = ds_timeseries.where(ds_timeseries<=10000)
        
                        
        log.info(f'                          ')
        log.info('Computing spectral indices:')
        SIS = ['evi', 'ndvi', 'psri2']

        log.info('    EVI...')
        ds_timeseries['evi'] = 2.5 * ((ds_timeseries.B8A - ds_timeseries.B04)/10000) / ((ds_timeseries.B8A/10000 + 6*ds_timeseries.B04/10000 - 7.5*ds_timeseries.B02/10000) + 1)
        log.info('    NDVI...')
        ds_timeseries['ndvi'] = ((ds_timeseries.B05 - ds_timeseries.B03) / ds_timeseries.B07).astype('float32')
        log.info('    PSRI2...')
        ds_timeseries['psri2'] = ((ds_timeseries.B8A - ds_timeseries.B04) / (ds_timeseries.B8A + ds_timeseries.B04)).astype('float32')

        log.info('Clip to typical value range')
        for si in SIS:
            if si in ['ndvi', 'evi']:
                ds_timeseries[si] = ds_timeseries[si].where((ds_timeseries[si]>=-1)&(ds_timeseries[si]<=1))
            else:
                ds_timeseries[si] = ds_timeseries[si].where((ds_timeseries[si]>=-1)&(ds_timeseries[si]<=4))
        
        log.info('Reducing to median value temporal composite')
        # ds_timeseries = ds_timeseries.chunk(chunks)
        ds_timeseries = ds_timeseries.sortby('time')
        composite = ds_timeseries.median(dim='time').astype('float32').compute()
        
        log.info('Define data types and nodata per band')
        VARS = BANDS+SIS

        for band in BANDS:
            dtype = 'uint16'
            nodata = np.iinfo(np.uint16).min #0
            composite[band] = composite[band].astype(dtype)
            composite[band] = composite[band].rio.write_nodata(nodata, inplace=True)
            composite[band].encoding.update({"dtype": dtype})
            
        for si in SIS:  
            scale = 1000 
            dtype = 'int16'
            nodata = np.iinfo(np.int16).min #-32768
            composite[si] = (composite[si]*scale).round()
            composite[si] = composite[si].fillna(nodata).astype(dtype)
            composite[si] = composite[si].rio.write_nodata(nodata, inplace=True)
            composite[si].encoding.update({"dtype": dtype})
            # composite[si].encoding["scale_factor"] = 1/scale


        log.info('Assign time range and tile ID in metadata')
        yyyy = ds_timeseries.isel(time=0).time.dt.year.item()
        mm1 = ds_timeseries.isel(time=0).time.dt.month.item()
        mm2 = ds_timeseries.isel(time=-1).time.dt.month.item()
        dd1 = ds_timeseries.isel(time=0).time.dt.day.item()
        dd2 = ds_timeseries.isel(time=-1).time.dt.day.item()
        composite.attrs['dtr:start_datetime']=f'{yyyy}-{mm1:02d}-{dd1:02d}'
        composite.attrs['dtr:end_datetime']=f'{yyyy}-{mm2:02d}-{dd2:02d}'
        composite.attrs['odc:region_code']=tile
        

        log.info('Create directories and naming conversions')        
        NASROOT='//nas-rs.topo.auth.gr/Latomeia/DROUGHT'
        FOLDER=f'COMPOS/{yyyy}/{mm1}/{tile}'
        DATASET= f'S2L2A_medcomp_{tile}_{yyyy}{mm1:02d}'
        PRODUCT_NAME = 'composites'
        collection_path = f"{NASROOT}/{FOLDER}"
        mkdir(collection_path)
        eo3_path = f'{collection_path}/{DATASET}.odc-metadata.yaml'
        stac_path = f'{collection_path}/{DATASET}.stac-metadata.json'
        datetime_list = [
            ds_timeseries.isel(time=0).time.dt.year.item(),
            ds_timeseries.isel(time=0).time.dt.month.item(),
            1
        ]
        
        
        log.info('Write bands to raster COG files')
        name_measurements = []
        for var in list(composite.data_vars):
            print(var)
                
            file_path = f'{collection_path}/{DATASET}_{var}.tif'
            
            composite[var].rio.to_raster(
                raster_path=file_path, 
                driver='COG',
                dtype=str(composite[var].dtype),
                windowed=True
                )
            name_measurements.append(file_path)
            
        log.info('Prepare metadata YAML document')
        eo3_doc, stac_doc = prepare_eo3_metadata_NAS(
            
        )
        
        log.info('Write metadata YAML document to disk')
        serialise.to_path(Path(eo3_path), eo3_doc)
        with open(stac_path, 'w') as json_file:
            json.dump(stac_doc, json_file, indent=4, default=False)
        
        log.info('Create datacube.model.Dataset from eo3 metadata')
        WORKING_ON_CLOUD=False
        uri = eo3_path if WORKING_ON_CLOUD else f"file:///{eo3_path}"

        resolver = Doc2Dataset(dc.index)
        dataset_tobe_indexed, err  = resolver(doc_in=serialise.to_doc(eo3_doc), uri=uri)
        
        if err:
            log.error(err)
            
        log.info('Index to datacube')
        dc.index.datasets.add(dataset=dataset_tobe_indexed, with_lineage=False)
        
    except Exception as exc:
        msg=f'Failed loading for : Tile {tile_id} | Time: {year_month}\nwith Exception: {exc}'
        log.error(msg)
        client.close()
        cluster.close()
        return
    
    
    
    
    

    
    try:
        log.info(f'Downloading bands and computing median composites: Tile {tile_id} | Time: {year_month}')
        composite = ds_cube_cf[BANDS].median(dim='time').astype('float32')
        composite = composite.compute()
        
        log.info('Convert to unsigned integer 16-bits')
        composite = composite.where(~composite.isnull(), 0).astype('uint16')

        for var in list(composite.data_vars):
            composite[var].attrs['nodata'] = 0

        # Convert to a multi-band DataArray
        da = xr.concat([composite[var] for var in composite.data_vars], dim="band")
        da = da.assign_coords(band=("band", list(range(1, len(composite.data_vars) + 1))))
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