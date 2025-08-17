'''
######################################################################
## ARISTOTLE UNIVERSITY OF THESSALONIKI
## PERSLAB
## REMOTE SENSING AND EARTH OBSERVATION TEAM
##
## DATE:             Aug-2025
## SCRIPT:           composites.py
## AUTHOR:           fotakidis@topo.auth.gr
##
## DESCRIPTION:      Script to compute Median monthly Composites from Sentinel-2 L2A time series and index into ODC
##
#######################################################################
'''

import os
os.environ.setdefault("MPLBACKEND", "Agg")  # must be set before matplotlib is imported

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
from shapely.geometry import shape as shapely_shape

import pystac_client
import odc.stac
import pystac
from odc.stac import configure_rio
# from datacube.utils.aws import configure_s3_access
import planetary_computer
from pystac_client.stac_api_io import StacApiIO
from urllib3 import Retry

import dask.distributed
from dask.distributed import LocalCluster, Client
import tempfile

import gc 
import json
import datetime, pytz
from pathlib import Path
import time, logging

from utils.metadata import prepare_eo3_metadata_NAS
from utils.sentinel2 import check_gri_refinement, plot_mgrs_tiles_with_aoi
from utils.timeseries_processing import merge_nodata0, save_dataset_preview, process_epsg
from utils.utils import mkdir, setup_logger, generate_geojson_files_for_composites

# Ignore warnings
import warnings
import logging
warnings.filterwarnings('ignore') 
logging.getLogger("distributed.worker.memory").setLevel(logging.ERROR)


# ---- keep native libs and Dask tidy (Windows-friendly) ----
os.environ.setdefault("GDAL_CACHEMAX", "512")             # MB; limit GDAL's cache (unmanaged memory)
os.environ["CPL_VSIL_CURL_CACHE_SIZE"] = str(16 * 1024)     # 16 KiB    # avoid big HTTP cache if you hit cloud blobs


def generate_composite(year_month: str, tile_id: str, tile_geom: dict):
    """
    Parameters
    ----------
    year_month : str
        Year–month string in the format "YYYY-MM" (e.g., "2020-01").
    tile_id : str
        Identifier of the tile (e.g., "x09_y07").
    tile_geom : dict or shapely geometry
        Tile geometry in one of the following formats:
          - GeoJSON geometry dict (as stored in your .geojson files, e.g.:
            {
                "type": "Polygon",
                "coordinates": [
                    [
                        [21.8109, 37.0269],
                        [21.73834, 36.5964],
                        ...
                    ]
                ]
            }
          )
          - A shapely geometry object (Polygon, MultiPolygon, etc.).
    """
    
    try:
        start_time = time.time()
        client = None
        cluster = None
        
        logging.info('#######################################################################')
        
        logging.info('Processing started')
        logging.info(f'        Tile: {tile_id}')
        logging.info(f'        Time: {year_month}')
        
        
        logging.info('                          ')
        logging.info('Establishing connection to datacube')
        dc = datacube.Datacube(app='Composite generation', env='drought')
        
        logging.info('                          ')
        logging.info('Sanity test: Check if dataset already exists in the datacube')
        find_ds_in_sc = dc.find_datasets(
            **dict(
                product='composites',
                time=year_month,
                region_code=tile_id
            ),
            ensure_location=True
        )
            
        if find_ds_in_sc:
            logging.warning(f"This composite already exists in {find_ds_in_sc[0].uri}")
            logging.warning("The composite is skipped. Exit function. Continuing to next.")
            logging.info(f'')
            logging.info(f'             !!! SKIPPED: Tile {tile_id} | Time: {year_month} | In {round((time.time() - start_time)/60, 2)} minutes')
            logging.info(f'')
            return
        else:
            logging.info("The composite requested will be computed")
        
        
        logging.info('                                 ')
        logging.info('Initializing Dask cluster for parallelization')
        
        dask.config.set({
            'array.chunk-size': "256 MiB",
            'array.slicing.split_large_chunks': True, #This can make AXIOM very slow
            'distributed.comm.timeouts.connect': '120s',
            'distributed.comm.timeouts.tcp': '120s',
            'distributed.comm.retry.count': 10,
            'distributed.scheduler.allowed-failures': 20,
            "distributed.scheduler.worker-saturation": 1.1, #This should use the new behaviour which helps with memory pile up
            })
        
        cluster = LocalCluster(
            n_workers=8, 
            threads_per_worker=1, 
            processes=True,
            memory_limit='auto', 
            local_directory=tempfile.mkdtemp(),
            dashboard_address=":8787",
            # silence_logs=logging.WARN,
            )
        client = Client(cluster)
        
        # # Dask memory policy: start spilling earlier so RSS stays lower
        # client.run(lambda: __import__("dask").config.set({
        #     "distributed.worker.memory.target":    0.50,  # start spilling at 50% of limit
        #     "distributed.worker.memory.spill":     0.60,
        #     "distributed.worker.memory.pause":     0.80,
        #     "distributed.worker.memory.terminate": 0.95,
        # }))
        
        configure_rio(cloud_defaults=True, client=client) # For Planetary Computer
        logging.info(f'Dask dashboard is available at: {client.dashboard_link}')
        
        
        logging.info('Create directories and naming conversions')   
        yyyy = year_month[0:4]
        mm1 = year_month[5:8]     
        NASROOT='//nas-rs.topo.auth.gr/Latomeia/DROUGHT'
        FOLDER=f'COMPOS/{yyyy}/{mm1}/{tile_id}'
        DATASET= f'S2L2A_medcomp_{tile_id}_{yyyy}{mm1}'
        PRODUCT_NAME = 'composites'
        collection_path = f"{NASROOT}/{FOLDER}"
        mkdir(collection_path)
        eo3_path = f'{collection_path}/{DATASET}.odc-metadata.yaml'
        stac_path = f'{collection_path}/{DATASET}.stac-metadata.json'
        logging.info(f'Dataset location: {collection_path}')
        
        
        logging.info('                          ')
        logging.info('Retrieve tile geometry')
        # # Ensure shapely geometry
        if isinstance(tile_geom, dict): 
            geom_ll = shapely_shape(tile_geom)   
        else:
            geom_ll = tile_geom   
        geom_3035 = gpd.GeoSeries([geom_ll], crs="EPSG:4326").to_crs(epsg=3035)
        
        minl, minf, maxl, maxf = geom_ll.bounds
        minx, miny, maxx, maxy = geom_3035.total_bounds

        logging.info('Create the Bounding Box (φ,λ)')
        aoi_bbox = BoundingBox.from_xy(
            (minl, maxl),
            (minf, maxf)
        ).buffered(xbuff=0.025, ybuff=0.025)

        logging.info('Create the Bounding Box (x,y)')
        tile_bbox = BoundingBox.from_xy(
            (minx, maxx),
            (miny, maxy),
            crs='EPSG:3035'
        )
        
        tile_geobox = odc.geo.geobox.GeoBox.from_bbox(
            tile_bbox, 
            resolution=odc.geo.Resolution(x=20,y=-20)
        )

        
        logging.info('                          ')
        logging.info('Connect to Planetary Computer STAC Catalog')
        stac_api_io = StacApiIO(max_retries=Retry(total=5, backoff_factor=5))
        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
            stac_io=stac_api_io
        )
        
        logging.info('Search the STAC Catalog')
        cloud_cover = 70
        search = catalog.search(
            collections=["sentinel-2-l2a"], #hls2-s30
            bbox=aoi_bbox,
            datetime=year_month,
            limit=100,
            query={
                "eo:cloud_cover": {"lt":cloud_cover},
                "s2:nodata_pixel_percentage": {"lt":33},
            },
        )
        logging.info('        Query parameters:')
        logging.info(f'            url:          {search.url}')
        logging.info(f'            client:       {search.client.id}')
        logging.info(f'            collection:   {search._parameters['collections'][0]}')
        logging.info(f'            bbox:         {search._parameters['bbox']}')
        logging.info(f'            time range:   {search._parameters['datetime']}')
        logging.info(f'            cloud cover:  0% - {search._parameters['query']['eo:cloud_cover']['lt']}%')
        logging.info(f'            nodata cover: 0% - {search._parameters['query']['s2:nodata_pixel_percentage']['lt']}%')
        
        logging.info('                          ')
        logging.info('Searching...')
        items = search.item_collection()
        
        logging.info(f'Query found {len(items)} items')
        
        logging.info('Searching for GRI REFINED scenes:')
        refined_items, df_refinement_status = check_gri_refinement(items)
        
        logging.info('                                 ')
        logging.info(f'{len(refined_items)}/{len(df_refinement_status)} were refined by GRI.')
        
        if len(refined_items) == 0:
            msg = f"Tile {tile_id} | Time: {year_month}: All scenes are flagged NOT REFINED."
            logging.warning(msg)
            logging.warning(f"Tile {tile_id} | Time: {year_month}: The composite will be flagged with _NOREFINED.")
            REFINEMENT_FLAG = 'NOTREFINED'
            refined_items = items
        else:
            REFINEMENT_FLAG = 'REFINED'

        N = 10
        logging.info(f'Looking for up to {N} cleanest images within spatiotemporal range of each MGRS tile')
        filtered_items = []
        mgrs_tiles = np.unique([i.properties['s2:mgrs_tile'] for i in refined_items])
        epsgs = np.unique([i.properties['proj:epsg'] for i in refined_items])
        
        for mgrstile in mgrs_tiles:
            item_mgrs_sorted = sorted([
                i for i in refined_items if i.properties['s2:mgrs_tile'] == str(mgrstile)
                ], key=lambda item: item.properties['eo:cloud_cover'])
            
            if len(item_mgrs_sorted) > N:
                filtered_items.extend(item_mgrs_sorted[:N])
            else:
                filtered_items.extend(item_mgrs_sorted)
        logging.info(f"Filtered cleanest scenes: Kept {len(filtered_items)} out of {len(refined_items)} items. ")
        logging.info(f"        Included MGRS Tiles: {mgrs_tiles}")
        logging.info(f"        Included EPSG codes: {epsgs}")

        logging.info('Selected scenes:')
        for stacitem in filtered_items:
            logging.info(f'        {stacitem.id}')
            
        s2l2a_ids = [stacitem.id for stacitem in filtered_items]
            
        with open(f"{collection_path}/{DATASET}_IncludedScenes.txt", "w") as f:
            for stacitem in filtered_items:
                f.write(stacitem.id + "\n")

        plot_mgrs_tiles_with_aoi( # It has logging in it
            filtered_items, 
            aoi_bbox, 
            save_path=f'{collection_path}/{DATASET}_InDataFootprint.jpeg'
        )

        logging.info(f'                                 ')
        logging.info(f'Downstream STAC items from Planetary Computer')

        # Loading for each EPSG and Native RESOLUTION separatelly, then merging into a single xr.Dataset
        # to to a correct downsampling
        processed_epsgs = []
        for EPSG in epsgs:
            processed_epsgs.append(process_epsg(filtered_items, aoi_bbox, EPSG))
               
        
        logging.info('Clip and align to tile geometry')
        processed_epsgs_to_tile = [ds.odc.reproject(how=tile_geobox) for ds in processed_epsgs]
        del processed_epsgs
        gc.collect()
        
        if len(processed_epsgs_to_tile)>1:
            logging.info(f'                          ')
            logging.info(f'Mosaic datasets of different native UTM zones to a single dataset')
            ds_timeseries = merge_nodata0(processed_epsgs_to_tile, vars_mode="intersection", method="mean", chunks=None)
        else:
            ds_timeseries = processed_epsgs_to_tile[0]
            
        del processed_epsgs_to_tile
        client.run(lambda: __import__("gc").collect()); gc.collect()
        gc.collect()
        
        # https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a#Baseline-Change
        BANDS = ['B02', 'B03', 'B04', 'B05', 'B07', 'B8A']
        baseline400_mask = ds_timeseries.time > pd.Timestamp('2022-01-25')
        if baseline400_mask.any():
            logging.info('Scale SR to Sen2Cor Baseline 4.00 - Subtract 1000 in dates post 2022-01-25')
            boa = ds_timeseries[BANDS].astype('i4')                       # avoid uint16 underflow
            adj = xr.where(baseline400_mask, boa - 1000, boa)                    # subtract only post-cutover
            adj = adj.clip(min=0).astype('u2') 
            
            ds_timeseries = ds_timeseries.assign({b: adj[b] for b in BANDS})
            
            for b in BANDS:
                ds_timeseries[b].attrs['nodata'] = np.iinfo(np.uint16).min #0
        else:
            del baseline400_mask
        
        
        logging.info('Creating preview plot of input scenes')
        save_dataset_preview(ds_timeseries, "B04", f'{collection_path}/{DATASET}_InDataPreview.jpeg', dpi=300)
        
        # reset bands list
        ds_timeseries = ds_timeseries[['B02', 'B03', 'B04', 'B05', 'B07', 'B8A']]
        
        logging.info('Clip value range')
        ds_timeseries = ds_timeseries.where(ds_timeseries > 0, np.nan)
        ds_timeseries = ds_timeseries.where(ds_timeseries<=10000)
        
                        
        logging.info(f'                          ')
        logging.info('Computing spectral indices:')
        SIS = ['evi', 'ndvi', 'psri2']

        logging.info('    EVI...')
        ds_timeseries['evi'] = 2.5 * ((ds_timeseries.B8A - ds_timeseries.B04)/10000) / ((ds_timeseries.B8A/10000 + 6*ds_timeseries.B04/10000 - 7.5*ds_timeseries.B02/10000) + 1)
        logging.info('    NDVI...')
        ds_timeseries['ndvi'] = ((ds_timeseries.B05 - ds_timeseries.B03) / ds_timeseries.B07).astype('float32')
        logging.info('    PSRI2...')
        ds_timeseries['psri2'] = ((ds_timeseries.B8A - ds_timeseries.B04) / (ds_timeseries.B8A + ds_timeseries.B04)).astype('float32')

        logging.info('Clip to typical value range')
        for si in SIS:
            if si in ['ndvi', 'evi']:
                ds_timeseries[si] = ds_timeseries[si].where((ds_timeseries[si]>=-1)&(ds_timeseries[si]<=1))
            else:
                ds_timeseries[si] = ds_timeseries[si].where((ds_timeseries[si]>=-1)&(ds_timeseries[si]<=4))
        
        logging.info('Reducing to median value temporal composite')
        ds_timeseries = ds_timeseries.sortby('time')
        composite = ds_timeseries.median(dim='time').astype('float32').compute()
        
        # keep time metadata
        yyyy = ds_timeseries.isel(time=0).time.dt.year.item()
        mm1 = ds_timeseries.isel(time=0).time.dt.month.item()
        mm2 = ds_timeseries.isel(time=-1).time.dt.month.item()
        dd1 = ds_timeseries.isel(time=0).time.dt.day.item()
        dd2 = ds_timeseries.isel(time=-1).time.dt.day.item()
        datetime_list = [
            ds_timeseries.isel(time=0).time.dt.year.item(),
            ds_timeseries.isel(time=0).time.dt.month.item(),
            1
        ]
        del ds_timeseries
        client.run(lambda: __import__("gc").collect()); gc.collect()
        gc.collect()
        
        logging.info('Define data types and nodata per band')
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


        logging.info('Assign time range and tile ID in metadata')
        composite.attrs['dtr:start_datetime']=f'{yyyy}-{mm1:02d}-{dd1:02d}'
        composite.attrs['dtr:end_datetime']=f'{yyyy}-{mm2:02d}-{dd2:02d}'
        composite.attrs['odc:region_code']=tile_id
        composite.attrs['gri:refinement']=REFINEMENT_FLAG
        composite.attrs['composite:input']=s2l2a_ids
        
        
        logging.info('Write bands to raster COG files')
        name_measurements = []
        for var in list(composite.data_vars):
            file_path = f'{collection_path}/{DATASET}_{var}.tif'
            
            composite[var].rio.to_raster(
                raster_path=file_path, 
                driver='COG',
                dtype=str(composite[var].dtype),
                windowed=True
                )
            name_measurements.append(file_path)
            
            logging.info(f'Write {var.upper()} -> {file_path}')
            


        logging.info('Prepare metadata YAML document')        
        eo3_doc, stac_doc = prepare_eo3_metadata_NAS(
            dc=dc,
            xr_cube=composite, 
            collection_path=collection_path,
            dataset_name=DATASET,
            product_name=PRODUCT_NAME,
            product_family='ard',
            bands=VARS,
            name_measurements=name_measurements,
            datetime_list=datetime_list,
            set_range=False,
            lineage_path=None,
            version=1,
            )
        
        del composite
        client.run(lambda: __import__("gc").collect()); gc.collect()
        gc.collect()
        
        
        logging.info('Write metadata YAML document to disk')
        serialise.to_path(Path(eo3_path), eo3_doc)
        with open(stac_path, 'w') as json_file:
            json.dump(stac_doc, json_file, indent=4, default=False)
        
        logging.info('Create datacube.model.Dataset from eo3 metadata')
        WORKING_ON_CLOUD=False
        uri = eo3_path if WORKING_ON_CLOUD else f"file:///{eo3_path}"

        resolver = Doc2Dataset(dc.index)
        dataset_tobe_indexed, err  = resolver(doc_in=serialise.to_doc(eo3_doc), uri=uri)
        
        if err:
            logging.error(f'✗ {err}')
            
        logging.info('Index to datacube')
        dc.index.datasets.add(dataset=dataset_tobe_indexed, with_lineage=False)
        
        logging.info(f'')
        logging.info(f'             ✔✔✔ COMPLETED: Tile {tile_id} | Time: {year_month} | In {round((time.time() - start_time)/60, 2)} minutes')
        logging.info(f'')
    except Exception as exc:
        msg=f'             ✖✖✖ FAILED loading for : Tile {tile_id} | Time: {year_month} | with Exception: {exc}' # ✗
        logging.error(msg)
        return
    finally:
        try:
            if client is not None:
                logging.info('Closing Dask client')
                client.close()
        finally:
            if cluster is not None:
                logging.info('Closing Dask cluster')
                cluster.close()
                
                
if __name__ == "__main__":
    import argparse, json, sys, os, datetime, pytz
    from utils.utils import setup_logger

    p = argparse.ArgumentParser(description="Run ONE composite from a single .geojson and exit.")
    p.add_argument("--geojson", required=True, help="Path to a single GeoJSON file")
    args = p.parse_args()

    try:
        with open(args.geojson, "r", encoding="utf-8") as f:
            d = json.load(f)
        year_month = d["properties"]["year_month"]
        tile_id    = d["properties"]["tile_id"]
        tile_geom  = d["geometry"]
        
        log = setup_logger(
            logger_name='compgen_',
            logger_path=f'../logs/compgen/compgen_{year_month}_{tile_id}_{datetime.datetime.now(pytz.timezone("Europe/Athens")).strftime("%Y%m%dT%H%M%S")}.log',
            logger_format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        )

        generate_composite(year_month=year_month, tile_id=tile_id, tile_geom=tile_geom)
        sys.exit(0)         # success (including "skipped" is still success)
    except Exception:
        import logging
        logging.exception("Fatal error in composites.py")
        sys.exit(1)        # fail