import logging
import odc.geo

import pystac_client
from pystac_client.stac_api_io import StacApiIO
from urllib3 import Retry

import xarray as xr
import numpy as np

from utils.downsample import s2_downsample_dataset_10m_to_20m
from utils.sentinel2 import mask_with_scl

import gc


BANDS_R10m = ['B02', 'B03', 'B04']
BANDS_R20m = ['B05', 'B07', 'B8A', 'SCL']

def connect_to_STAC_catalog(catalog_endpoint="planetary_computer"):
    # Open a client 
    retry = Retry(
        total=10,
        backoff_factor=1,
        status_forcelist=[404, 502, 503, 504],
        allowed_methods=None, # {*} for CORS
    )
        
    stac_api_io = StacApiIO(max_retries=retry)

    logging.info("        Initialize the STAC client")
    if catalog_endpoint=='planetary_computer':
        # Planetary Computer
        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            # modifier=planetary_computer.sign_inplace,
            stac_io=stac_api_io
        )
    elif catalog_endpoint=='earth_search':
        # AWS
        catalog = pystac_client.Client.open(
            "https://earth-search.aws.element84.com/v1",
            stac_io=stac_api_io
            )
    elif catalog_endpoint=='landsatlook':
        catalog = pystac_client. Client.open(
            "https://landsatlook.usgs.gov/stac-server/",
            stac_io=stac_api_io
        )
    else:
        logging.error("You must provide a catalog endpoint alias, available [planetary_computer, earth_search, landsatlook]")

    return catalog


def refetch_S2L2A_items_from_catalog(epsg_filtered_items):
    catalog = connect_to_STAC_catalog(catalog_endpoint="planetary_computer")
    ids = [item.id for item in epsg_filtered_items]
    search = catalog.search(
        ids=ids,
        collections=["sentinel-2-l2a"],  # optional but recommended
        limit=len(ids)
    )

    return search.item_collection()


def odc_stac_load_Items(unsigned_items, aoi_bbox, bands, epsg, resolution):
    # Harden GDAL before odc/rasterio touch anything networky
    import os
    os.environ.setdefault("GDAL_HTTP_TIMEOUT", "30")
    os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "2")
    os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "1")
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "YES")
    os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.tiff,.TIF,.TIFF,.json")
    os.environ.setdefault("VSI_CACHE", "FALSE")
        
    import planetary_computer as pc
    import odc.stac
    
    signed_items = [pc.sign(it) for it in unsigned_items]
    return odc.stac.stac_load(
        signed_items,
        bbox=aoi_bbox,
        bands=bands,
        chunks=dict(y=1024, x=1024),
        crs=f'EPSG:{epsg}',  # {epsgs[0]}
        resolution=resolution,
        groupby='time', # if 'time' loads all items, retaining duplicates
        fail_on_error=False,
        # resampling={
        #     "*": RESAMPLING_ALGO,
        # },
    ).compute()
    

def process_epsg(filtered_items, aoi_bbox, EPSG):
    
    logging.info(f'                                 ')
    logging.info(f'Loading bands of diferent resolutions in EPSG:{EPSG}')
    
    processed_bands = []
    geobox = None
    logging.info(f'    Filtering items with native UTM EPSG:{EPSG}')
    epsg_filtered_items = [item for item in filtered_items if item.properties['proj:epsg']==EPSG]
    logging.info(f'    {len(epsg_filtered_items)} Items in EPSG:{EPSG}')
    for RESOLUTION in [20, 10]:
        if RESOLUTION==10:
            BANDS=BANDS_R10m
        else:
            BANDS=BANDS_R20m
        
        logging.info('                       ')
        logging.info('    Loading parameters:')
        logging.info(f'        Bands: {BANDS}')
        logging.info(f'        Spatial resolution: {RESOLUTION}')

        logging.info(f'        Get Assets from STAC Items')
        assets_to_load = refetch_S2L2A_items_from_catalog(epsg_filtered_items)
        
        logging.info(f'        Loading {len(epsg_filtered_items)} STAC Items....')
        ds_cube = odc_stac_load_Items(
            assets_to_load, 
            aoi_bbox, 
            BANDS, 
            EPSG,
            RESOLUTION
        )
        # ds_cube = odc.stac.stac_load(
        #     epsg_filtered_items,
        #     bbox=aoi_bbox,
        #     bands=BANDS,
        #     chunks=dict(y=1024, x=1024),
        #     crs=f'EPSG:{EPSG}',  # {epsgs[0]}
        #     resolution=RESOLUTION,
        #     groupby='time', # if 'time' loads all items, retaining duplicates
        #     fail_on_error=True,
        #     # resampling={
        #     #     "*": RESAMPLING_ALGO,
        #     # },
        # ).compute()
        
        logging.info(f'        Cube has been downloaded')
        
        if RESOLUTION==10:
            logging.info('        Downsample 10m bands to 20m by average 2x2 binning')
            ds_cube = s2_downsample_dataset_10m_to_20m(ds_cube)
            RESAMPLING_ALGO = "bilinear"
            logging.info(f'        Warp (reproject) 2x2 binned bands to match shape of native 20m bands: method={RESAMPLING_ALGO}')
            ds_bands = ds_cube.odc.reproject(how=geobox, resampling=RESAMPLING_ALGO)
        elif RESOLUTION==20:
            logging.info('        Fix order of dimensions')
            ds_bands = ds_cube[['time','y','x']+list(ds_cube.data_vars)]
            geobox = ds_cube.odc.geobox
        
        del ds_cube
        logging.info(f'        Append bands to the band-list of EPSG:{EPSG}')
        processed_bands.append(ds_bands)
    
    
    logging.info('                       ')
    logging.info('    Merging bands in a single dataset')
    logging.info('    Ensure all partial datasets are at (x=20, y=-20) resolution')
    for dataset in processed_bands:
        dataset.odc.geobox.resolution == odc.geo.Resolution(x=20, y=-20)
    
    logging.info('    Clip to the intersection of indexes')
    xmin = xmax = ymin = ymax = None
    for da in processed_bands:
        xmin = da.x.min().item() if xmin is None or da.x.min() > xmin else xmin
        xmax = da.x.max().item() if xmax is None or da.x.max() < xmax else xmax
        ymin = da.y.min().item() if ymin is None or da.y.min() > ymin else ymin
        ymax = da.y.max().item() if ymax is None or da.y.max() < ymax else ymax

    merge_bands = []
    logging.info('    Make a list of bands to merge')
    for da in processed_bands:
        merge_bands.append(da.sel(x=slice(xmin, xmax), y=slice(ymax,ymin)))
    
    logging.info('    Merge on intersection')
    ds_epsg = xr.merge(
        merge_bands,
        compat="no_conflicts",
        combine_attrs="drop_conflicts",
        join="inner"
    )
    
    logging.info(f'    Apply masks on clouds, shadows, thin cirrus, and snow/ice')
    BANDS = ['B02', 'B03', 'B04', 'B05', 'B07', 'B8A', 'SCL']
    ds_epsg_masked = mask_with_scl(ds_epsg, BANDS)
    
    RESAMPLING_ALGO = "bilinear"
    EPSG = '3035'
    logging.info(f'    Reproject to Tiling Schema projection EPSG:{EPSG}')
    ds_epsg_masked = ds_epsg_masked.odc.reproject(how=f'EPSG:{EPSG}', resampling=RESAMPLING_ALGO)
    
    # processed_epsgs.append(ds_epsg_masked)
    # gc.collect()
    return ds_epsg_masked


def merge_nodata0(
    datasets, 
    vars_mode="intersection", 
    method="max", 
    chunks=None
):
    """
    Merge multiple xarray Datasets on time,y,x where:
      - union of times is used
      - 0 is treated as NoData (ignored)
      - overlaps are combined using the given method (max, mean, median, min)
      - result preserves original integer dtype (0 for NoData)

    Parameters
    ----------
    datasets : list[xr.Dataset]
        Input datasets. Must be on the same grid (x,y) and share coordinates when times overlap.
    vars_mode : {"intersection", "union"}
        Which set of data variables to merge.
    method : {"max", "mean", "median", "min"}
        Aggregation method for overlaps.
    chunks : dict or "auto" or None
        Optional dask chunking to apply to the result.

    Returns
    -------
    xr.Dataset
        Merged dataset with union time and aggregated overlaps.
    """
    if len(datasets) == 0:
        raise ValueError("Provide at least one dataset")

    # 1) Union of times
    all_times = xr.concat([ds["time"] for ds in datasets], dim="time")
    all_times = xr.DataArray(np.unique(all_times.values), dims=("time",))

    # 2) Align on union of times
    ds_aligned = [ds.reindex(time=all_times) for ds in datasets]

    # 3) Variable selection
    if vars_mode == "intersection":
        var_names = set(ds_aligned[0].data_vars)
        for ds in ds_aligned[1:]:
            var_names &= set(ds.data_vars)
    elif vars_mode == "union":
        var_names = set()
        for ds in ds_aligned:
            var_names |= set(ds.data_vars)
    else:
        raise ValueError("vars_mode must be 'intersection' or 'union'")

    # Only keep 3D (time,y,x) vars
    def is_band(da):
        return set(da.dims) == {"time", "y", "x"}

    var_names = [v for v in var_names if v in ds_aligned[0] and is_band(ds_aligned[0][v])]

    # 4) Merge each variable
    merged = {}
    for v in var_names:
        out_dtype = None
        for ds in ds_aligned:
            if v in ds:
                out_dtype = ds[v].dtype
                break

        sources = []
        for ds in ds_aligned:
            if v in ds:
                sources.append(ds[v].where(ds[v] != 0))  # 0 -> NaN
            else:
                shape = (all_times.size, ds_aligned[0].sizes["y"], ds_aligned[0].sizes["x"])
                sources.append(
                    xr.full_like(ds_aligned[0][var_names[0]], np.nan).rename(v)
                )

        stacked = xr.concat(sources, dim="source")

        # Choose aggregation method
        if method == "max":
            agg = stacked.max("source", skipna=True)
        elif method == "min":
            agg = stacked.min("source", skipna=True)
        elif method == "mean":
            agg = stacked.mean("source", skipna=True)
        elif method == "median":
            agg = stacked.median("source", skipna=True)
        else:
            raise ValueError("method must be one of: 'max', 'mean', 'median', 'min'")

        # Fill NoData with 0 and cast back to original dtype
        if out_dtype is None:
            out_dtype = np.uint16
        agg = agg.fillna(0).astype(out_dtype)

        merged[v] = agg

    # 5) Build output dataset
    out = xr.Dataset(merged, coords={c: ds_aligned[0][c] for c in ds_aligned[0].coords})
    out = out.sortby("time")

    if "spatial_ref" in ds_aligned[0].coords:
        out = out.assign_coords(spatial_ref=ds_aligned[0].coords["spatial_ref"])

    if chunks is not None:
        out = out.chunk(chunks if chunks != "auto" else {})

    return out


def save_dataset_preview(ds, var_name, save_path, dpi=300, col_wrap=4, **plot_kwargs):
    """
    Create a FacetGrid preview of a variable in a Dataset over time and save as a JPEG.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset containing the variable to plot.
    var_name : str
        Name of the variable to plot.
    save_path : str
        Path to save the JPEG (should end with .jpg or .jpeg).
    dpi : int, optional
        Resolution in dots per inch (default 300).
    col_wrap : int, optional
        Number of columns before wrapping in the FacetGrid (default 4).
    **plot_kwargs :
        Additional keyword arguments passed to xarray.DataArray.plot.
    """
    
    q=(0.02, 0.98)
    
    if var_name not in ds.data_vars:
        raise ValueError(f"Variable '{var_name}' not found in dataset.")

    # fg = ds[var_name].plot(col='time', col_wrap=col_wrap, **plot_kwargs)
    da = ds[var_name]
    da_valid = da.where(da != 0)                     # mask nodata=0 so it doesn’t skew scaling

    # pick sensible bounds from quantiles across all times (lazy-safe with dask)
    vmin = float(da_valid.quantile(q[0]))
    vmax = float(da_valid.quantile(q[1]))

    fg = da_valid.plot(col='time', col_wrap=col_wrap, vmin=vmin, vmax=vmax)
    
    # Add a single title for the whole figure
    fg.fig.suptitle(f"Preview – {var_name}", fontsize=16, fontweight="bold", y=1.02)

    fg.fig.savefig(save_path, format="jpeg", dpi=dpi, bbox_inches="tight")
    gc.collect()
    logging.info(f"Saved preview for '{var_name}' to {save_path}")