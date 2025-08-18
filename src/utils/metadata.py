'''
######################################################################
## ARISTOTLE UNIVERSITY OF THESSALONIKI
## PERSLAB
## REMOTE SENSING AND EARTH OBSERVATION TEAM
##
## DATE:             Aug-2025
## SCRIPT:           utils/metadata.py
## AUTHOR:           Vangelis Fotakidis (fotakidis@topo.auth.gr)
##
## DESCRIPTION:      Utility module to record EO3 metadata, compatible with ODC schema
##
#######################################################################
'''

import rasterio as rio
from eodatasets3 import DatasetPrepare, DatasetDoc, ValidDataMethod
from eodatasets3.model import ProductDoc, AccessoryDoc
from eodatasets3 import serialise
from eodatasets3.stac import to_stac_item

from shapely import Polygon

import datetime
import pandas as pd
from pathlib import Path


def prepare_eo3_metadata_NAS(
    dc,
    xr_cube, 
    collection_path,
    dataset_name,
    product_name,
    product_family,
    bands,
    name_measurements,
    datetime_list,
    set_range=False,
    lineage_path=None,
    version=1,
    ) -> tuple[DatasetDoc, dict]:
    """
    Prepare eo3 metadata with NAS paths
    """

    y,m,d = datetime_list
    
    with DatasetPrepare(
        dataset_location=Path(collection_path),                 #  A string location is expected to be a URL or VSI path.
        metadata_path=Path(f'{collection_path}/{dataset_name}.odc-metadata.yaml'), #  A string location is expected to be a URL or VSI path.
        allow_absolute_paths=False,
        naming_conventions="default"
    ) as preparer:

        preparer.valid_data_method = ValidDataMethod.bounds

        preparer.product_name = product_name
        preparer.product_family = product_family
        preparer.product_maturity = "stable"
        preparer.dataset_version = str(version) # if used without product_name then 'product': {'name': 'productname_1'}

        preparer.datetime = datetime.datetime(y,m,d)
        if set_range:
            preparer.datetime_range = [
                datetime.datetime(int(xr_cube.attrs['dtr:start_datetime'][0:4]),
                                int(xr_cube.attrs['dtr:start_datetime'][5:7]),
                                int(xr_cube.attrs['dtr:start_datetime'][8:10]),
                                0,0,0),
                datetime.datetime(int(xr_cube.attrs['dtr:end_datetime'][0:4]),
                                int(xr_cube.attrs['dtr:end_datetime'][5:7]),
                                int(xr_cube.attrs['dtr:end_datetime'][8:10]),
                                23,59,59)]
        preparer.processed_now()

        preparer.properties["odc:region_code"] = xr_cube.attrs['odc:region_code']
        preparer.properties["odc:file_format"] = "GeoTIFF"
        preparer.properties["odc:processing_datetime"] = datetime.datetime.now().isoformat()
        preparer.properties["gri:refinement"] = xr_cube.attrs["gri:refinement"]
        preparer.properties["composite:input"] = xr_cube.attrs['composite:input']
        
        if hasattr(xr_cube, 'eo:instrument"'):
            preparer.properties["eo:instrument"] = xr_cube.attrs['eo:instrument']
        if hasattr(xr_cube, 'eo:platform"'):
            preparer.properties["eo:platform"] = xr_cube.attrs['eo:platform']
        preparer.properties["eo:gsd"] = int(abs(xr_cube.odc.geobox.resolution.x))


        if lineage_path:
            preparer.add_accessory_file("lineage", lineage_path) # For composites, path to json with S2 IDs

        # if uuid_lineage:
        #     preparer.note_source_datasets(product_lineage, uuid_lineage) # As in ("ard", metadata["id"]), UUIDs from datacube schema

        polygon_geometry = Polygon(xr_cube.odc.geobox.boundingbox.polygon.boundary.coords)
        preparer.geometry = polygon_geometry

        bands = list(dc.list_measurements().loc[product_name].name.values)
        for name, path in zip(bands, name_measurements):
            preparer.note_measurement(name, str(Path(path).resolve()), relative_to_dataset_location=False) # else: (name, f'{granule_dir}/{path}', relative_to_dataset_location=False)

        eo3_doc = preparer.to_dataset_doc()

        crs, grid_docs, measurement_docs = preparer._measurements.as_geo_docs()

        eo3 = DatasetDoc(
            id=preparer.dataset_id,
            label=preparer.label,
            product=ProductDoc(
                name=preparer.names.product_name, href=preparer.names.product_uri
            ),
            crs=preparer._crs_str(crs) if crs is not None else None,
            geometry=polygon_geometry,
            grids=grid_docs,
            properties=preparer.properties,
            accessories={
                name: AccessoryDoc(path, name=name)
                for name, path in preparer._accessories.items()
            },
            measurements=measurement_docs,
            # lineage=preparer._lineage, # Preparer does not have _lineage
        )

        for measurement in eo3.measurements.values():
            if measurement.grid is None:
                measurement.grid = 'default'

    stac_path = f'{collection_path}/{dataset_name}.stac-metadata.json'
    stac_doc = to_stac_item(dataset=eo3, stac_item_destination_url=stac_path, collection_url=f'file://{collection_path}')
    return eo3_doc, stac_doc