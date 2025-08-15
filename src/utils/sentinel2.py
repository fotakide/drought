import logging
import requests

import xml.etree.ElementTree as ET
from typing import List, Tuple

import pystac
import pandas as pd

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
    
    for i, item in enumerate(items):
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

                logging.info(f"{i+1}/{len(items)} - {item.id} -> {refining_flag}")

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


def mask_with_scl(ds, bands):
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
    logging.info(f'    Masking bits: {invalid_scl_values}')
    cloud_binary_mask = ds.SCL.isin(invalid_scl_values)

    bands.remove('SCL')    
    ds = ds[bands].where(~cloud_binary_mask, 0).astype('uint16')
    
    return ds