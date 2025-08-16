import logging
import requests

import xml.etree.ElementTree as ET
from typing import List, Tuple

import pystac
import pandas as pd


import geopandas as gpd
import numpy as np
from shapely.geometry import box


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
    logging.info(f'               Masking bits: {invalid_scl_values}')
    cloud_binary_mask = ds.SCL.isin(invalid_scl_values)

    bands.remove('SCL')    
    ds = ds[bands].where(~cloud_binary_mask, 0).astype('uint16')
    
    return ds


def plot_mgrs_tiles_with_aoi(filtered_items, aoi_bbox, save_path=None):    
    # https://gist.github.com/scottyhq/ed8247f3ae1d42543f7bbfb02a5fa8ad
    """
    Plot MGRS tiles from a STAC ItemCollection with an AOI bbox overlay.
    Either displays the plot in the notebook or saves it as a 300 DPI JPEG.

    Parameters
    ----------
    filtered_items : list or pystac.ItemCollection
        List of filtered STAC items or an ItemCollection.
    aoi_bbox : object
        AOI bounding box (must have left/bottom/right/top or xmin/ymin/xmax/ymax,
        or be an iterable [left, bottom, right, top], or a shapely Polygon).
    save_path : str or None
        If None, plot will be displayed in the current cell.
        If a string, the plot will be saved at that location (JPEG, 300 DPI).
    """
    
    import matplotlib
    matplotlib.use("Agg", force=True)    # ensure no Tk
    import matplotlib.pyplot as plt       # import AFTER forcing backend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    logging.info("Matplotlib backend: %s", matplotlib.get_backend())

    # --- STAC items -> GeoDataFrame ---
    if not isinstance(filtered_items, pystac.ItemCollection):
        items = pystac.ItemCollection(filtered_items)
    else:
        items = filtered_items

    gf = gpd.GeoDataFrame.from_features(items.to_dict(), crs="EPSG:4326")
    tile_col = 's2:mgrs_tile'
    gf[tile_col] = gf[tile_col].astype(str)

    # --- AOI bbox handling ---
    if hasattr(aoi_bbox, "geom_type"):  # Shapely geometry
        aoi_gdf = gpd.GeoDataFrame(geometry=[aoi_bbox], crs="EPSG:4326")
    elif all(hasattr(aoi_bbox, k) for k in ("left", "bottom", "right", "top")):
        l, b, r, t = aoi_bbox.left, aoi_bbox.bottom, aoi_bbox.right, aoi_bbox.top
        aoi_gdf = gpd.GeoDataFrame(geometry=[box(l, b, r, t)], crs="EPSG:4326")
    elif all(hasattr(aoi_bbox, k) for k in ("xmin", "ymin", "xmax", "ymax")):
        l, b, r, t = aoi_bbox.xmin, aoi_bbox.ymin, aoi_bbox.xmax, aoi_bbox.ymax
        aoi_gdf = gpd.GeoDataFrame(geometry=[box(l, b, r, t)], crs="EPSG:4326")
    else:  # assume iterable (left, bottom, right, top)
        l, b, r, t = aoi_bbox
        aoi_gdf = gpd.GeoDataFrame(geometry=[box(l, b, r, t)], crs="EPSG:4326")

    # --- Unique tiles + colors ---
    tiles = np.sort(gf[tile_col].unique())
    cmap = plt.cm.get_cmap('tab20', max(len(tiles), 1))
    color_map = {t: cmap(i % cmap.N) for i, t in enumerate(tiles)}

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 8))

    for t in tiles:
        gf[gf[tile_col] == t].plot(
            ax=ax, facecolor=color_map[t], edgecolor='none', alpha=0.30
        )

    # Overlay AOI bbox
    aoi_gdf.boundary.plot(ax=ax, edgecolor='red', linewidth=2)

    # Legend handles
    tile_handles = [Patch(facecolor=color_map[t], edgecolor='none', alpha=0.30, label=t)
                    for t in tiles]
    aoi_handle = Line2D([0], [0], color='red', linewidth=2, label='AOI Bounding Box')

    ax.legend(
        handles=tile_handles + [aoi_handle],
        title="MGRS Tile IDs (s2:mgrs_tile)",
        loc='upper left',
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
        frameon=True
    )

    # Title & axes styling
    if save_path:
        componame = f"composite {save_path.split('/')[-1].split('_InDataFootprint.jpeg')[0]}"
    else:
        componame = 'composite'
    ax.set_title(f"Tiles Included in the {componame} (WGS84)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save or show
    if save_path:
        logging.info(f'Write input scenes footprint overlay: {save_path}')
        fig.savefig(save_path, format='jpeg', dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()