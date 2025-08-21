'''
######################################################################
## ARISTOTLE UNIVERSITY OF THESSALONIKI
## PERSLAB
## REMOTE SENSING AND EARTH OBSERVATION TEAM
##
## DATE:             Aug-2025
## SCRIPT:           tile_generation.py
## AUTHOR:           Vangelis Fotakidis (fotakidis@topo.auth.gr)
##
## DESCRIPTION:      Module to retrieve the tiling grid schema based on an input AOI shapefile.
##
#######################################################################
'''

import os
import logging

from typing import Tuple, Optional, Dict
import geopandas as gpd

from odc.geo.gridspec import GridSpec
from odc.geo import Resolution, CRS, XY, Shape2d, Geometry
from odc.io.text import split_and_check, parse_range_int


# ====================
# sources: 
# 1. https://github.com/opendatacube/odc-tools/blob/dff7b984464a4cc9d6bd9f6f444ef4a292c730d0/libs/dscache/odc/dscache/tools/tiling.py#L13-L41
# 2. https://github.com/digitalearthafrica/deafrica-waterbodies/blob/3517fc6985d89fa006c1644d2080ce73ada054f6/deafrica_waterbodies/tiling.py#L10
# ====================


GRIDS = {
    **{
        f"lambert_gr_{n}": GridSpec(
            crs=CRS("EPSG:3035"),
            tile_shape=Shape2d(48_000.0/n, 48_000.0/n),
            resolution=Resolution(x=n, y=-n),
            origin=XY(4943980.0, 1247980.0),   # Captures Greece without negative indices.
            # origin=(1391985.0, 5087985.0),
            # origin=(2015985.0, 5375985.0),
        )
        for n in (10, 20, 25, 30, 60)
    },
}


def _parse_gridspec_string(s: str) -> GridSpec:
    """
    "epsg:3035;30;1600"
    "epsg:3035;-30x30;1600x1600"
    """

    crs, res, shape = split_and_check(s, ";", 3)
    try:
        if "x" in res:
            res = tuple(float(v) for v in split_and_check(res, "x", 2))
        else:
            res = float(res)
            res = (-res, res)

        if "x" in shape:
            shape = parse_range_int(shape, separator="x")
        else:
            shape = int(shape)
            shape = (shape, shape)
    except ValueError:
        raise ValueError(f"Failed to parse gridspec: {s}") from None

    tsz = tuple(abs(n * res) for n, res in zip(res, shape))

    return GridSpec(crs=CRS(crs), tile_shape=tsz, resolution=res, origin=(0, 0))

def _norm_gridspec_name(s: str) -> str:
    return s.replace("-", "_")


def parse_gridspec_with_name(
    s: str, grids: Optional[Dict[str, GridSpec]] = None
) -> Tuple[str, GridSpec]:
    if grids is None:
        grids = GRIDS

    named_gs = grids.get(_norm_gridspec_name(s))
    if named_gs is not None:
        return (s, named_gs)

    gs = _parse_gridspec_string(s)
    s = s.replace(";", "_")
    return (s, gs)


def get_tiles(resolution, aoi_gdf_path, outfile):
    grid_name = f"lambert_gr_{resolution}"
    grid, gridspec = parse_gridspec_with_name(grid_name)

    print(f"Using the grid {grid} with {gridspec}")
   
    # From the GridSpec get the crs and resolution.
    crs = gridspec.crs
    resolution = abs(gridspec.resolution.x)

    area_footprint = gpd.read_file(aoi_gdf_path).to_crs(crs).dissolve()
    
    # Get the product footprint geopolygon.
    area_footprint = Geometry(geom=area_footprint.geometry[0], crs=crs)

    tiles = gridspec.tiles_from_geopolygon(geopolygon=area_footprint)
    tiles = list(tiles)

    # Get the individual tile geometries.
    tile_geometries = []
    tile_ids = []
    for tile in tiles:
        tile_idx, tile_idy = tile[0]
        tile_geometry = tile[1].extent.geom

        tile_geometries.append(tile_geometry)
        tile_ids.append(f"x{tile_idx:02d}_y{tile_idy:02d}")

    tiles_gdf = gpd.GeoDataFrame(data={"tile_ids": tile_ids, "geometry": tile_geometries}, crs=crs)
    
    tiles_gdf = tiles_gdf.to_crs(crs=4326)
    
    # outfile = f"../geojson/emt_grid_v1.geojson"
    if not os.path.isfile(outfile):
        tiles_gdf.to_file(outfile, driver="GeoJSON")

    return tiles_gdf