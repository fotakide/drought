# pylint: skip-file

bands_s2l2a_compos = {
    "B02": ["blue"],
    "B03": ["green"],
    "B04": ["red"],
    "B05": ["rededge1"],
    "B07": ["rededge3"],
    "B8A": ["nir8A"],
    "NDVI": ["ndvi"],
    "EVI": ["evi"],
    "PSRI2": ["psri2"],
}

style_rgb = {
    "name": "simple_rgb",
    "title": "Simple RGB",
    "abstract": "Simple true-colour image, using the red, green and blue bands",
    # The component keys MUST be "red", "green" and "blue" (and optionally "alpha")
    "components": {
        "red": {
            "B04": 1.0
        },
        "green": {
            "B03": 1.0
        },
        "blue": {
            "B02": 1.0
        }
    },
    "scale_range": [200.0, 3000.0],
    "legend": {
        "show_legend": True,
    }

}

style_ndvi = {
    "name": "ndvi",
    "title": "NDVI",
    "abstract": "Normalised Difference Vegetation Index - a derived index that correlates well with the existence of vegetation",
    "needed_bands": ["NDVI"],
    "index_function": {
        "function": "datacube_ows.band_utils.single_band",
        "mapped_bands": True,
        "kwargs": {
            "band": "ndvi",
        }
    },
    "color_ramp": [
        {
            "value": -0.0,
            "color": "#8F3F20",
            "alpha": 0.0
        },
        {
            "value": 0.0,
            "color": "#8F3F20",
            "alpha": 1.0
        },
        {
            "value": 100,
            "color": "#A35F18"
        },
        {
            "value": 200,
            "color": "#B88512"
        },
        {
            "value": 300,
            "color": "#CEAC0E"
        },
        {
            "value": 400,
            "color": "#E5D609"
        },
        {
            "value": 500,
            "color": "#FFFF0C"
        },
        {
            "value": 600,
            "color": "#C3DE09"
        },
        {
            "value": 700,
            "color": "#88B808"
        },
        {
            "value": 800,
            "color": "#529400"
        },
        {
            "value": 900,
            "color": "#237100"
        },
        {
            "value": 1000,
            "color": "#114D04"
        }
    ],
    # "include_in_feature_info": True,
    # "legend": {
    #     "show_legend": True,
    # }
}

standard_resource_limits = {
    "wms": {
        "zoomed_out_fill_colour": [150, 180, 200, 160],
        "min_zoom_factor": 500.0,
        "max_datasets": 10,
        "dataset_cache_rules": [
            {
                "min_datasets": 4, # Must be greater than zero.  Blank tiles (0 datasets) are NEVER cached
                # The cache-control max-age for this rule, in seconds.
                "max_age": 86400,  # 86400 seconds = 24 hours
            },
            {
                # Rules must be sorted in ascending order of min_datasets values.
                "min_datasets": 8,
                "max_age": 604800,  # 604800 seconds = 1 week
            },
        ]
    },
    "wcs": {
        "max_datasets": 16,
    }
}


# MAIN CONFIGURATION OBJECT

ows_cfg = {
    "global": {
        "response_headers": {
            "Access-Control-Allow-Origin": "*",  # CORS header (strongly recommended)
        },
        "services": {
            "wms": True,
            "wmts": True,
            "wcs": True
        },
        "title": "Open web-services for ODC in Drought Monitoring in Greece",
        "allowed_urls": ["http://localhost:9000",
                         "http://localhost:9000/wms",
                        #  "https://emt-datacube-ows.ngrok.app/wms",
                        #  "https://emt-datacube-ows.ngrok.app"
                          ],
        "info_url": "https://github.com/fotakide",
        "abstract": """This research project analyzes drought impacts on Greek mountain ecosystems, focusing on fir forest (Abies cephalonica) die-off in regions such as Chelmos, Mainalo, Taygetos, Parnonas, and Epirus. The work includes environmental monitoring, GIS mapping, climate data analysis, and the development of strategies for resilience and restoration.""",
        "keywords": [
            "ard",
        ],
        "contact_info": {
            "person": "Vangelis Fotakidis",
            "organisation": "AUTH",
            "position": "PhD Candidate",
            "address": {
                "type": "University",
                "address": "Aristotle University Campus",
                "city": "Thessaloniki",
                "country": "Greece",
            },
            "telephone": "+30 2310",
            "email": "fotakidis@topo.auth.gr",
        },
        "attribution": {
            "title": "Acme Satellites",
            "url": "http://www.acme.com/satellites",
            "logo": {
                "width": 370,
                "height": 247,
                "url": "https://www.auth.gr/wp-content/uploads/banner-horizontal-default-en.png",
                "format": "image/png",
            }
        },
        "fees": "",
        "access_constraints": "",
        "published_CRSs": {
            "EPSG:3035": {  # ETRS89-extended / LAEA Europe
                "geographic": False,
                "horizontal_coord": "x",
                "vertical_coord": "y",
            },
            "EPSG:3857": {  # Web Mercator
                "geographic": False,
                "horizontal_coord": "x",
                "vertical_coord": "y",
            },
            "EPSG:4326": {  # WGS-84
                "geographic": True,
                "vertical_coord_first": True
            },
        }
    },   #### End of "global" section.

    
    "wms": {
        "s3_url": "http://data.au",
        "s3_bucket": "s3_bucket_name",
        "s3_aws_zone": "ap-southeast-2",
        "max_width": 512,
        "max_height": 512,

        "authorities": {
            "auth": "https://authoritative-authority.com",
            "idsrus": "https://www.identifiers-r-us.com",
        }
    }, ####  End of "wms" section.


    "wmts": {
        "tile_matrix_sets": {
            "eeagrid": {
                # The CRS of the Tile Matrix Set
                "crs": "EPSG:3035",
                # The coordinates (in the CRS above) of the upper-left
                # corner of the tile matrix set.
                # My edit: Changed from https://epsg.io/map#srs=32634&x=502338.404794&y=3876270.085061&z=8&layer=streets
                "matrix_origin": (5000000.0, 2300000.0),
                # "matrix_origin": (4321000.0, 3210000.0),
                "tile_size": (256, 256),
                "scale_set": [
                    3779769.4643008336,
                    1889884.7321504168,
                    944942.3660752084,
                    472471.1830376042,
                    236235.5915188021,
                    94494.23660752083,
                    47247.11830376041,
                    23623.559151880207,
                    9449.423660752083,
                    4724.711830376042,
                    2362.355915188021,
                    1181.1779575940104,
                    755.9538928601667,
                ],
                "matrix_exponent_initial_offsets": (1, 0),
            },
        }
    },

    # Config items in the "wcs" section apply to the WCS service to all WCS coverages
    "wcs": {
        "formats": {
            "GeoTIFF": {
                "renderers": {
                    "1": "datacube_ows.wcs1_utils.get_tiff",
                    "2": "datacube_ows.wcs2_utils.get_tiff",
                },
                "mime": "image/geotiff",
                "extension": "tif",
                "multi-time": False
            },
            "netCDF": {
                "renderers": {
                    "1": "datacube_ows.wcs1_utils.get_netcdf",
                    "2": "datacube_ows.wcs2_utils.get_netcdf",
                },
                "mime": "application/x-netcdf",
                "extension": "nc",
                "multi-time": True,
            }
        },
        "native_format": "GeoTIFF",
    }, ###### End of "wcs" section

    # Products published by this datacube_ows instance.
    # The layers section is a list of layer definitions.  Each layer may be either:
    # 1) A folder-layer.  Folder-layers are not named and can contain a list of child layers.  Folder-layers are
    #    only used by WMS and WMTS - WCS does not support a hierarchical index of coverages.
    # 2) A mappable named layer that can be requested in WMS GetMap or WMTS GetTile requests.  A mappable named layer
    #    is also a coverage, that may be requested in WCS DescribeCoverage or WCS GetCoverage requests.

    "layers": [
        {
            # NOTE: This layer IS a mappable "named layer" that can be selected in GetMap requests
            "name": "composites",
            "title": "Sentinel-2 L2A Composites",
            "abstract": "Sentinel-2 L2A monthly median composites from imagery retrieved by Microsoft Planetary Computer",
            "product_name": "composites",
            "bands": bands_s2l2a_compos,
            "resource_limits": standard_resource_limits,
            "native_crs": "EPSG:3035",
            "native_resolution": [20.0, -20.0],
            "flags": None,
            "dynamic": True,
            "patch_url_function":  "datacube_ows.ogc_utils.nas_patch",
                # https://datacube-ows.readthedocs.io/en/latest/cfg_layers.html#url-patching-patch-url-function
                # https://github.com/digitalearthpacific/pacific-cube-in-a-box/blob/main/ows/ows_config/radar_backscatter/ows_s1_cfg.py#L88
            "image_processing": {
                "extent_mask_func": "datacube_ows.ogc_utils.mask_by_val",
                "always_fetch_bands": [],
                "fuse_func": None,
                "manual_merge": False,
                "apply_solar_corrections": False,
            },
            "styling": {
                "styles": [
                    style_rgb, style_ndvi
                    ],
            },
        }
    ]  ##### End of "layers" list.
} #### End of configuration object
