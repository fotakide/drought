# Drought monitoring in Greece

## Description
Research project analyzing drought impacts on Greek mountain ecosystems, with a focus on fir forest (Abies cephalonica) die-off in regions like Chelmos, Mainalo, Taygetos, Parnonas, and Epirus. Includes environmental monitoring, GIS mapping, climate data analysis, and strategies for resilience and restoration. [source: [dasarxeio.com](https://dasarxeio.com/2025/08/01/145507/?fbclid=IwQ0xDSwL56MVleHRuA2FlbQIxMQABHkbJokQhCMbPWyp9B5BhfTiQjc_i3rtFTZOzDlfeDrLWeoQALBKSSqs7HktX_aem_boICuxbIGToYjKLmx3ZoFQ)]

## EODC Development
Based on [Open Data Cube](https://www.opendatacube.org/). For the EODC the following have been done:
1. Created a [DB](https://opendatacube.readthedocs.io/en/latest/installation/database/setup.html) and [credentials](https://opendatacube.readthedocs.io/en/latest/installation/database/passing-configuration.html)
2. Created a python `environment.yaml`
3. [Initialized](https://opendatacube.readthedocs.io/en/latest/installation/cli.html#datacube-system-init) datacube schema
4. Generated the tiling grid schema that encapsulates the Natura sites
5. [Defined EO products](https://opendatacube.readthedocs.io/en/latest/installation/product-definitions.html)
6. Verified dataset save/retrieval from NAS using EO3 datasets and EODC

### Tiling schema
The study area is dividing by tiling schema in 20 tiles of size 48x48km. The naming follows `x00_y00` format without nagative indexes.

![Grid](wiki_img/Grid.jpg)

## Composite generation
The composite pipeline automates the creation of monthly median mosaics of Sentinel-2 L2A data using [Planetary Computer](https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a) via the [STAC catalog](https://planetarycomputer.microsoft.com/api/stac/v1).

### Key points
- Input: JSON configuration files define the year–month and tile code to process.
- Data access: Images are retrieved from the STAC catalog, filtered by cloud cover (`70%`) and data quality (`nodata lt 33%`).
- Native resolutions: Sentinel-2 provides different bands at 10 m and 20 m resolutions.
- Since 10 m bands are not natively [available at 20 m](https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a), we use the method of [Sen2Cor](https://step.esa.int/main/snap-supported-plugins/sen2cor/sen2cor-v2-12/) by binning (2×2 mean aggregation) to bring them to 20 m instead of simple resampling, according to the [S2 MPC L2A ATBD](https://step.esa.int/thirdparties/sen2cor/2.10.0/docs/S2-PDGS-MPC-L2A-ATBD-V2.10.0.pdf). Sen2Cor uses [`skimage.measure.block_reduce`](https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/measure/block.py#L5-L94), as it can be found in the `L2A_Tables.py` module. However, `xarray`'s `coarsen` function is an exact copy of this function, as stated [in this issue](https://github.com/pydata/xarray/issues/2525). This ensures consistency with ESA’s Sen2Cor processing chain.
- Working with S2-L2A time series from MS PC [requires a baseline change](https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a#Baseline-Change) [to `4.00` Sen2Cor](https://sentinels.copernicus.eu/web/sentinel/-/copernicus-sentinel-2-major-products-upgrade-upcoming), so for scenes post `2022-01-05` (January 25th, 2022) an offset of `-1000` is applied.
- Masking: Cloud, shadow, cirrus, and snow/ice pixels are masked using the Scene Classification Layer (SCL).
- Spectral indices: [NDVI](https://www.indexdatabase.de/db/i-single.php?id=58), [EVI](https://www.indexdatabase.de/db/i-single.php?id=16), and [PSRI2](https://www.indexdatabase.de/db/i-single.php?id=69) are calculated from the composites.
- Compositing: A median temporal composite is produced per tile and month.
- Output: Results are stored as Cloud-Optimized GeoTIFFs (COGs), with metadata ([dataset definitions](https://opendatacube.readthedocs.io/en/latest/installation/dataset-documents.html)) written in both [EO3 YAML](https://eodatasets.readthedocs.io/en/eodatasets3-1.9.3/) and [STAC JSON](https://pystac.readthedocs.io/en/latest/index.html) formats for indexing in the datacube.
