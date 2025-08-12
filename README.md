# Drought monitoring in Greece

## Description
Research project analyzing drought impacts on Greek mountain ecosystems, with a focus on fir forest (Abies cephalonica) die-off in regions like Chelmos, Mainalo, Taygetos, Parnonas, and Epirus. Includes environmental monitoring, GIS mapping, climate data analysis, and strategies for resilience and restoration. [source: [dasarxeio.com](https://dasarxeio.com/2025/08/01/145507/?fbclid=IwQ0xDSwL56MVleHRuA2FlbQIxMQABHkbJokQhCMbPWyp9B5BhfTiQjc_i3rtFTZOzDlfeDrLWeoQALBKSSqs7HktX_aem_boICuxbIGToYjKLmx3ZoFQ)]

## EODC Development
For the EODC the following have been done:
1. Created a DB and credentials
2. Created a python environment.yaml
3. Initialized datacube schema
4. Generated the tiling grid schema that encapsulates the Natura sites
5. Product definitions
6. Checked how to save/retrieve new datasets from NAS using eo3 datasets and EODC

## Composite generation

Retrieve imagery from Microsoft Planetary Computer and utilize STAC to get the values:
1. Based on EMSN217 for composite generation
