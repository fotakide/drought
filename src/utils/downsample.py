'''
######################################################################
## ARISTOTLE UNIVERSITY OF THESSALONIKI
## PERSLAB
## REMOTE SENSING AND EARTH OBSERVATION TEAM
##
## DATE:             Aug-2025
## SCRIPT:           utils/downsample.py
## AUTHOR:           fotakidis@topo.auth.gr
##
## DESCRIPTION:      Utility module to downsample by binning 10m bands to 20m resolution like ESA Sen2Cor
##
#######################################################################
'''

import xarray as xr
import numpy as np
from numpy import uint16, mean
from skimage.measure import block_reduce

def s2_downsample_dataset_10m_to_20m(ds: xr.Dataset, y="y", x="x") -> xr.Dataset:
    """
    Sen2Cor-style 10m -> 20m downsampling for all variables in a Dataset:
      - 2x2 block mean over (y, x)
      - round-half-up (+0.5) for integer vars
      - cast back to original dtype for those integer vars
    Time (and any other) dims are preserved.
    """
    if y not in ds.dims or x not in ds.dims:
        raise ValueError(f"Dataset must have '{y}' and '{x}' dims.")

    # Coarsen spatially; boundary='trim' drops the last row/col if odd-sized.
    coarsened = ds.coarsen({y: 2, x: 2}, boundary="trim").mean()

    # Round/cast back for integer-typed variables (e.g., uint16 Sentinel-2 DNs)
    out = coarsened.copy()
    for name, da in coarsened.data_vars.items():
        orig_dt = ds[name].dtype
        if np.issubdtype(orig_dt, np.integer):
            out[name] = (da + 0.5).astype(orig_dt)

    return out


# def s2_downsample_10m_to_20m(da: xr.DataArray, y="y", x="x") -> xr.DataArray:
#     # Expect uint16 DNs; zeros are valid samples (no special NoData treatment).
#     if da.sizes[y] < 2 or da.sizes[x] < 2:
#         raise ValueError("Array must be at least 2x2 in the spatial dims.")

#     # Trim to an even size so 2x2 blocks line up exactly
#     H2 = da.sizes[y] // 2
#     W2 = da.sizes[x] // 2
#     da = da.isel({y: slice(0, H2 * 2), x: slice(0, W2 * 2)})

#     # Mean over 2x2 windows, then round-half-up and cast to uint16
#     out = da.coarsen({y: 2, x: 2}, boundary="trim").mean()      # float
#     out = (out + 0.5).astype(uint16)                          # round-half-up

#     return out


# def s2_downsample_10m_to_20m_with_block_reduce(da: xr.DataArray, y="y", x="x") -> xr.DataArray:
#     """Just for copy reasons, I exaplain the login in this issue
#         https://github.com/fotakide/drought/issues/22

#         Coasen method is prefered and is exactly the same to reduce_blocks.
#     """
#     H2 = da.sizes[y] // 2
#     W2 = da.sizes[x] // 2
#     da = da.isel({y: slice(0, H2 * 2), x: slice(0, W2 * 2)})

#     def f(a):
#         return np.uint16(block_reduce(a, block_size=(2, 2), func=np.mean) + 0.5)

#     out = xr.apply_ufunc(
#         f, da,
#         input_core_dims=[[y, x]],
#         output_core_dims=[[y, x]],
#         output_sizes={y: H2, x: W2},
#         dask="parallelized",
#         output_dtypes=[np.uint16],
#         vectorize=False,
#     )
#     return out


