def evi(ds):
    # constants
    G = 2.5
    C1 = 6.0
    C2 = 7.5
    L = 1.0
    
    # scale bands to [0, 1]
    nir = ds.B8A / 10000.0
    red = ds.B04 / 10000.0
    blue = blue / 10000.0
    
    return (G * ((nir-red) / (nir + C1 * red - C2 * blue +L))).astype('float32')

def ndvi(ds):
    
    return ((ds.B8A - ds.B04) / (ds.B8A + ds.B04)).astype('float32')

def psri2(ds):
    
    return ((ds.B05 - ds.B03) / ds.B07).astype('float32')