import numpy as np


def rgb2cie(r, g, b):
    X = 2.7689 * r + 1.7517 * g + 1.1302 * b
    Y = 1.0000 * r + 4.5907 * g + 0.0601 * b
    Z = 0.0000 * r + 0.0565 * g + 5.5943 * b
    total = X + Y + Z
    x = np.divide(X, total, out=np.zeros_like(
        X, dtype=float), where=(total != 0))
    y = np.divide(Y, total, out=np.zeros_like(
        Y, dtype=float), where=(total != 0))
    return x, y


def get_ndii(nir, swir1):
    ndii = np.divide((nir - swir1), (nir + swir1),
                     out=np.zeros_like(nir, dtype=float), where=(nir + swir1 != 0))
    return ndii


def get_eri(nir, swir2):
    eri = np.divide((nir - swir2), (nir + swir2),
                    out=np.zeros_like(nir, dtype=float), where=(nir + swir2 != 0))
    return eri


def get_ndrbi(blue, red):
    ndnbi = np.divide((red - blue), (blue + red),
                      out=np.zeros_like(blue, dtype=float), where=(blue + red != 0))
    return ndnbi


def get_ndnbi(blue, nir):
    ndnbi = np.divide((nir - blue), (blue + nir),
                      out=np.zeros_like(blue, dtype=float), where=(blue + nir != 0))
    return ndnbi


def get_swir12(swir1, swir2):
    swir12 = np.divide((swir1 - swir2), (swir1 + swir2),
                       out=np.zeros_like(swir1, dtype=float), where=(swir1 + swir2 != 0))
    return swir12


def get_ndwi(green, nir):
    ndwi = np.divide((green - nir), (green + nir),
                     out=np.zeros_like(green, dtype=float), where=(green + nir != 0))
    return ndwi


def get_ndsi(green, swir1):
    ndsi = np.divide((green - swir1), (green + swir1),
                     out=np.zeros_like(green, dtype=float), where=(green + swir1 != 0))
    return ndsi


def get_guano_index1(blue, green, red, nir, swir1, swir2):
    term1 = np.divide(10 * (nir - green), nir + green + 0.1,
                      out=np.zeros_like(nir, dtype=float), where=(nir + green + 0.1 != 0))

    term2 = np.divide(nir - swir1, nir + swir1,
                      out=np.zeros_like(nir, dtype=float), where=(nir + swir1 != 0))

    term3 = np.divide(red - blue, red + blue + 0.01,
                      out=np.zeros_like(red, dtype=float), where=(red + blue + 0.01 != 0))

    term4 = np.divide(10 * (swir1 - swir2), swir1 + swir2 + 0.1,
                      out=np.zeros_like(swir1, dtype=float), where=(swir1 + swir2 + 0.1 != 0))

    return term1 + term2 + term3 + term4


def get_guano_index2(green, nir, swir1, swir2):
    numerator = (swir1 - swir2) + 1.5 * (nir - green)
    denominator = 2 * (swir1 + swir2) + 0.5

    guano_index2 = np.divide(numerator, denominator,
                             out=np.zeros_like(numerator, dtype=float), where=denominator != 0)

    return guano_index2


def indices_generate(data):
    [blue, green, red, nir, swir1, swir2] = data

    [tx, ty] = rgb2cie(red, green, blue)
    [fx, fy] = rgb2cie(nir, red, blue)
    ndii = get_ndii(nir, swir1)
    eri = get_eri(nir, swir2)
    ndrbi = get_ndrbi(blue, red)
    ndnbi = get_ndnbi(blue, nir)
    swir12 = get_swir12(swir1, swir2)
    ndwi = get_ndwi(green, nir)
    ndsi = get_ndsi(green, swir1)
    guano_index1 = get_guano_index1(blue, green, red, nir, swir1, swir2)
    guano_index2 = get_guano_index2(green, nir, swir1, swir2)
    indices_list = np.array([tx, ty, fx, fy, ndii, eri, ndrbi,
                             ndnbi, swir12, ndwi, ndsi, guano_index1, guano_index2])
    return indices_list
