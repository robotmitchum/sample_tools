# coding:utf-8
"""
    :module: df_utils.py
    :description: 2D distance functions to render and composite shapes

    Largely ported from Inigo Quilez's articles about distance functions :

    https://iquilezles.org/articles/distfunctions2d/

    :author: Michel 'Mitch' Pecqueur
    :date: 2025.04
"""

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter, distance_transform_edt, convolve1d

from common_math_utils import linstep


# Distance function

def point_df(shape: tuple[int, int],
             p: np.ndarray | tuple[float, float]) -> np.ndarray:
    """
    Return point distance function (circle)
    :param shape: width, height
    :param p: Point/Center position
    """
    sx, sy = shape
    y, x = np.ogrid[:sy, :sx]

    px, py = np.array(p)
    df = np.sqrt((x - px) ** 2 + (y - py) ** 2)

    return df


def segment_df(shape: tuple[int, int],
               a: np.ndarray | tuple[float, float],
               b: np.ndarray | tuple[float, float]) -> np.ndarray:
    """
    Return segment distance function (capsule)
    :param shape: width, height
    :param a: 1st point coordinates in pixels
    :param b: 2nd point coordinates in pixels
    :return:
    """
    sx, sy = shape
    y, x = np.ogrid[:sy, :sx]

    if np.linalg.norm(np.array(b) - np.array(a)) > 1e-3:
        uv = np.stack(np.broadcast_arrays(x, y), axis=-1)
        pa = uv - np.array(a)
        ba = np.array(b) - np.array(a)
        h = np.clip(np.dot(pa, ba) / np.dot(ba, ba), 0, 1)
        df = np.linalg.norm(pa - ba * np.stack((h, h), axis=-1), axis=-1)
    else:
        px, py = np.array(a)
        df = np.sqrt((x - px) ** 2 + (y - py) ** 2)

    return df


def arc_df(shape: tuple[int, int],
           p: np.ndarray | tuple[float, float],
           rot: float, angle: float,
           r: float) -> np.ndarray:
    """
    Return arc distance function
    :param shape: width, height
    :param p: Center position
    :param rot: Arc rotation in degrees
    :param angle: Arc aperture in degrees
    :param r: Arc radius
    """
    sx, sy = shape
    y, x = np.ogrid[:sy, :sx]
    uv = np.stack(np.broadcast_arrays(x, y), axis=-1).astype(float)

    # Transform coordinates
    uv = uv - np.array(p)
    rot_rad = np.radians(rot)
    rot_mat = np.array([[np.cos(rot_rad), np.sin(rot_rad)],
                        [-np.sin(rot_rad), np.cos(rot_rad)]])
    uv = uv @ rot_mat

    x, y = np.abs(uv[..., 0]), uv[..., 1]
    uv[..., 0] = x

    rad = np.radians(angle)
    sc = np.array((np.sin(rad), np.cos(rad)))
    s, c = sc

    df = np.where((c * x > s * y), np.abs(np.linalg.norm(uv, axis=-1) - r),
                  np.linalg.norm(uv - sc * r, axis=-1))

    return df


def box_sdf(shape: tuple[int, int],
            a: np.ndarray | tuple[float, float],
            b: np.ndarray | tuple[float, float]) -> np.ndarray:
    """
    Compute 2D signed distance field of a box
    :param shape: width, height
    :param a: Center of the box
    :param b: Half extents (box size / 2)
    """
    sx, sy = shape
    y, x = np.ogrid[:sy, :sx]
    uv = np.stack(np.broadcast_arrays(x, y), axis=-1)

    d = np.abs(uv - np.array(a)) - np.array(b)

    outside = np.linalg.norm(np.maximum(d, 0), axis=-1)
    inside = np.minimum(np.maximum(d[..., 0], d[..., 1]), 0)

    return outside + inside


def stroke_sdf(df: np.ndarray, r: float) -> np.ndarray:
    """
    Stroke shape border
    :param df: Input distance function
    :param r: Stroke radius
    """
    return abs(df) - r


def sdf_from_alpha(alpha: np.ndarray,
                   gamma_correct: bool = True,
                   distance: int | None = None,
                   scl: int = 4) -> np.ndarray:
    """
    Convert an alpha to a signed distance field

    :param alpha: Input anti-aliased alpha
    :param gamma_correct: Compensate srgb in input alpha
    :param distance: Normalization distance
    :param scl: Up-sampling factor
    :return: Resulting sdf
    """
    alpha_upsampled = np.repeat(np.repeat(alpha, scl, axis=0), scl, axis=1)

    kernel = 1 - np.abs(np.linspace(-1, 1, scl * 2)[1:-1])
    kernel /= kernel.sum()

    for i in range(2):
        alpha_upsampled = convolve1d(alpha_upsampled, kernel, axis=i, mode='nearest')

    if gamma_correct:
        th = lin_to_srgb(.5)
    else:
        th = .5

    inside = alpha_upsampled >= th
    outside = ~inside

    dist_inside = distance_transform_edt(inside)
    dist_outside = distance_transform_edt(outside)

    full_sdf = np.zeros_like(alpha_upsampled)
    full_sdf[outside] = dist_outside[outside]
    full_sdf[inside] = -dist_inside[inside]

    filtered = uniform_filter(full_sdf, size=scl, mode='nearest')
    offset = scl // 2
    result = filtered[offset::scl, offset::scl]

    d = (distance, 1)[distance is None]

    result = result / (d * scl)

    return result


def test_df(df: np.ndarray, ofs: float = 0, r: float = .5) -> np.ndarray:
    """
    Return anti-Aliased distance function testing
    :param df: Distance function
    :param ofs: Offset distance in pixels
    :param r: Smoothness radius
    """
    alpha = 1 - linstep(ofs - r, ofs + r, df)
    return alpha


# Compositing operations

def gamma_rgb(arr: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    tol = 0.5 / 255  # tolerance to prevent square root error
    result = np.copy(arr)
    result[..., :-1] = np.power(arr[..., :-1], np.where(arr[..., :-1] < tol, 1, 1.0 / gamma))
    return result


def srgb_to_lin(x: float):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(x: float):
    return np.where(x <= 0.00031308, 12.92 * x, 1.055 * x ** (1.0 / 2.4) - 0.055)


def grade_rgb(arr: np.ndarray,
              lift: tuple[float, float, float] | np.ndarray = (0, 0, 0),
              gain: tuple[float, float, float] | np.ndarray = (1, 1, 1)) -> np.ndarray:
    result = np.copy(arr)
    result[..., :-1] = (np.array(gain) - np.array(lift)) * result[..., :-1] + np.array(lift)
    return result


def fill_alpha(arr: np.ndarray, rgba: tuple[float, float, float, float] = (1, 1, 1, 1)) -> np.ndarray:
    """
    Fill alpha with rgba color, result will be pre-multiplied
    :param arr:
    :param rgba:
    """
    sx, sy = arr.shape
    result = arr.reshape(sx, sy, 1) * rgba[-1]
    result = np.repeat(result, 4, axis=-1) * np.array(list(rgba)[:-1] + [1])
    return result


def unpremult(arr: np.ndarray, th_alpha: bool = False) -> np.ndarray:
    """
    Convert image as straight/un-matted by dividing rgb by alpha
    :param arr: input rgba array
    :param th_alpha: Apply threshold to alpha, used by edge extension
    """
    sx, sy, nch = arr.shape
    tol = .5 / 255  # tolerance to prevent division by zero
    result = np.copy(arr)
    div = np.where(result[..., -1] < tol, 1, result[..., -1]).reshape(sx, sy, 1)
    result[..., :-1] = result[..., :-1] / div
    if th_alpha:
        result[..., -1] = np.where(result[..., -1] < tol, 0, 1)
    return result


def extend_edge(arr: np.ndarray, iterations: int = 3) -> np.ndarray:
    """
    Cheap edge extension, repeated blur and unpremult
    :param arr: rgba array
    :param iterations:
    """
    result = unpremult(arr, th_alpha=True)
    # Feedback loop
    for i in range(iterations):
        blurred = box_blur(result, size=3)
        unpr = unpremult(blurred, th_alpha=True)
        result = merge(result, unpr)
    result[..., -1] = np.copy(arr[..., -1])  # Restore original alpha
    return result


def merge(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Merge a over b using alpha from a
    :param a:
    :param b:
    """
    # Over mode
    a_alpha = np.expand_dims(a[..., -1], axis=-1)
    return a + b * (1 - a_alpha)


def box_blur(image, size=3):
    return uniform_filter(image, size=(size, size, 0), mode='nearest')


# Auxiliary definitions


def gen_uv(shape: tuple[int, int], nrm: bool = False) -> np.ndarray:
    """
    Generate UV array
    :param shape: width, height
    :param nrm: Normalize array
    :return:
    """
    sx, sy = shape
    y, x = np.ogrid[:sy, :sx]
    uv = np.stack(np.broadcast_arrays(x, y), axis=-1)
    if nrm:
        uv = uv.astype(np.float32) / (np.array(shape) - 1)
    return uv


def np_to_pil(arr: np.ndarray) -> Image:
    """
    Convert a 2D numpy array (up to 4 channels) to a PIL Image object
    """
    w, h, n = arr.shape
    pad = (3 - n, 0)[n > 3]
    mode = ('RGB', 'RGBA')[n > 3]
    if pad:
        img = np.pad(arr, ((0, 0), (0, 0), (0, pad)), 'constant', constant_values=0)
    else:
        img = arr
    im = Image.fromarray(np.round(np.clip(img * 255, 0, 255)).astype(np.uint8), mode=mode)
    return im
