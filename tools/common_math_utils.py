# coding:utf-8
"""
    :module: math_utils.py
    :description:
    :author: Michel 'Mitch' Pecqueur
    :date: 2024.08
"""
import hashlib

import numpy as np


def lerp(a, b, x):
    return a + (b - a) * x


def clamp(x, a=0, b=255):
    return min(max(x, a), b)


def h_cos(x):
    return .5 - .5 * np.cos(x * np.pi)


def equal_power_sine(x):
    """Equal power"""
    return np.sin(x * np.pi / 2)


def q_cos(x):
    """Equal power, quadratic approximation"""
    return x * (2 - x)


def q_log(x):
    return 1 - (1 - x) ** 4


def q_exp(x):
    return x ** 4


def smoothstep(a=0, b=1, x=.5):
    """
    Hermite interpolation
    :param float a: min value
    :param float b: max value
    :param float or np.array x: value to process
    :return: resulting percentage
    :rtype: float
    """

    w = (x - a) / (b - a)
    w = np.clip(w, 0.0, 1.0)
    return w * w * (3 - 2 * w)


def linstep(a, b, x):
    w = (x - a) / (b - a)
    # return min(max(w, 0), 1)
    return np.clip(w, 0.0, 1.0)


def array_to_seed(arr):
    """
    Generate integer seed from a numpy array
    :param np.array arr:
    :return:
    :rtype: int
    """
    hash_object = hashlib.sha256(arr.tobytes())
    hash_int = int(hash_object.hexdigest(), 16)
    return hash_int % (2 ** 32)


def random_from_string(name):
    """
    Generate pseudo-random value from a string
    :param str name:
    :return:
    :rtype: np.array
    """
    seed = int(hashlib.md5(name.encode()).hexdigest(), 16) % (2 ** 32)
    np.random.seed(seed)
    return np.random.random()
