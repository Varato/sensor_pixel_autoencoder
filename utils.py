from enum import Enum
import math

import cv2

def max_pool_output_size(w, k, p=0, s=None, d=1):
    s = k if s is None else s
    return (w + 2*p - d* (k-1) - 1) // s + 1

def transposed_conv_output_size(w, k, p=0, s=1, d=1):
    return (w - 1) * s - 2 * p + d * (k-1) + 1


class BayerPattern(Enum):
    RGGB = 0
    UNDEFINED = 5


def extract_bayer_channels(raw, bayer_pattern: BayerPattern=BayerPattern.RGGB):
    if bayer_pattern == BayerPattern.RGGB:
        r  = raw[..., 0::2, 0::2]
        gr = raw[..., 1::2, 0::2]
        gb = raw[..., 0::2, 1::2]
        b  = raw[..., 1::2, 1::2]
        return r, gr, gb, b
    else:
        raise ValueError(f"Unrecognoized BayerPattern {bayer_pattern.name}, {bayer_pattern.value}")


def image_rescale(img):
    """
    only 3840x1440 and 1920x1080 supported

    The img is first cropped to 1440x1440 or 1080x1080, and then rescaled to
    """
    output_size = math.gcd(1440 // 2, 1080 // 2)

    h, w = img.shape[-2:]
    if h == 1440 and w == 3840:
        pass
    elif h == 1080 and w == 1920:
        pass
    else:
        raise ValueError("only images of size 3840x1440 or 1920x1080 supported")

    r, gr, gb, b = extract_bayer_channels(img, bayer_pattern=BayerPattern.RGGB)
    img = (gr + gb) * 0.5
    h, w = h // 2, w // 2

    crop_start = (w - h) // 2
    img = img[..., crop_start:crop_start + h]
    img = cv2.resize(img, dsize=[448, 448], interpolation=cv2.INTER_CUBIC)
    return img
