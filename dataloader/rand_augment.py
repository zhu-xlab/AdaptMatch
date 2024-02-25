# code in this file is adpated from
# https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py
# https://github.com/google-research/fixmatch/blob/master/third_party/auto_augment/augmentations.py
# https://github.com/google-research/fixmatch/blob/master/libml/ctaugment.py
import random

import numpy as np
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image
import cv2
from .weak_augment import color_swap, hsv_shift, horizontal_flip, vertical_flip, rotate, resize_crop, hist_match
from .fft_augment import FFT_augment


PARAMETER_MAX = 10

def ColorSwap(img, **kwarg):
    order = [0,1,2]
    random.shuffle(order)
    channels = img.split()    
    r,g,b = channels[order[0]], channels[order[1]], channels[order[2]]
    img = Image.merge('RGB', (r,g,b))
    return img

def HsvShift(img, **kwarg):
    img = np.asarray(img)

    h_range, s_range, v_range = 20, 30, 20
    h_shift = random.randint(-h_range, h_range)
    s_shift = random.randint(-s_range, s_range)
    v_shift = random.randint(-v_range, v_range)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hue, sat, val =cv2.split(img)

    lut_hue = np.arange(0, 256, dtype=np.int16)
    lut_hue = np.mod(lut_hue+h_shift, 180).astype(np.uint8)
    hue = cv2.LUT(hue, lut_hue)

    lut_sat = np.arange(0, 256, dtype=np.int16)
    lut_sat = np.clip(lut_sat+s_shift, 0, 255).astype(np.uint8)
    sat = cv2.LUT(sat, lut_sat)

    lut_val = np.arange(0, 256, dtype=np.int16)
    lut_val = np.clip(lut_val+v_shift, 0, 255).astype(np.uint8)
    val = cv2.LUT(val, lut_val)

    img = cv2.merge((hue, sat, val)).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    img = Image.fromarray(img)
    return img

def AutoContrast(img, **kwarg):
    return PIL.ImageOps.autocontrast(img)

def Brightness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Brightness(img).enhance(v)

def Color(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Color(img).enhance(v)

def Contrast(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Contrast(img).enhance(v)

def Cutout(img, v, max_v, bias=0):
    if v == 0:
        return img
    v = _float_parameter(v, max_v) + bias
    v = int(v * min(img.size))
    return CutoutAbs(img, v)

def CutoutAbs(img, v, **kwarg):
    w, h = img.size
    x0 = np.random.uniform(0, w)
    y0 = np.random.uniform(0, h)
    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = int(min(w, x0 + v))
    y1 = int(min(h, y0 + v))
    xy = (x0, y0, x1, y1)
    # gray
    color = (127, 127, 127)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img

def Equalize(img, **kwarg):
    return PIL.ImageOps.equalize(img)

def Identity(img, **kwarg):
    return img

def Invert(img, **kwarg):
    return PIL.ImageOps.invert(img)

def Posterize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.posterize(img, v)

def Rotate(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.rotate(v)

def Sharpness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Sharpness(img).enhance(v)

def ShearX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))

def ShearY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))

def Solarize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.solarize(img, 256 - v)

def SolarizeAdd(img, v, max_v, bias=0, threshold=128):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    img_np = np.array(img).astype(np.int)
    img_np = img_np + v
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)

def TranslateX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[0])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))

def TranslateY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[1])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))

def _float_parameter(v, max_v):
    return float(v) * max_v / PARAMETER_MAX

def _int_parameter(v, max_v):
    return int(v * max_v / PARAMETER_MAX)


def fixmatch_augment_pool():
    # FixMatch paper
    augs = [(AutoContrast, None, None),
            (Brightness, 0.9, 0.05),
            (Color, 0.9, 0.05),
            (Contrast, 0.9, 0.05),
            (Equalize, None, None),
            (Identity, None, None),
            (Posterize, 4, 4),
            (Sharpness, 0.9, 0.05),
            (Solarize, 256, 0),
            ]
    return augs
    

class RandAugmentMC(object):
    def __init__(self, n, m):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_pool = fixmatch_augment_pool()

    def __call__(self, img):
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            v = np.random.randint(1, self.m)
            if random.random() < 0.5:
                img = op(img, v=v, max_v=max_v, bias=bias)
        img = CutoutAbs(img, 16)

        return img
