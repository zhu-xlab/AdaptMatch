import random
from skimage.exposure import match_histograms
import numpy as np
import PIL
from PIL import Image
import cv2



def color_swap(img, prob=0.5):
    if random.random() < prob:
        order = [0,1,2]
        random.shuffle(order)
        channels = img.split()    
        r,g,b = channels[order[0]], channels[order[1]], channels[order[2]]
        img = Image.merge('RGB', (r,g,b))
    return img


def hsv_shift(img, prob=0.5):
    if random.random() < prob:
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

def horizontal_flip(image, label, prob=0.5):
    if random.random() < prob:
        image = image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        label = label.transpose(PIL.Image.FLIP_LEFT_RIGHT)

    return image, label

def vertical_flip(image, label, prob=0.5):
    if random.random() < prob:
        image = image.transpose(PIL.Image.FLIP_TOP_BOTTOM)
        label = label.transpose(PIL.Image.FLIP_TOP_BOTTOM)

    return image, label

def rotate(image, label):
    angle = random.choice([90, 180, 270, 360])
    image = image.rotate(angle, expand=True)
    label = label.rotate(angle, expand=True)

    return image, label

def resize_crop(image, label, max_size):
    scale = random.uniform(0.5, 2)
    width, height = image.size 
    new_width, new_height = int(width*scale), int(height*scale)

    # resize
    image = image.resize((new_width, new_height), resample=PIL.Image.BILINEAR)
    label = label.resize((new_width, new_height), resample=PIL.Image.NEAREST)
    image, label = np.asarray(image), np.asarray(label)

    if new_width < width:
        image_pad_width = np.zeros((width-new_width, new_height, 3))
        image_pad_height = np.zeros((width, height-new_height, 3))
        label_pad_width = np.zeros((width-new_width, new_height))
        label_pad_height = np.zeros((width, height-new_height))
        image = np.concatenate((image, image_pad_width), axis=0)
        image = np.concatenate((image, image_pad_height), axis=1)
        label = np.concatenate((label, label_pad_width), axis=0)
        label = np.concatenate((label, label_pad_height), axis=1)

    # crop
    [max_width, max_height] = max_size
    width, height = min(width, max_width), min(height, max_height)
    w_str = random.randint(0, max(new_width-width, 0))
    w_end = w_str + width
    h_str = random.randint(0, max(new_height-height, 0))
    h_end = h_str + height

    image = image[w_str:w_end, h_str:h_end,:]
    label = label[w_str:w_end, h_str:h_end]
    image = np.uint8(image)
    label = np.uint8(label)

    image = Image.fromarray(image)
    label = Image.fromarray(label)

    return image, label


def hist_match(self, img_A, img_B):
    if random.random() < prob:
        img_A = match_histograms(img_A, img_B, multichannel=True)

    return A
