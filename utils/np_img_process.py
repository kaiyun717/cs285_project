import cv2
import numpy as np

def opencv_downsample(image: np.ndarray, out_size: tuple):
    resized = cv2.resize(image, dsize=out_size, interpolation=cv2.INTER_CUBIC)
    return resized

def rgb2gray(rgb: np.ndarray):
    if len(rgb.shape) == 3:
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    elif len(rgb.shape) == 2:
        gray = rgb
    else:
        raise Exception('Image needs to be either rgb or gray')
    
    return gray