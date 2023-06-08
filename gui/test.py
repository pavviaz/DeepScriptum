from launcher import crop_pad_image
import os
from tqdm import tqdm
from PIL import Image
import numpy as np


PATH_FROM = "C:/users/shace/desktop/original_im2latex100k"
PATH_TO = "C:/users/shace/desktop/original_im2latex100k_cropped"


def rgba2rgb( rgba, background=(255,255,255) ):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros( (row, col, 3), dtype='float32' )
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]

    a = np.asarray( a, dtype='float32' ) / 255.0

    R, G, B = background

    rgb[:,:,0] = r * a + (1.0 - a) * R
    rgb[:,:,1] = g * a + (1.0 - a) * G
    rgb[:,:,2] = b * a + (1.0 - a) * B

    return np.asarray( rgb, dtype='uint8' )


for el in tqdm(os.listdir(PATH_FROM)):
    img = Image.open(os.path.join(PATH_FROM, el)).convert("RGBA")
    
    img = crop_pad_image(rgba2rgb(np.array(img)))
    
    img.save(os.path.join(PATH_TO, el))
    