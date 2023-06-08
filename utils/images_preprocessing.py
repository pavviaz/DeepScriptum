import random
from PIL import Image
import numpy as np
import math
from PIL import ImageEnhance
from matplotlib import pyplot as plt

# RESIZE_W = 1500
# RESIZE_H = 175
# RESIZE_W = 940
# RESIZE_H = 110


# taken from
def crop_pad_image(img, output_path):
    """Crops image to content
    Args:
        img: (string) path to image
        output_path: (string) path to output image
    """
    old_im = Image.open(img).convert('L')
    img_data = np.asarray(old_im, dtype=np.uint8)  # height, width
    nnz_inds = np.where(img_data != 255)
    if len(nnz_inds[0]) == 0:
        raise IOError("Input image is blank!")

    y_min = np.min(nnz_inds[0])
    y_max = np.max(nnz_inds[0])
    x_min = np.min(nnz_inds[1])
    x_max = np.max(nnz_inds[1])
    old_im = old_im.crop((x_min, y_min, x_max + 1, y_max + 1))
    return old_im
    # old_im.save(output_path)


def make_fix_size(numpy_image, RESIZE_W, RESIZE_H, random_resize: bool, alpha: bool = False):

    img = Image.fromarray(np.uint8(numpy_image)).convert('RGB')

    if img.size[0] <= RESIZE_W and img.size[1] <= RESIZE_H:
        if not alpha:
            background = Image.new('RGB', (RESIZE_W, RESIZE_H), (255, 255, 255))
        else:
            background = Image.new('RGBA', (RESIZE_W, RESIZE_H), (255, 255, 255, 255))

        if random_resize: #  and random.randint(0, 1) == 0:
            a = 0.55 * min(math.floor(RESIZE_W / img.size[0]), math.floor(RESIZE_H / img.size[1]))
            # 
            coeff = random.uniform(0.8, a if a > 0.8 else 1)
            img = img.resize((int(img.size[0] * coeff), int(img.size[1] * coeff)))
        
        background.paste(img, (int(RESIZE_W / 2) - int(img.size[0] / 2), int(RESIZE_H / 2) - int(img.size[1] / 2)))
        return np.array(background)
        
    else:
        # return None
        print(f"image is too big! w = {img.size[0]} ; h = {img.size[1]}")  # This can't happen
        

def compile_handwritten_image(numpy_background, numpy_formula):
    backgr = Image.fromarray(np.uint8(numpy_background)).convert('RGBA')
    formula = Image.fromarray(np.uint8(numpy_formula)).convert('RGBA')
        
    resize_coeff = random.choice(np.arange(1/3, min(backgr.size[0] / formula.size[0], backgr.size[1] / formula.size[1])))
    
    formula = formula.resize((int(formula.size[0] * resize_coeff), int(formula.size[1] * resize_coeff)))
    
    # formula.save("1.png")
    
    backgr.paste(formula, (int(backgr.size[0] / 2) - int(formula.size[0] / 2), int(backgr.size[1] / 2) - int(formula.size[1] / 2)), formula)
    
    # backgr.save("1.png")
    
    return np.array(backgr.convert("RGB"))