import os
from tqdm import tqdm
from PIL import Image


def width_height_info(path):
    '''
    Get some info about images in directory
    '''
    dir = os.listdir(path)
    w = []
    h = []
    w_m = 0
    h_m = 0

    for d in tqdm(dir):
        img = Image.open(path + d)
        w.append(img.size[0])
        h.append(img.size[1])
        w_m += img.size[0]
        h_m += img.size[1]
    print(f"w_m = {w_m} ; h_m = {h_m}\nw_m = {w_m / len(dir)} ; h_m = {h_m / len(dir)}\nw_max = {max(w)} ; h_max = {max(h)}")


def resize(image_pil, width, height):
    '''
    Resize PIL image keeping ratio and using white background.
    '''
    ratio_w = width / image_pil.width
    ratio_h = height / image_pil.height
    if ratio_w < ratio_h:
        # It must be fixed by width
        resize_width = width
        resize_height = round(ratio_w * image_pil.height)
    else:
        # Fixed by height
        resize_width = round(ratio_h * image_pil.width)
        resize_height = height
    image_resize = image_pil.resize((resize_width, resize_height), Image.BILINEAR)
    background = Image.new('RGBA', (width, height), (255, 255, 255, 255))
    offset = (round((width - resize_width) / 2), round((height - resize_height) / 2))
    background.paste(image_resize, offset)
    return background.convert('RGB')


width_height_info("C:\\Users\\shace\\Documents\\GitHub\\im2latex\\datasets\\handlim\\")