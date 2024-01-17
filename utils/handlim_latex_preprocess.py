import os
import re
import ast
import json
import multiprocessing
from io import BytesIO

import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw

from multiprocess_funcs import compute_threads_work


# w_max = 5340 ; h_max = 1197
MIN_W = 1780  # 1/3 of w_max
MIN_H = 399  # 1/3 of h_max

MIN_W = 1800  # 1/3 of w_max
MIN_H = 410  # 1/3 of h_max

BACKGR = "handlim_backgrounds"
FORMULAS = "handlim_formulas"


def split_data(string: str):
    string = string.replace("\n", "").replace("\t", " ").replace("\r", "")
    res_temp = ""
    res = ""
    r = re.findall(r"[\\]([a-zA-Z]+|\D)", string)
    r = ["\\" + str(reg) for reg in r]
    while len(r) != 0:
        lst = string.split(str(r[0]))
        if lst[0] != "":
            for char in lst[0]:
                res_temp += char + " "
        res_temp += str(r[0]) + " "

        string = string[string.index(str(r[0])) + len(str(r[0])) :]

        r.remove(r[0])

        for space in re.findall(r"[ ]+", res_temp):
            res_temp = res_temp.replace(str(space), " ")

        res_temp = res_temp.replace("' '", "''")

        res += res_temp
        res_temp = ""
    if string.replace(res.replace(" ", ""), "") != "":
        for char in string.replace(res.replace(" ", ""), ""):
            res += char + " "

    return res


def convert_string_to_bytes(string):
    """
    Converts a string representation of bytes back to bytes.

    Parameters
    ----------
    string : str
        A string of a bytes-like object.

    Returns
    ----------
    bytes : bytes
        bytes (i.e 'b\'\\x89PNG\\r\\n ... ' -> '\\x89PNG\\r\\n ...').

    """
    return ast.literal_eval(string)


def unpack_list_of_masks(string_list):
    """
    Unpacks a list of string represented bytes objects and returns a list of bytes objects.

    Parameters
    ----------
    string_list : list
        A list of string representation bytes-like objects

    Returns
    ----------
    mask_bytes: list
        A list of png masks as bytes.
    """
    return [convert_string_to_bytes(string) for string in string_list]


def inpainting(im, mask):
    im = im.convert("RGB")
    pix = im.load()

    # mask = mask.convert("L")
    mask = Image.new("L", im.size)
    maskdraw = ImageDraw.Draw(mask)
    for x in range(im.size[0]):
        for y in range(im.size[1]):
            if pix[(x, y)] == (0, 0, 0):
                maskdraw.point((x, y), 255)

    # convert image and mask to opencv format
    im = np.array(im)[:, :, ::-1].copy()
    mask = np.array(mask)[:, :, np.newaxis].copy()

    dst = cv2.inpaint(im, mask, 5, cv2.INPAINT_NS)

    im = Image.fromarray(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
    return im


def convert_mask_bytes_to_rgba_color_scheme(mask_bytes, img_path):
    """
    This function makes the png-masks transparent everywhere except the
    masked area of interest, enhancing the visualization.
    PNG masks are not inherently RGBA, this function
    adds the fourth depth of 'alpha' or transparency.

    Parameters
    ----------
    mask_bytes : list
        A png masks as bytes.
    label :
        The png mask's corresponding label. (Used for coloring the mask)

    Returns
    ----------
    mask: PIL.Image.Image
        An RGBA Image object representing the mask of interest as RGBA, appropriately colored.
    """
    # Open the mask.
    img = Image.open(os.path.join("handlim", img_path))
    res = None
    for mask_byte in mask_bytes:
        mask = Image.open(BytesIO(mask_byte))
        mask = mask.convert("RGBA")

        datas = mask.getdata()

        newData = []
        for item in datas:
            if item[0] == 0 and item[1] == 0 and item[2] == 0:
                newData.append((255, 255, 255, 0))
            else:
                newData.append(item)
        im2 = Image.new(mask.mode, mask.size)
        im2.putdata(newData)

        if not res:
            res = im2
            continue
        res.paste(im2, (0, 0), im2)

    # mask_pxl = res.getdata()  # x, y
    ext_formula = Image.new(mask.mode, mask.size)

    for x in range(mask.size[0]):
        for y in range(mask.size[1]):
            if res.getpixel((x, y))[-1] != 0:
                ext_formula.putpixel((x, y), img.getpixel((x, y)))

    for x in range(mask.size[0]):
        for y in range(mask.size[1]):
            if res.getpixel((x, y))[-1] != 0:
                for i in range(1, 4):
                    try:
                        img.putpixel((x, y), (0, 0, 0, 0))
                        img.putpixel((x + i, y), (0, 0, 0, 0))
                        img.putpixel((x - i, y), (0, 0, 0, 0))
                        img.putpixel((x, y + i), (0, 0, 0, 0))
                        img.putpixel((x, y - i), (0, 0, 0, 0))
                        img.putpixel((x + i, y + i), (0, 0, 0, 0))
                        img.putpixel((x - i, y + i), (0, 0, 0, 0))
                        img.putpixel((x - i, y - i), (0, 0, 0, 0))
                        img.putpixel((x + i, y - i), (0, 0, 0, 0))
                    except:
                        pass

    background = inpainting(img, res)
    background = background.resize((MIN_W, MIN_H))

    # img.show()


def preprocess_and_save(dct, start, end):
    dict_temp = dct[start:end]
    for img in tqdm(dict_temp):
        convert_mask_bytes_to_rgba_color_scheme(
            unpack_list_of_masks(img["image_data"]["png_masks"]), img["filename"]
        )


def main():
    for i in range(1, 11, 1):
        with open(
            f"datasets/handwritten_to_latex (limits)/batch_{i}/JSON/kaggle_data_{i}.json",
            "r",
        ) as f:
            data = json.load(f)

        download_threads = 16
        threads = [
            multiprocessing.Process(
                target=preprocess_and_save, args=(data, arg[0], arg[1])
            )
            for arg in compute_threads_work(len(data), download_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()


if __name__ == "__main__":
    main()
