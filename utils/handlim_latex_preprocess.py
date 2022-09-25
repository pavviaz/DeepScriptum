import ast
from io import BytesIO
import json
import re
import threading
import PIL
import numpy as np
from PIL import Image, ImageDraw
import cv2
from download_dataset import compute_threads_work
from tqdm import tqdm
import multiprocessing

# w_max = 5340 ; h_max = 1197
MIN_W = 1780  # 1/3 of w_max
MIN_H = 399  # 1/3 of h_max

MIN_W = 1800  # 1/3 of w_max
MIN_H = 410  # 1/3 of h_max

BACKGR = "C:\\Users\\shace\\Documents\\GitHub\\im2latex\\datasets\\handlim_backgrounds\\"
FORMULAS = "C:\\Users\\shace\\Documents\\GitHub\\im2latex\\datasets\\handlim_formulas\\"


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

        string = string[string.index(str(r[0])) + len(str(r[0])):]

        r.remove(r[0])

        for space in re.findall(r"[ ]+", res_temp):
            res_temp = res_temp.replace(str(space), " ")

        res_temp = res_temp.replace("' '", "''")
        # res_temp = res_temp.replace(" . ", ".")

        res += res_temp
        res_temp = ""
    if string.replace(res.replace(" ", ""), "") != "":
        for char in string.replace(res.replace(" ", ""), ""):
            res += char + " "

    return res


def make_json():
    with open("C:\\Users\\shace\\Documents\\GitHub\\im2latex\\5_dataset_large.json", "r") as file:
        s = set()
        dataset = json.load(file)["annotations"]
        for cap in dataset:
            [s.add(token) for token in list(filter(lambda x: len(x) != 0, cap["caption"].split(" ")))]


    with open("D:\\datasets\\handwritten_to_latex (limits)\\batch_10\\JSON\\kaggle_data_10.json", "r") as f, open("C:\\Users\\shace\\Documents\\GitHub\\im2latex\\handlim_dataset_full.json", "r+") as f1:
        data = json.load(f)
        l = []
        for img in data:
            cap = list(filter(lambda x:len(x) != 0, split_data(img["latex"]).split(" ")))
            l.append({"image_id": str(img["uuid"]), "caption": split_data(img["latex"])})
        dat = json.load(f1)
        dat["annotations"].extend(l)
        json.dump(dat, f1, indent=4)
    

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


# def inpainting(im):
#     pix = im.load()

#     #create a mask of the background colors
#     # this is slow, but easy for example purposes
#     mask = Image.new('L', im.size)
#     maskdraw = ImageDraw.Draw(mask)
#     for x in range(im.size[0]):
#         for y in range(im.size[1]):
#             if pix[(x,y)] == (0,0,0):
#                 maskdraw.point((x,y), 255)

#     #convert image and mask to opencv format
#     cv_im = cv2.CreateImageHeader(im.size, cv2.IPL_DEPTH_8U, 3)
#     cv2.cv.SetData(cv_im, im.tostring())
#     cv_mask = cv2.cv.CreateImageHeader(mask.size, cv2.cv.IPL_DEPTH_8U, 1)
#     cv2.cv.SetData(cv_mask, mask.tostring())
    
#     #do the inpainting
#     cv_painted_im = cv2.cv.CloneImage(cv_im)
#     cv2.cv.Inpaint(cv_im, cv_mask, cv_painted_im, 3, cv2.cv.CV_INPAINT_NS)

#     #convert back to PIL
#     painted_im = Image.fromstring("RGB", cv2.cv.GetSize(cv_painted_im), cv_painted_im.tostring())
#     painted_im.show()
    

def inpainting(im, mask):
    im = im.convert("RGB")
    pix = im.load()
    
    # for x in range(mask.size[0]):
    #     for y in range(mask.size[1]):
    #         if mask.getpixel((x, y))[-1] != 0:
    #             ext_formula.putpixel((x, y), img.getpixel((x, y)))
    #             img.putpixel((x, y), (0,0,0,0))
    
    # mask = mask.convert("L")
    mask = Image.new('L', im.size)
    maskdraw = ImageDraw.Draw(mask)
    for x in range(im.size[0]):
        for y in range(im.size[1]):
            if pix[(x,y)] == (0,0,0):
                maskdraw.point((x,y), 255)
    
    # mask = PIL.ImageOps.invert(mask)
    # im.show()
    # maskdraw.show()

    #create a mask of the background colors
    # this is slow, but easy for example purposes
    # mask = Image.new('L', im.size)
    # maskdraw = ImageDraw.Draw(mask)
    # for x in range(im.size[0]):
    #     for y in range(im.size[1]):
    #         if pix[(x,y)] == (0,0,0):
    #             maskdraw.point((x,y), 255)

    #convert image and mask to opencv format
    im = np.array(im)[:, :, ::-1].copy() 
    mask = np.array(mask)[:, :, np.newaxis].copy() 
    
    
    dst = cv2.inpaint(im, mask, 5, cv2.INPAINT_NS)

    im = Image.fromarray(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
    return im
    



def convert_mask_bytes_to_rgba_color_scheme(mask_bytes, img_path):
    """
    This function makes the png-masks transparent everywhere except the masked area of interest, enhancing the visualization. PNG masks are not inherently RGBA, this function adds the fourth depth of 'alpha' or transparency.
    
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
    img = Image.open("C:\\users\\shace\\documents\\github\\im2latex\\datasets\\handlim\\" + img_path)
    res = None
    for mask_byte in mask_bytes:
        mask = Image.open(BytesIO(mask_byte))
        mask = mask.convert("RGBA")
        # mask = PIL.ImageOps.invert(mask)
        
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
        # res.show()
    # res.show()
    
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
                        img.putpixel((x, y), (0,0,0,0))
                        img.putpixel((x+i, y), (0,0,0,0))
                        img.putpixel((x-i, y), (0,0,0,0))
                        img.putpixel((x, y+i), (0,0,0,0))
                        img.putpixel((x, y-i), (0,0,0,0))
                        img.putpixel((x+i, y+i), (0,0,0,0))
                        img.putpixel((x-i, y+i), (0,0,0,0))
                        img.putpixel((x-i, y-i), (0,0,0,0))
                        img.putpixel((x+i, y-i), (0,0,0,0))
                    except:
                        pass
    
    # ext_formula.show()
    # ext_formula.save("123.png")
    background = inpainting(img, res)
    background = background.resize((MIN_W, MIN_H))
    # background.save(BACKGR + img_path.split(".")[0] + ".png")
    # ext_formula.save(FORMULAS + img_path.split(".")[0] + ".png")
    # ext_formula = ext_formula.resize((int(ext_formula.size[0] * 1/3), int(ext_formula.size[1] * 1/3)))
    # background.paste(ext_formula, (int(MIN_W / 2) - int(ext_formula.size[0] / 2), int(MIN_H / 2) - int(ext_formula.size[1] / 2)), ext_formula)
    # background.show(f"{img_path}")
    # img.paste(res, (0, 0), res)
    # img.show()

def preprocess_and_save(dct, start, end):
    dict_temp = dct[start:end]
    for img in tqdm(dict_temp):
        convert_mask_bytes_to_rgba_color_scheme(unpack_list_of_masks(img["image_data"]["png_masks"]), img["filename"])


def main():
    for i in range(1, 11, 1):
        with open(f"D:\\datasets\\handwritten_to_latex (limits)\\batch_{i}\\JSON\\kaggle_data_{i}.json", "r") as f:
            data = json.load(f)
            
        
        download_threads = 16
        threads = [multiprocessing.Process(target=preprocess_and_save, args=(data, arg[0], arg[1])) for arg in
                compute_threads_work(len(data), download_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()


if __name__ == "__main__":
    main()