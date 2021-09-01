import os
import threading
from PIL import Image
import json
import urllib.request
from tqdm import tqdm
import random
import numpy as np
import cv2

CAPTION_PATH = "C:\\Users\\shace\\Documents\\GitHub\\im2latex\\Untitled-3.json"


dct = {}
with open(CAPTION_PATH, 'r+') as file:
    file_data = json.load(file)
    capt = file_data["annotations"]
    for cap in capt:
        dct[cap["image_id"]] = cap["caption"]


def codecogs_translate(latex: str):
    res = ""
    for ch in latex:
         if not ch.isdigit() and not ch in "!()*-.ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz~'":
            res += "%" + format(ord(ch), "x").upper()
         else:
            res += ch
    return res.replace("%9", "%09")


def func_5(start, end):  # download latex images
    dict_temp = dict(list(dct.items())[start:end])
    for k,v in dict_temp.items():
        res = codecogs_translate(v)
        try:
            urllib.request.urlretrieve(f"https://latex.codecogs.com/png.download?%5Cdpi%7B100%7D%20%5Cbg_white%20%5Clarge%20{res}", 'C:/Users/shace/Desktop/formula_images_png_4/' + k + ".png")
        except:
            with open("C:/Users/shace/Desktop/errors.txt", "a")  as f:
                f.write(f"\nkey - {k}  val = {v}  res = {res}\n")


def func_7(limit): # find exact symbol in math image using cv2

    file = []
    capt_1 = os.listdir("C:\\Users\\shace\\Desktop\\conv_latex_exp\\")
    with open("C:\\Users\\shace\\Desktop\\123.txt", "r") as f:
        for line in f:
            file.append(line.replace("\n", ""))

    capt_1 = [cap for cap in file if encode(cap) in os.listdir("C:\\Users\\shace\\Desktop\\conv_latex_exp\\")]
    #capt_1 = ["("]
    capt = []
    for line in capt_1:
        line = line.replace("\n", "").replace("\\", "_sl_").replace("/", "_revsl_").replace(":", "_cln_").replace("*", "_mlt_").replace("?", "_quest_").replace("<", "_less_").replace(">", "_grtr_").replace("|", "_strght_").replace(".", "_dt_").replace('"', "_dbqouts_")
        for ch in line:
            if ch.isupper() and ch.isalpha():
                line = line.replace(ch, "_upper_" + ch.lower(), 1) 
        capt.append(line)
    
    for cap in capt_1:
        print("now - " + cap)
        tr = codecogs_translate(cap)
        try:
            urllib.request.urlretrieve(f"https://latex.codecogs.com/png.download?%5Cdpi%7B300%7D%20%5Cbg_white%20%5Clarge%20{tr}", 'temp.png')
        except:
             with open("C:/Users/shace/Desktop/errors.txt", "a") as f:
                print("error!")
                f.write(f"\ntr - {tr}  cap - {cap}\n")
                continue

        work_list = [k for k,v in dct.items() if cap in v]
        i = 0
        for dir in tqdm(work_list):
   
            if i == limit:
                break
            f = [0.5, 1, 1.5, 2]
            fx = f[random.randint(0, len(f) - 1)]
            method = cv2.TM_CCOEFF_NORMED

            # Read the images from the file
            small_image = cv2.imread('temp.png')
            large_image = cv2.imread(f'C:\\Users\\shace\\Desktop\\formula_images_png_3\\{dir}.png')

            try:
                result = cv2.matchTemplate(small_image, large_image, method)
            except:
                continue

            coef = random.randint(25, 35) / 100
            threshold = .8
            loc = np.where(result >= threshold)
            w, h = small_image.shape[:-1]
            #print(f"w = {w} ; h = {h}")
            for pt in zip(*loc[::-1]):  # Switch collumns and rows
                #print(f"pt0 = {pt[0]} ; pt1 = {pt[1]}") 
                #cv2.rectangle(large_image, pt, (pt[0] + h, pt[1] + w), (0, 0, 255), 2)
                croped = large_image[pt[1] - int(h * coef):pt[1] + w + int(h*coef), pt[0] - int(w * coef):pt[0] + h + int(w * coef)]
                #print(f"cr shape = {croped.shape}")
                # try:
                #     cv2.imshow('1', croped)
                #     cv2.waitKey(0)
                # except:
                #     pass
                if croped.shape[0] > 0.6 * h:
                    try:
                        #print("YEP - " + str(i))
                        final = cv2.resize(croped, (0,0), fx = fx, fy= fx)
                        cv2.imwrite("C:\\Users\\shace\\Desktop\\conv_latex_exp\\" + capt[capt_1.index(cap)] + f"\\{i}.png", final)
                        
                        i += 1
                        break
                    except:
                        continue

            
            # # We want the minimum squared difference
            # mn,_,mnLoc,_ = cv2.minMaxLoc(result)

            # # Draw the rectangle:
            # # Extract the coordinates of our best match
            # MPx,MPy = mnLoc

            # # Step 2: Get the size of the template. This is the same size as the match.
            # trows,tcols = small_image.shape[:2]

            #cv2.rectangle(large_image, (MPx,MPy),(MPx+tcols,MPy+trows),(0,0,255),2)

            # cv2.imshow('output',large_image)

            # cv2.waitKey(0)

            # coef = random.randint(1, 3) / 10
            # u_x, u_y, d_x, d_y = int(coef * 0.3 *  MPx), int(coef * MPy), int(coef * 0.3 * (MPx + tcols)), int(coef * 0.7 * (MPy + trows))

            # u_x, u_y, d_x, d_y = int(0.5 *  (tcols)), int(0.1 * (trows)), int(0.5 *  (tcols)), int(0.1 * (trows))

            # croped = large_image[MPy - u_y:MPy + trows + d_y, MPx - u_x:MPx + tcols + d_x]
            
            # try:
            #     final = cv2.resize(croped, (0,0), fx = fx, fy= fx)
            #     cv2.imwrite("C:\\Users\\shace\\Desktop\\conv_latex_exp\\" + capt[capt_1.index(cap)] + f"\\{i}.png", final)
            #     i += 1
            # except:
            #     pass

            # cv2.imshow('output',final)

            # cv2.waitKey(0)
    

# x = [threading.Thread(target=func_5, args=(0, 7452)),
#      threading.Thread(target=func_5, args=(7452, 14904)),
#      threading.Thread(target=func_5, args=(14904, 22356)),
#      threading.Thread(target=func_5, args=(22356, 29808)),
#      threading.Thread(target=func_5, args=(29808, 37260)),
#      threading.Thread(target=func_5, args=(37260, 44718))]
# for thread in x:
#     thread.start()
# for thread in x:
#     thread.join()


