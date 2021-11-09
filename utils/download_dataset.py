import os
import re
import threading
import json
import time
import urllib.request
import requests
from tqdm import tqdm


FONT_SIZE = 120
SIZE = "small"


def web_translate(latex: str):
    """Transforms LaTeX caption into web form

    Args:
           latex: caption string in LaTeX
        Returns:
            web form of `latex`
    """
    res = ""
    for ch in latex:
        if not ch.isdigit() and ch not in "!()*-.ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz~'":
            res += "%" + format(ord(ch), "x").upper()
        else:
            res += ch
    return res.replace("%9", "%09")


def download_images(dct, path, start, end):
    """Downloads dataset images to `dir` folder

    Args:
           dct: dataset dictionary {name: caption}
           path: downloading folder path
           start: 'from' index
           end: 'to' index
    """
    dict_temp = dict(list(dct.items())[start:end])
    for k, v in tqdm(dict_temp.items()):
        res = web_translate(v)
        try:
            urllib.request.urlretrieve(
                f"https://latex.codecogs.com/png.download?%5Cdpi%7B{FONT_SIZE}%7D%20%5Cbg_white%20%5C{SIZE}%20{res}",
                path + k + ".png")
            # time.sleep(0.25)
        except:
            # print()
            # print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
            # print('/'.join(check_path(path).split('/')[:-2]))
            with open(check_path('/'.join(check_path(path).split('/')[:-2])) + "errors.txt", "a+") as f:
                f.write(f"\nkey - {k}  val = {v}  res = {res}\n")


def compute_threads_work(length, download_threads):
    """Yields split data for threads

    Args:
           length: number of captions in dataset
           download_threads: number of downloading threads
        Returns:
            (from, to) tuple
    """
    amount = length // download_threads
    while length > 0:
        tmp = length
        length = 0 if length - amount < 0 else length - amount
        yield (length, tmp)


def check_path(path):
    """Checks if given str is path

    Args:
           path: path string
        Returns:
            path or path\\
    """
    return path + "\\" if path[-1] != "\\" else path


def download_dataset(directory: str, caption_path: str, amount_prop: float = 1.0, download_threads: int = 10):
    """Downloads dataset images into `directory` folder using caption JSON file on `caption_path`

    Args:
            directory: path string to downloading directory
            caption_path: path string to JSON caption file
            amount_prop: dataset length control float (final_len = amount_prop * len())
            download_threads: number of downloading threads (6 by default, [1 <= download_threads <= 12])
    """
    os.makedirs(os.path.dirname(check_path(directory)), exist_ok=True)
    dataset_dct = {}
    try:
        with open(caption_path, 'r+') as file:
            file_data = json.load(file)
            capt = file_data["annotations"]
            for cap in capt:
                dataset_dct[cap["image_id"]] = cap["caption"]
    except:
        raise IOError("Caption file reading error")

    dataset_dct = dict(list(dataset_dct.items())[
                       :int(amount_prop * len(dataset_dct))])
    threads = [threading.Thread(target=download_images, args=(dataset_dct, directory) + arg) for arg in
               compute_threads_work(len(dataset_dct), download_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


# with open("C:/Users/shace/Desktop/errors.txt", "r") as f:
#     for line in f:
#         line = line.replace("\n", "")
#         if line != "":
#             key = line[line.find("key - "):line.find("val = ")].replace("key - ", "").replace(" ", "")
#             res = line[line.find("res = "):].replace("res = ", "").replace(" ", "")
#             try:
#                 urllib.request.urlretrieve(
#                 f"https://latex.codecogs.com/png.download?%5Cdpi%7B{FONT_SIZE}%7D%20%5Cbg_white%20%5C{SIZE}%20{res}",
#                 "C:/Users/shace/Desktop/lol/" + key + ".png")
#             #time.sleep(0.25)
#             except:
#             # print()
#                 print(f"WUT??? k = {key}     ;     r == {res}")
#             # print('/'.join(check_path(path).split('/')[:-2]))


# f = set(os.listdir("C:/Users/shace/Desktop/lol/"))
# s = set(os.listdir("C:\\Users\\shace\\Documents\\GitHub\\im2latex\\datasets\\images_300"))
# print(s- f)


# download_dataset("C:/Users/shace/Desktop/lol/", "C:\\Users\\shace\\Documents\\GitHub\\im2latex\\dataset\\dataset.json")
