from os import path, replace
import os
import re
import json


def load_latex_from_txt(path="C:\\Users\\shace\\Desktop\\temp\\math.txt", blacklist_path="C:\\Users\\shace\\Desktop\\black.txt"):

    blacklist = []
    with open(blacklist_path, "r+") as file:
        for line in file:
            blacklist.append(line.replace("\n", ""))
    accepted = []
    with open(path, 'r+') as file:
        for line in file:
            if not any(substring in line for substring in blacklist):
                accepted.append(line)
    print(f"top 10 caps : {accepted[:10]}\nlen = {len(accepted)}")
    with open("C:\\Users\\shace\\Desktop\\temp\\math2.txt", "a") as file:
        for line in accepted:
            file.write(split_data(line) + "\n")



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

    # res = res.replace("c m", "cm")
    # res = res.replace("m m", "mm")
    # res = res.replace("i n", "in")
    # res = res.replace(" a r r a y ", "array")
    # res = res.replace(" c a s e s ", "cases")
    # res = res.replace(" m a t r i x ", "matrix")
    # res = res.replace(" p i c t u r e ", "picture")
    # res = res.replace("pmatrix", "pmatrix ")
    return res

def update_json():
    path = "C:\\Users\\shace\\Documents\\GitHub\\im2latex\\datasets\\formula_images_png_4_large_resized\\"
    dir = os.listdir(path)
    ddir = []
    for el in dir:
        ddir.append(el.split(".")[0])

    with open("C:/Users/shace/Documents/GitHub/im2latex/4_dataset_large.json", 'r+') as file: # updating dat6aset JSON file
        list_upd = []
        file_data = json.load(file)
        capt = file_data["annotations"]
        print(len(capt))
        for cap in capt:
            # if not any(substring in cap["caption"] for substring in ["hskip", "hspace", "put", "rule"]):
            #     cap["caption"] = split_data(cap["caption"])
            #     list_upd.append(cap)
            # try:
            #     cap["caption"] = cap["caption"].replace(str(re.findall(r"[\\]label[ ]+", cap["caption"])[0]) + str(re.findall(r"[\\]label[ ]+([^{]|{[^}]+})", cap["caption"])[0]) + " ", "")
            # except:
            #     pass

            # if not cap["image_id"] in errors:
            #     list_upd.append(cap)

            # if not "\\\\" in cap["caption"]:
            #     list_upd.append(cap)

            # if not any(substring in cap["caption"] for substring in ["choose"]):
            #     list_upd.append(cap)

            if cap["image_id"] in ddir:
                list_upd.append(cap)

        print(len(list_upd))
        file_data["annotations"] = list_upd
        # convert back to json.

        json.dump(file_data, file, indent=4)


def update_json_2(path_json="C:/Users/shace/Documents/GitHub/im2latex/new_dataset.json", path_txt="C:\\Users\\shace\\Desktop\\temp\\math2.txt"):
    with open(path_json, 'r+') as file: # updating dat6aset JSON file
        list_upd = []
        file_data = json.load(file)
        capt = file_data["annotations"]
        print(f"before = {len(capt)}")
        i = 1
        with open(path_txt, "r") as txt:
            for line in txt:
                list_upd.append({"image_id":str(i), "caption":line.replace("\n", "")})
                i += 1

        print(len(list_upd))
        file_data["annotations"] = list_upd
        # convert back to json.
        json.dump(file_data, file, indent=4)

#load_latex_from_txt()
update_json()