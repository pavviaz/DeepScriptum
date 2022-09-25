import re
import json


def get_dataset_tokens(d_path="C:\\Users\\shace\\Documents\\GitHub\\im2latex\\5_dataset_large.json"):
    with open(d_path, "r") as file:
        s = set()
        dataset = json.load(file)["annotations"]
        for cap in dataset:
            [s.add(token) for token in list(filter(lambda x: len(x) != 0, cap["caption"].split(" ")))]
        # with open("C:\\Users\\shace\\Desktop\\temp\\black.txt", "w+") as f:
        #     [f.write(sym + "\n") for sym in s]
    with open("C:\\Users\\shace\\Desktop\\temp\\formulas.norm.txt", "r") as f, open("C:\\Users\\shace\\Documents\\GitHub\\im2latex\\5_dataset_large_val.json", "r+") as f1:
        dat = json.load(f1)
        l = []
        for idx, line in enumerate(f):
            line = line.replace("\n", "")
            tok = list(filter(lambda x:len(x) != 0, line.split(" ")))
            if all(sym in s for sym in tok) and len(tok) < 322:
                l.append({"image_id": str(idx), "caption": line})
        dat["annotations"] = l
        json.dump(dat, f1, indent=4)
    

def load_latex_from_txt(path="C:\\Users\\shace\\Desktop\\temp\\formulas.norm.txt", blacklist_path="C:\\Users\\shace\\Desktop\\temp\\black.txt"):
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
    with open("C:\\Users\\shace\\Desktop\\temp\\math2.txt", "w+") as file:
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

    return res


# get_dataset_tokens()
# load_latex_from_txt()
# update_json()
print(split_data("8/4\\lim_{x\\to\\infty}\\frac{-8\\frac{\\sin{3/x}}{x^{5}}}{-2x^{-3}}"))