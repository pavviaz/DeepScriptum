import random
import re
import json
import string

from tqdm import tqdm


def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str


def get_dataset_tokens(d_path="C:\\Users\\shace\\Documents\\GitHub\\im2latex\\5_dataset_large.json"):
    with open(d_path, "r") as file:
        s = set()
        dataset = json.load(file)["annotations"]
        for cap in dataset:
            [s.add(token) for token in list(filter(lambda x: len(x) != 0, cap["caption"].split(" ")))]
        with open("C:\\Users\\shace\\Desktop\\temp\\black.txt", "w+") as f:
            [f.write(sym + "\n") for sym in s]
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

def mbox_fix(d_path="C:\\Users\\shace\\Documents\\GitHub\\im2latex\\5_dataset_large_mbox_train.json"):
    str_to_find = "\\mbox {"
    l = []
    with open(d_path, "r+") as file:
        dataset = json.load(file)
        for cap in tqdm(dataset["annotations"]):
            capt = cap["caption"]
            tmp_capt = capt
            find_idx = 0
            # tmp_capt = "J _ { \\mu \\nu } = 2 i \\mbox { } \\big \\{ e _ { [ \\mu } { } ^ { \\alpha \\dot \\alpha } e _ { \\nu ] } { } ^ \\beta { } _ { \\dot \\alpha } \\widehat A ^ { ( 1 ) } _ \\alpha \\star \\widehat A ^ { ( 1 ) } _ \\beta + \\mbox { r q s w w $ wwdwd $ r q s w w $ \\frac { 1 } { 2 } $ $ wwdwd $ $ wwdwd $ $ wwdwd $ f w f f w w e w e $ wwdwd $ } \\big \\} _ { z = 0 } + L _ { \\mu \\nu } + \\mbox { $ W $ - t e r m s } ,"
            tmp_capt = "x ^ k w ( x ) \\mbox { i s b o u n d e d f o r e a c h } k = 0 , 1 , \\cdots "
            while tmp_capt.find(str_to_find, find_idx) != -1:
                mbox_inner = ""
                brace_count = 1
                start_idx = tmp_capt.find(str_to_find, find_idx) 
                tmp_idx = start_idx + len(str_to_find)
                
                while brace_count != 0:
                    mbox_inner += tmp_capt[tmp_idx]
                    tmp_idx += 1
                    if tmp_capt[tmp_idx] == "{":
                        brace_count += 1
                    elif tmp_capt[tmp_idx] == "}":
                        brace_count -= 1
                        
                if len(mbox_inner.replace(" ", "")) == 0:
                    tmp_capt = tmp_capt[:start_idx] + tmp_capt[tmp_idx + 1:]
                    continue
                
                r = re.findall(r"\$ [^\$]+ \$", mbox_inner)
                tmp_change_dct = { get_random_string(5): v  for v in r}
                
                for k, v in tmp_change_dct.items():
                    v_idx = mbox_inner.find(v)
                    mbox_inner = mbox_inner[:v_idx] + k + mbox_inner[v_idx + len(v):]
                
                # ' r q s w w arekr r q s w w gljts qqyza kpjwq owyfr f w f f w w e w e fnvjk '
                
                word_str = ""
                ts = ""
                for idx in range(len(mbox_inner)):
                    ts += mbox_inner[idx]
                    aa = ts.replace(" ", "")
                    if aa in eng_dict and len(aa) > 1:
                        word_str += aa + " "
                        ts = ""
                    
                
                result_str_CAP = " ".join([ch + "_" if len(ch) == 1 else " " + ch for ch in mbox_inner.split()])
                result_str_DOWNLOAD = "".join([ch if len(ch) == 1 else " " + ch + " " for ch in mbox_inner.split()])
                for k, v in tmp_change_dct.items():
                    result_str_CAP = result_str_CAP.replace(k, v)
                    result_str_DOWNLOAD = result_str_DOWNLOAD.replace(k, v)
                tmp_capt = tmp_capt[:start_idx + len(str_to_find) + 1] + result_str_CAP + tmp_capt[tmp_idx - 1:]
                find_idx = start_idx + len(result_str_CAP)
                
            l.append({"image_id": cap["image_id"], "caption": tmp_capt})
                
            # print()
        dataset["annotations"] = l
        json.dump(dataset, file, indent=4)
                # dat["annotations"] = l
                # json.dump(dat, f1, indent=4)
           
                
# eng_dict = set()
# with open("C:/users/shace/desktop/eng_dict.txt", "r") as f:
#     for line in f:
#         eng_dict.add(line.rstrip("\n"))

# mbox_fix()
# get_dataset_tokens()
# load_latex_from_txt()
# update_json()
# print(split_data("8/4\\lim_{x\\to\\infty}\\frac{-8\\frac{\\sin{3/x}}{x^{5}}}{-2x^{-3}}"))


# import re

# fef = ["\operatorname { l o g } M _ { \mathrm { 2 0 0 b } } - \operatorname { l o g } M _ { \mathrm { v i r } } = 0 . 0 0 5 2 ( \operatorname { l o g } M _ { \mathrm { v i r } } - 1 3 . 5 ) + 0 . 0 4 9 8",
#        "\operatorname* { l i m } _ { t \rightarrow + \infty } \widetilde { V } _ { N _ { R } - n + 1 } ^ { t } ( \widetilde { S } _ { f } ^ { i , s } ) = \widetilde { V } _ { N _ { R } - n + 1 } ( \widetilde { S } _ { f } ^ { i , s } ) \nonumber ."]

# for line in fef:
#     line = line.replace("\n", "")
    
#     if not "operatorname" in line:
#         continue
    
#     r = re.findall(r"\\operatorname {[^\}]+}", line) + re.findall(r"\\operatorname\* {[^\}]+}", line)
#     for sub in r:
#         rr = re.findall(r"{[^\}]+}", sub)
#         res = "\\mathrm { " + rr[0].strip("{}").replace(" ", "") + " }"
#         line = line.replace(sub, res, 1)