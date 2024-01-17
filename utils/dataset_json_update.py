import re


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
