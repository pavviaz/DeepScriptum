import json
import multiprocessing
import os
import subprocess
from multiprocess_funcs import compute_threads_work
from tqdm import tqdm
from datetime import datetime
import re
from convert_rgba_to_rgb import generate


DOWNLOAD_THREADS = 20
DATASET_DIR = "images_150_regenerated"
RGB_DATASET_DIR = "images_150_regenerated_rgb"
LABEL_PATH = "5_dataset_large.json"

TEX_F = "\\documentclass[\n  9pt,\n  convert={\n    density=175,\n    outext=.png\n  },\n]{standalone}\n\\usepackage{amsfonts,amsmath,amssymb,mathrsfs,braket,mathtools}\n\\begin{document}\n"
TEX_S = "\\end{document}\n"


def save_image(id, caption):
    with open(os.path.join(DATASET_DIR, f"{id}.tex"), "w+") as tex:
        tex.write(TEX_F + f"  $ \\displaystyle {caption} $\n" + TEX_S)
    try:
        subprocess.run(
            ["pdflatex", "--shell-escape", "--enable-write18", f"{id}.tex"], timeout=5
        )
    except Exception as e:
        with open("errors.txt", "a") as err:
            err.write(
                f"ERROR: {e} with cap {caption} and id {id} at {datetime.now()}\n"
            )


def preprocess_and_save(dct, start, end):
    dict_temp = dct[start:end]
    for cap in tqdm(dict_temp):
        save_image(cap["image_id"], cap["caption"])


def main(dataset):
    threads = [
        multiprocessing.Process(
            target=preprocess_and_save, args=(dataset, arg[0], arg[1])
        )
        for arg in compute_threads_work(len(dataset), DOWNLOAD_THREADS)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def remove_bad_captions():
    with open(LABEL_PATH, "r+") as file:
        dataset = json.load(file)

        with open(os.path.join(DATASET_DIR, "errors.txt")) as err:
            errors = [
                (t := re.findall(r"\d+.tex", line.rstrip("\n"))[0])[: t.find(".tex")]
                for line in filter(lambda x: len(x) > 0, [e for e in err])
            ]
        l = []
        for el in tqdm(dataset["annotations"]):
            if not el["image_id"] in errors:
                l.append(el)
        dataset["annotations"] = l
        json.dump(dataset, file, indent=4)


def rgba2rgb():
    for el in tqdm(filter(lambda x: ".png" in x, os.listdir(DATASET_DIR))):
        generate(os.path.join(DATASET_DIR, el), os.path.join(RGB_DATASET_DIR, el))


if __name__ == "__main__":
    with open(LABEL_PATH, "r") as file:
        dataset = json.load(file)["annotations"]

    os.chdir(DATASET_DIR)

    main(dataset)
    remove_bad_captions()
    rgba2rgb()
