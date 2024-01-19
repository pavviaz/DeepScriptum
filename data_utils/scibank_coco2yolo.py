import os
import csv
import argparse
from collections import defaultdict

from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split


SIZE = 1025

_CLS_ID = {
    "abstract": 0,
    "block_text": 1,
    "caption": 2,
    "equation": 3,
    "equation_inline": 4,
    "figure": 5,
    "IEEEKeywords": 6,
    "reference": 7,
    "section": 8,
    "subsection": 9,
    "table": 10,
    "title": 11,
}


def main(coco_path, yolo_path):
    os.makedirs(yolo_path, exist_ok=True)

    annotations = defaultdict(list)
    with open(os.path.join(coco_path, "METADATA_FINAL.csv")) as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            dx, dy = SIZE / int(row["Width_Page"]), SIZE / int(row["Height_Page"])
            annotations[
                os.path.join(
                    coco_path, "PAPER_TAR", row["Folder"], "ImagePage", row["Page"]
                )
            ].append(
                {
                    "category_id": _CLS_ID[row["Class"]],
                    "bbox": [
                        int(row["CoodX"]) * dx,
                        int(row["CoodY"]) * dy,
                        int(row["Width"]) * dx,
                        int(row["Height"]) * dy,
                    ],
                }
            )

    print("dict completed")

    X_train, X_test, y_train, y_test = train_test_split(
        list(annotations.keys()), list(annotations.values()), test_size=0.10
    )

    for t, kv in [
        ("train", zip(X_train, y_train)),
        ("val", zip(X_test, y_test)),
    ]:
        yolo_img_path = os.path.join(yolo_path, "images", t)
        yolo_lab_path = os.path.join(yolo_path, "labels", t)
        os.makedirs(yolo_img_path, exist_ok=True)
        os.makedirs(yolo_lab_path, exist_ok=True)

        for k, v in tqdm(kv):
            f = k.split(os.path.sep)
            filename = f"{f[-3]}_{f[-1].split('.')[0]}"

            for ann in v:
                _cls = ann["category_id"]
                left, top, w, h = ann["bbox"]
                xywh = [(left + w / 2) / SIZE, (top + h / 2) / SIZE, w / SIZE, h / SIZE]
                with open(os.path.join(yolo_lab_path, f"{filename}.txt"), "a+") as f:
                    f.write(" ".join(map(str, [_cls] + xywh)) + "\n")

            img = Image.open(k)
            img = img.resize((SIZE, SIZE))
            img.save(os.path.join(yolo_img_path, f"{filename}.png"))


# /home/shace_linux/projects/deep_scriptum/im2latex/training_data/SciBank
# /home/shace_linux/projects/deep_scriptum/im2latex/training_data/SciBank_yolo
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--coco-path",
        type=str,
        help="Path to source coco-format dataset",
        required=True,
    )
    parser.add_argument(
        "--yolo-path", 
        type=str, 
        help="Path to yolo-reformatted dataset", 
        required=True
    )

    args = parser.parse_args()

    main(coco_path=args.coco_path, yolo_path=args.yolo_path)
