import os
import json
import argparse
from shutil import copy
from collections import defaultdict

from tqdm import tqdm


SIZE = 1025


def main(coco_path, yolo_path):
    os.makedirs(yolo_path)
    for s in ["test", "train", "val"]:
        with open(os.path.join(coco_path, "COCO", f"{s}.json")) as stream:
            l = json.load(stream)

        yolo_img_path = os.path.join(yolo_path, "images", s)
        yolo_lab_path = os.path.join(yolo_path, "labels", s)
        os.makedirs(yolo_img_path, exist_ok=True)
        os.makedirs(yolo_lab_path, exist_ok=True)

        annotations = defaultdict(list)
        for ann in l["annotations"]:
            annotations[ann["image_id"]].append(
                {"category_id": ann["category_id"] - 1, "bbox": ann["bbox"]}
            )
        print("dict completed")

        for img in tqdm(l["images"]):
            _id, filename = img["id"], img["file_name"].split(".")[0]

            for ann in annotations[_id]:
                _cls = ann["category_id"]
                left, top, w, h = ann["bbox"]

                xywh = [(left + w / 2) / SIZE, (top + h / 2) / SIZE, w / SIZE, h / SIZE]

                with open(os.path.join(yolo_lab_path, f"{filename}.txt"), "a+") as f:
                    f.write(" ".join(map(str, [_cls] + xywh)) + "\n")

            copy(
                src=os.path.join(coco_path, "PNG", f"{filename}.png"), dst=yolo_img_path
            )


# /home/shace_linux/projects/deep_scriptum/im2latex/training_data/DocLayNet_core
# /home/shace_linux/projects/deep_scriptum/im2latex/training_data/DocLayNet_yolo
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
