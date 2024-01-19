import os
import argparse
from shutil import copy

from tqdm import tqdm


DLN_CLS_ID = {
    "0": "0",  # Caption combo
    "1": "1",  # Footnote as is
    "2": "2",  # Formula combo
    "3": "3",  # List-item as is
    "4": "4",  # Page-footer as is
    "5": "5",  # Page-header as is
    "6": "6",  # Picture as is
    "8": "7",  # Table combo
    "9": "8",  # Text as is
    "10": "9",  # Title combo
}

SB_CLS_ID = {
    "0": "10",  # abstract
    "2": "0",  # caption
    "3": "2",  # equation
    "5": "6",  # figure
    "8": "11",  # section
    "9": "12",  # subsection
    "10": "7",  # table
    "11": "9",  # title
}


def main(dln_path, sb_path, sdln_path):
    os.makedirs(sdln_path, exist_ok=True)

    for d, _cls_id in [(dln_path, DLN_CLS_ID), (sb_path, SB_CLS_ID)]:
        for s in ["val", "train", "test"]:
            img_dir = os.path.join(d, "images", s)
            lab_dir = os.path.join(d, "labels", s)

            print(f"processing {img_dir}")

            if not os.path.exists(img_dir) or not os.path.exists(lab_dir):
                print(f"No such dir: {img_dir}")
                continue

            imgs = map(
                lambda x: os.path.basename(x).split(".")[0],
                os.listdir(img_dir),
            )
            labs = map(
                lambda x: os.path.basename(x).split(".")[0],
                os.listdir(lab_dir),
            )
            pairs = set(imgs) & set(labs)

            yolo_img_path = os.path.join(sdln_path, "images", s)
            yolo_lab_path = os.path.join(sdln_path, "labels", s)
            os.makedirs(yolo_img_path, exist_ok=True)
            os.makedirs(yolo_lab_path, exist_ok=True)

            for p in tqdm(pairs):
                res = []
                with open(os.path.join(lab_dir, f"{p}.txt")) as s_l:
                    for l in s_l.readlines():
                        ll = l.rstrip("\n").split()
                        _id = _cls_id.get(ll[0])
                        if _id:
                            res.append(" ".join([_id] + ll[1:]) + "\n")

                if res:
                    with open(os.path.join(yolo_lab_path, f"{p}.txt"), "w+") as n_l:
                        [n_l.write(el) for el in res]

                    copy(src=os.path.join(img_dir, f"{p}.png"), dst=yolo_img_path)


# /home/shace_linux/projects/deep_scriptum/im2latex/training_data/DocLayNet_yolo
# /home/shace_linux/projects/deep_scriptum/im2latex/training_data/SciBank_yolo
# /home/shace_linux/projects/deep_scriptum/im2latex/training_data/SciDLN
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dln-path",
        type=str,
        help="Path to DLN-yolo",
        required=True,
    )
    parser.add_argument(
        "--sb-path", 
        type=str, 
        help="Path to SciBank-yolo", 
        required=True
    )
    parser.add_argument(
        "--sdln-path", 
        type=str, 
        help="Path to SciDLN-yolo", 
        required=True
    )

    args = parser.parse_args()

    main(dln_path=args.dln_path, 
         sb_path=args.sb_path, 
         sdln_path=args.sdln_path)
