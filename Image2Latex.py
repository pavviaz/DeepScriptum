import argparse
import os
import shutil
import tensorflow as tf
from PIL import ImageGrab
from call_model import predict
from train_model import train

MODEL_PATH = "model_in_usage\\"
CHECKPOINTS_PATH = "checkpoints\\"
DATASET_PATH = "dataset\\"
IMAGES_PATH = "images\\"
META_PATH = "meta.txt"
TOKENIZER_PATH = "tokenizer.json"
TMP_IMG_PATH = "tmp.png"
CAPTION_PATH = "dataset.json"
DATASET_AMOUNT = 100011
DATASET_URL = "https://drive.google.com/u/0/uc?export=download&confirm=64GF&id=1dBiNSowXV2MQfpOGFk8jU6m-ll-4Lz9n"


def use_gpu(turn, gb: int = 4):  # enable or disable GPU backend
    if turn:
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.compat.v1.Session(config=config)

        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            # Restrict TensorFlow to only allocate 1*X GB of memory on the first GPU
            try:
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=(1024 * gb))])
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Virtual devices must be set before GPUs have been initialized
                print(e)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        try:
            # Disable all GPUS
            tf.config.set_visible_devices([], 'GPU')
            visible_devices = tf.config.get_visible_devices()
            for device in visible_devices:
                assert device.device_type != 'GPU'
        except:
            # Invalid device or cannot modify virtual devices once initialized.
            pass


def input_check(inp, from_to):
    if inp.isnumeric() and from_to[0] <= int(inp) <= from_to[1]:
        print("Something wrong with input")
    return int(inp)


def check_path(path):
    """Checks if given str is path

    Args:
           path: a path string
        Returns:
            path or path\\
    """
    return path + "\\" if path[-1] != "\\" else path


def input_default(inp, val):
        return val if inp == "" else inp


def main():
    argparser = argparse.ArgumentParser(description='The script for using and training model + downloading dataset', add_help=True)
    argparser.add_argument('--train', action='store_true', help='Call interactable menu')
    argparser.add_argument('-i', '--image', type=str, default=None, help='Path to LaTeX image')
    argparser.add_argument('-b', '--bwidth', type=int, default=25, help='Beam width value')
    argparser.add_argument('-t', '--temperature', type=float, default=0.1, help='TODO')
    argparser.add_argument('-c', '--clipboard', action='store_true', help='Copy image from clipboard')
    argparser.add_argument('--compute-threads', type=int, default=6, help='Number of dataset downloading and preprocessing threads')
    argparser.add_argument('--use-gpu', action='store_true', help='Enable GPU for training (CUDNN installation required)')
    args = argparser.parse_args()

    # print(args)
    if args.use_gpu:
        use_gpu(True, 9)
    else:
        use_gpu(False)


    if args.train:
        print("Training mode enabled")

        if args.use_gpu:
            use_gpu(True, 9)
        else:
            use_gpu(False)

        path = check_path(os.path.abspath('.')) + DATASET_PATH
        print(f"Trying to catch dataset files at {path}...")
        try:
            if os.path.exists(path + CAPTION_PATH):
                print(f"--Caption file has been detected")
        except:
            raise IOError("Dataset file is missing! Expected at '{DATASET_PATH}' folder")
        
        if os.path.exists(path + IMAGES_PATH) and len(os.listdir(path + IMAGES_PATH)) == DATASET_AMOUNT:
            print("--Dataset images have been detected")
        else:
            print(f"--Dataset images are missing or damaged. Please download dataset ({DATASET_URL}) and place unzipped images to {path + IMAGES_PATH}")
            return

        train()
    else:
        print("Prediction mode enabled")
        path = check_path(os.path.abspath('.')) + MODEL_PATH
        print(f"Trying to catch model in {path}...")
        try:
            if len(os.listdir(path + CHECKPOINTS_PATH)) > 1:
                print(f"--Model checkpoints have been detected")
            if all(os.path.exists(path + p) for p in [META_PATH, TOKENIZER_PATH]):
                print(f"--Meta and Tokenizer files have been detected")
        except:
            raise IOError("Model files are missing!\nExpected in '{MODEL_PATH}' folder")

        if args.clipboard:
            try:
                ImageGrab.grabclipboard().save(TMP_IMG_PATH)
                im_path = TMP_IMG_PATH
            except:
                raise IOError("No image in clipboard!")
        else:
            if args.image is not None and os.path.exists(args.image):
                im_path = args.image
            else:
                raise IOError("No input image!")

        predict(im_path, path + TOKENIZER_PATH, path + META_PATH, path + CHECKPOINTS_PATH, beam_width=int(args.bwidth) if int(args.bwidth) > 0 else 25, temperature=args.temperature if 0 < args.temperature <= 1 else 0.1)


if __name__ == "__main__":
    main()