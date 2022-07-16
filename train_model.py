import collections
import logging
import os
import shutil
import json
import inspect
import random
import time
import PIL
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from encoder import Encoder
from decoder import Decoder
from keras.optimizers import adam_v2
from keras.losses import SparseCategoricalCrossentropy
from utils.images_preprocessing import make_fix_size

MODEL_PATH = "trained_model\\"
CONFIG_PATH = "train_config.json"
CHECKPOINTS_PATH = "checkpoints\\"
DATASET_PATH = "dataset\\"
IMAGES_PATH = "images\\"
META_PATH = "meta.txt"
TOKENIZER_PATH = "tokenizer.json"
CAPTION_PATH = "dataset.json"

RESIZED_IMG_H = 175
RESIZED_IMG_W = 1500
DATA_SPLIT = 1

def check_path(path):
    """Checks if given str is path

    Args:
           path: a path string
        Returns:
            path or path\\
    """
    return path + "\\" if path[-1] != "\\" else path


def get_config():
    try:
        with open(check_path(os.path.abspath(".")) + CONFIG_PATH, 'r+') as file:
            config = dict(json.load(file)["params"][0])
    except:
        raise IOError("Training config file reading failed. Expected at project root folder")
    return config


def log_init(path, name):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s:    %(message)s")

    file_handler = logging.FileHandler(f"{path}/{name}.txt", mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter) 

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def calc_max_length(tensor):
    return max(len(t) for t in tensor)


def map_func(img_name, cap):
    img = tf.io.read_file(img_name.decode("utf-8"))
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, (int(img.shape[0]*2), int(img.shape[1]*2)), tf.image.ResizeMethod.BILINEAR)
    enhancer = PIL.ImageEnhance.Sharpness(PIL.Image.fromarray(np.uint8(img.numpy())).convert('RGB'))
    img = enhancer.enhance(3)

    img = make_fix_size(img, False)

    img = tf.image.resize(img, (RESIZED_IMG_H, RESIZED_IMG_W))
    return img.numpy(), cap


def get_image_to_caption(annotations_file):
    image_path_to_caption = collections.defaultdict(list)
    for val in annotations_file['annotations']:
        caption = f"<start> {val['caption']} <end>"
        image_path = check_path(os.path.abspath(".")) + DATASET_PATH + IMAGES_PATH + val["image_id"] + ".png"
        image_path_to_caption[image_path].append(
                caption)
    return image_path_to_caption


def loss_function(real, pred, loss_object):
    mask = tf.math.logical_not(tf.math.equal(real, 0))

    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)


@tf.function
def train_step(img_tensor, tokenizer, target, decoder, encoder, optimizer, loss_object):
    loss = 0

    hidden = decoder.get_initial_state(batch_size=target.shape[0], dtype="float32")
    state_out = hidden[0]

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)

    with tf.GradientTape() as tape:
        features = encoder(img_tensor)

        for i in range(1, target.shape[1]):
            predictions, state_out, hidden, _ = decoder(dec_input, features, state_out, hidden)

            loss += loss_function(target[:, i], predictions, loss_object)

            dec_input = tf.expand_dims(target[:, i], 1)

    total_loss = (loss / int(target.shape[1]))

    trainable_variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, trainable_variables)

    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, total_loss


def data_preprocess(config, logger, m_path):
    with open(check_path(os.path.abspath(".")) + DATASET_PATH + CAPTION_PATH, 'r') as f:
        annotations = json.load(f)

    image_path_to_caption = get_image_to_caption(annotations)

    image_paths = list(image_path_to_caption.keys())
    random.shuffle(image_paths)

    train_image_paths = image_paths[:config["image_count"]]
    logger.info(
            f"Annotations and image paths have been successfully extracted from \
            {check_path(os.path.abspath('.')) + DATASET_PATH + CAPTION_PATH}\ntotal images count - {len(train_image_paths)}")

    train_captions = []
    img_name_vector = [] 

    for image_path in train_image_paths:
        caption_list = image_path_to_caption[image_path]
        train_captions.extend(caption_list)
        img_name_vector.extend([image_path] * len(caption_list))

    with open(m_path + TOKENIZER_PATH, 'w+', encoding='utf-8') as f:
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=config["top_words"], split=' ', oov_token="<unk>",
                                                                   lower=False, filters='%')
        tokenizer.fit_on_texts(train_captions)
        tokenizer.word_index['<pad>'] = 0
        tokenizer.index_word[0] = '<pad>'

        tokenizer_json = tokenizer.to_json()
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))

    logger.info("\nTokenizer has been created with params:")
    for k, v in tokenizer.get_config().items():
        logger.info(f"{k} - {v}")

    train_seqs = tokenizer.texts_to_sequences(
            train_captions)

    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')
    max_length = calc_max_length(train_seqs)

    with open(m_path + META_PATH, "a") as meta:
        meta.write(f"{max_length}")

    img_to_cap_vector = collections.defaultdict(list)
    for img, cap in zip(img_name_vector, cap_vector):
        img_to_cap_vector[img].append(cap)

    logger.info(f"Img_to_cap vector is compiled. Max cap length - {max_length}")
    return img_to_cap_vector, tokenizer


def training(img_to_cap_vector, tokenizer,config, logger, m_path):
    img_keys = list(img_to_cap_vector.keys())
    random.shuffle(img_keys)

    slice_index = int(len(img_keys) * DATA_SPLIT)
    img_name_train_keys, img_name_val_keys = img_keys[:slice_index], img_keys[slice_index:]

    img_name_train = []
    cap_train = []
    for imgt in img_name_train_keys:
        capt_len = len(img_to_cap_vector[imgt])
        img_name_train.extend([imgt] * capt_len)
        cap_train.extend(img_to_cap_vector[imgt])

    img_name_val = []
    cap_val = []
    for imgv in img_name_val_keys:
        capv_len = len(img_to_cap_vector[imgv])
        img_name_val.extend([imgv] * capv_len)
        cap_val.extend(img_to_cap_vector[imgv])

    logger.info(
            f"Dataset has been splitted:\nIMG_TRAIN - {len(img_name_train)}\n\
            CAP_TRAIN - {len(cap_train)}\nIMG_VAL - {len(img_name_val)}\nCAP_VAL - {len(cap_val)}")

    num_steps = len(img_name_train) // config["batch_size"]
    dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

    dataset = dataset.map(lambda item1, item2: tf.numpy_function(
            map_func, [item1, item2], [tf.float32, tf.int32]),
                              num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.shuffle(config["buffer_size"]).batch(config["batch_size"])
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    logger.info(f"Dataset is ready")

    encoder = Encoder(config['embedding_dim'])
    decoder = Decoder(config['embedding_dim'], config['units'], config['top_words'] + 1)

    optimizer = adam_v2()
    loss_object = SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')

    ckpt = tf.train.Checkpoint(encoder=encoder,
                                   decoder=decoder,
                                   optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, m_path + CHECKPOINTS_PATH, max_to_keep=10)

    logger.info(f"OPTIMIZER SUMMARY:\n{optimizer.get_config()}")

    loss_plot = []

    for epoch in range(config["epochs"]):
        start = time.time()
        total_loss = 0

        for (batch, (img_tensor, target)) in tqdm(enumerate(dataset)):
            batch_loss, t_loss = train_step(img_tensor, tokenizer, target, decoder, encoder, optimizer, loss_object)
            total_loss += t_loss

            if batch % 100 == 0:
                average_batch_loss = batch_loss.numpy() / int(target.shape[1])
                logger.info(
                        f'Epoch {epoch + 1} Batch {batch} Loss = {batch_loss.numpy()} / {int(target.shape[1])} = {average_batch_loss:.4f}')
            
        loss_plot.append(total_loss / num_steps)

        ckpt_manager.save()
        logger.info(f"Epoch {epoch} checkpoint saved!")

        logger.info(f'Epoch {epoch + 1} Loss {total_loss / num_steps:.6f}')
        logger.info(f'Time taken for 1 epoch {time.time() - start:.2f} sec\n')

    logger.info(f"Training is done!")

    # plt.plot(loss_plot)
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.title('Loss Plot')
    # plt.show()


def train():
    m_path = check_path(os.path.abspath(".")) + MODEL_PATH

    print(f"Trained model files will be created at {m_path}")

    os.makedirs(os.path.dirname(m_path), exist_ok=True)
    if len(os.listdir(m_path)) != 0:
        if input("The destination folder is not empty, all files will be deleted after training starts. Proceed? (y | n)") == "y":
            shutil.rmtree(m_path)
            os.makedirs(os.path.dirname(m_path), exist_ok=True)
        else:
            return

    config = get_config()

    with open(m_path + META_PATH, "w+") as meta:
        meta.write(f"{config['embedding_dim']}\n{config['units']}\n{config['top_words'] + 1}\n")

    logger = log_init(m_path, "model_log") 
    logger.info(f"Model '{m_path}' has been created with these params:")
    for k, v in config.items():
        logger.info(f"{k} - {v}")
    
    logger.info(f"---------------MODEL AND UTILS SUMMARY---------------")
    logger.info(f"ENCODER_INIT:\n{inspect.getsource(Encoder.__init__)}")
    logger.info(f"ENCODER_CALL:\n{inspect.getsource(Encoder.call)}")
    logger.info(f"RESIZE_HEIGHT: {RESIZED_IMG_H}")
    logger.info(f"RESIZE_WIDTH: {RESIZED_IMG_W}")
    logger.info(f"DATA_SPLIT_INDEX: {DATA_SPLIT}")
    logger.info(f"-----------------------------------------------------")

    img_to_cap, tokenizer = data_preprocess(config, logger, m_path)
    training(img_to_cap, tokenizer, config, logger, m_path)

    # tokenizer_path = model_path + "\\tokenizer.json"
    # checkpoint_path = model_path + "\\checkpoints\\"
    # #meta_path = 
