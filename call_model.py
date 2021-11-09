import PIL
import tensorflow as tf
import numpy as np
import json
import os
from keras_preprocessing.text import tokenizer_from_json
from encoder import Encoder
from decoder import Decoder
from utils.images_preprocessing import crop_pad_image, make_fix_size

RESIZED_IMG_H = 175
RESIZED_IMG_W = 1500

def index(array, item):
        for idx, val in np.ndenumerate(array):
            if val == item:
                return idx[1]


def find_n_best(array, n):
        probs = [np.partition(array[0], i)[i] for i in range(-1, -n - 1, -1)]
        ids = [index(array, p) for p in probs]
        return [[prob, id] for prob, id in zip(probs, ids)]


def load_image(image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, (int(img.shape[0]*2), int(img.shape[1]*2)), tf.image.ResizeMethod.BILINEAR)
        
        enhancer = PIL.ImageEnhance.Sharpness(PIL.Image.fromarray(np.uint8(img.numpy())).convert('RGB'))
        img = enhancer.enhance(3)

        img = make_fix_size(img, False)

        img = tf.image.resize(img, (RESIZED_IMG_H, RESIZED_IMG_W))
        os.remove(image_path)
        return img, image_path


def beam_evaluate(image, decoder, encoder, max_length, tokenizer, beam_width=10, temperature=0.3):
        # attention_plots = [np.zeros((max_length, 148)) for _ in range(beam_width)]

        hidden = decoder.get_initial_state(batch_size=1, dtype="float32")
        state_out = hidden[0]

        temp_input = tf.expand_dims(load_image(image)[0], 0)

        features = encoder(temp_input)

        dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)

        predictions, state_out, hidden, _ = decoder(dec_input, features, state_out, hidden)
        predictions = tf.nn.softmax(predictions / temperature).numpy()

        init = find_n_best(predictions, beam_width)
        results = [[obj[0], obj[1], hidden, tokenizer.index_word[int(obj[1])] + " ", state_out] for obj in
                   init]  # 0 - prob ; 1 - id ; 2 - hidden

        for i in range(max_length):
            tmp_res = []

            for r in results:
                tmp_preds, tmp_state_out, tmp_hidden, attention_plot = decoder(tf.expand_dims([r[1]], 0), features,
                                                                               r[4], r[2])

                for obj in find_n_best(tf.nn.softmax(tmp_preds / temperature).numpy(), beam_width):
                    tmp_res.append(
                        [obj[0] + r[0], obj[1], tmp_hidden, r[3] + tokenizer.index_word[int(obj[1])] + " ",
                         tmp_state_out, attention_plot])  # multiplied scores, curr id, hidden, prev id

            results.clear()
            tmp_res.sort(reverse=True, key=lambda x: x[0])
            # attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()
            for el in range(beam_width):
                results.append(tmp_res[el])

            # for res, att in zip(results, attention_plots):
            #     att[i] = tf.reshape(res[5], (-1,)).numpy()

            # if any(tokenizer.index_word[int(results[i][1])] == '<end>' for i in range(len(results))):
            #     break

            if all(['<end>' in r[3] for r in results]):
                break

        return results[0][3], None


def predict(image_path: str, tokenizer_path: str, meta_path: str, checkpoint_path: str, beam_width = 2, temperature = 0.3):
        """
        TODO docstring
        """
        try:
            with open(tokenizer_path, "r") as f:
                data = json.load(f)
                tokenizer = tokenizer_from_json(data)
        except:
            raise IOError("Something went wrong with initializing tokenizer")

        try:
            with open(meta_path, "r") as meta:
                for index, line in enumerate(meta):
                    if index == 0:
                        embedding_dim = int(line)
                    elif index == 1:
                        units = int(line)
                    elif index == 2:
                        vocab_size = int(line)
                    elif index == 3:
                        max_length = int(line)
        except:
            raise IOError("Meta file reading failed")

        encoder = Encoder(embedding_dim)
        decoder = Decoder(embedding_dim, units, vocab_size)

        ckpt = tf.train.Checkpoint(encoder=encoder,
                                   decoder=decoder)
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        else:
            print("Restored from scratch")
    
        crop_pad_image(image_path, image_path)

        result, attention_plot = beam_evaluate(image_path, decoder, encoder, max_length, tokenizer, beam_width=beam_width, temperature=temperature)

        try:
            result = result[:result.index("<end>")]
        except:
            print("Something wrong with decoded sequence")
        else:
            print(result)