import json
import os
from tokenizers import Tokenizer
import torch
import torchvision
import numpy as np
from os.path import join
from model.seq2seqCnnTransformer import Seq2SeqTransformer


WEIGHTS_PATH = "model/weights/weights.tar"
TOKENIZER_PATH = "model/weights/tokenizer.json"
META_PATH = "model/weights/meta.json"
DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def load_image(image, transform):
    # raise NotImplementedError
    if transform:
        image = transform(image)
    return image


def greedy_decoder(image, model, tokenizer, max_len, temp, device):
    model.eval()
    with torch.no_grad():
        img = load_image(image, torchvision.transforms.ToTensor()).unsqueeze(0).to(device)
        
        img_features = model.cnn_encoder(img)

        decoded_caption = "<start> "
        for i in range(max_len):
            tokenized_caption = np.expand_dims(np.array(tokenizer.encode(decoded_caption).ids), 0)[:, :-1]
            _, tgt_mask, _, tgt_padding_mask = model.create_mask(img_features, tokenized_caption, 0)
            
            tgt_emb = model.positional_encoding(model.tgt_tok_emb(torch.tensor(tokenized_caption).to(device)))
            src_emb = model.positional_encoding(img_features)
            transformers_out = model.transformer(src_emb, tgt_emb, None, tgt_mask, None, None, torch.tensor(tgt_padding_mask).to(device), None)
            logits = model.generator(transformers_out)
            
            sampled_token_index = np.argmax(torch.softmax(logits.cpu()[0, i, :], -1) / temp)
            sampled_token = tokenizer.id_to_token(sampled_token_index.item())
            decoded_caption += " " + sampled_token
            if sampled_token == "<end>":
                break
            
            yield decoded_caption.replace(" ##", "")
            

def predict(image, temp, decoder_type=None):
    with open(join(DIR_PATH, META_PATH)) as m:
        params = json.load(m)["params"][0]
        
    tokenizer = Tokenizer.from_file(join(DIR_PATH, TOKENIZER_PATH))
    tokenizer.enable_padding(pad_token="<pad>", pad_id=0, length=params["max_len"])
    
    model = Seq2SeqTransformer(num_encoder_layers=params["num_encoder_layers"], 
                               num_decoder_layers=params["num_decoder_layers"], 
                               emb_size=params["embedding_dim"],
                               nhead=params["nhead"],
                               tgt_vocab_size=params["vocab_size"],
                               dim_feedforward=params["ff_dim"],
                               device="cpu")
    
    checkpoint = torch.load(join(DIR_PATH, WEIGHTS_PATH))
    model.load_state_dict(checkpoint["model"], strict=False)
    
    return greedy_decoder(image, model, tokenizer, params["max_len"], temp=temp, device="cpu")


if __name__ == "__main__":
    pass