import json
from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.models import Unigram, WordLevel, WordPiece,
from tokenizers.trainers import UnigramTrainer, WordLevelTrainer, WordPieceTrainer


LABEL_PATH = "C:/users/shace/documents/github/im2latex/dataset_NG_cleaned.json"


def generate_tokenizer(equations, output, vocab_size):
    tokenizer = Tokenizer(WordPiece())
    # tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.pre_tokenizer = pre_tokenizers.CharDelimiterSplit()
    tokenizer.enable_padding(pad_token="<pad>", pad_id=0)
    # trainer = UnigramTrainer(unk_token="<unk>", special_tokens=["<unk>", "<sep>", "<mask>", "<cls>", "<pad>", "<start>", "<end>"], vocab_size=vocab_size, show_progress=True)
    trainer = WordPieceTrainer(vocab_size=500, special_tokens=["<pad>", "<unk>" , "<start>", "<end>"], show_progress=True)
    tokenizer.train_from_iterator(equations, trainer)
    t_ids = [el.ids for el in tokenizer.encode_batch(["<start> \\langle u _ { i } , v _ { i } \\rangle <end>", "<start> \\mathrm { Spec } k <end>", "<start> d N _ { e v t } \\propto d \\sigma <end>",])]
    print(t_ids)
    print([tokenizer.id_to_token(t) for t in t_ids[1]])
    print(tokenizer.decode(t_ids[1]))
    tokenizer.save(path=output, pretty=True)


if __name__ == "__main__":
    with open(LABEL_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)["annotations"]
    
    caps = [el["caption"] for el in dataset]
    generate_tokenizer(caps, "test_tokenizer.json", 500)