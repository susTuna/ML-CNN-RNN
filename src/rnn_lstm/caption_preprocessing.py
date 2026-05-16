import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

PAD_TOKEN = "<pad>"
START_TOKEN = "<start>"
END_TOKEN = "<end>"
UNK_TOKEN = "<unk>"

SPECIAL_TOKENS = (PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN)
PAD_ID = 0
START_ID = 1
END_ID = 2
UNK_ID = 3


def clean_caption(text):
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def wrap_with_special_tokens(text):
    return START_TOKEN + " " + text + " " + END_TOKEN


def preprocess_caption(text):
    return wrap_with_special_tokens(clean_caption(text))


def preprocess_captions(captions_source):
    if isinstance(captions_source, (str, Path)):
        captions_by_image = load_flickr8k_captions(captions_source)
        if len(captions_by_image) > 0:
            return preprocess_captions(captions_by_image)

        with open(captions_source, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        return preprocess_captions(lines)

    if isinstance(captions_source, dict):
        out = {}
        for image_id, captions in captions_source.items():
            out[image_id] = [preprocess_caption(c) for c in captions]
        return out

    return [preprocess_caption(c) for c in captions_source]


def load_flickr8k_captions(captions_path):
    captions_path = Path(captions_path)
    captions = {}

    with open(captions_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
        if first_line == "image,caption":
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",", 1)
                if len(parts) != 2:
                    continue
                image_filename, caption = parts
                if image_filename not in captions:
                    captions[image_filename] = []
                captions[image_filename].append(caption)
        else:
            f.seek(0)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if "\t" not in line:
                    continue
                image_part, caption = line.split("\t", 1)
                image_filename = image_part.split("#", 1)[0]
                if image_filename not in captions:
                    captions[image_filename] = []
                captions[image_filename].append(caption)

    return captions


def build_vocabulary(captions, min_freq=2):
    counter = Counter()
    for caption in captions:
        for token in caption.split():
            if token in SPECIAL_TOKENS:
                continue
            counter[token] += 1

    word2idx = {}
    for i, tok in enumerate(SPECIAL_TOKENS):
        word2idx[tok] = i

    next_idx = len(SPECIAL_TOKENS)
    sorted_words = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    for word, count in sorted_words:
        if count < min_freq:
            continue
        word2idx[word] = next_idx
        next_idx += 1

    return word2idx


def build_index_to_word(word2idx):
    return {idx: word for word, idx in word2idx.items()}


def encode_caption(caption, word2idx):
    ids = []
    for tok in caption.split():
        if tok in word2idx:
            ids.append(word2idx[tok])
        else:
            ids.append(UNK_ID)
    return ids


def pad_sequence(seq, max_len, pad_id=PAD_ID):
    arr = np.full(max_len, pad_id, dtype=np.int32)
    n = min(len(seq), max_len)
    arr[:n] = seq[:n]
    return arr


def tokenize_and_pad(captions, word2idx, max_len=35):
    n = len(captions)
    out = np.full((n, max_len), PAD_ID, dtype=np.int32)
    for i, caption in enumerate(captions):
        ids = encode_caption(caption, word2idx)
        k = min(len(ids), max_len)
        out[i, :k] = ids[:k]
    return out


def decode_sequence(ids, idx2word, strip_special=True):
    words = []
    for idx in ids:
        idx = int(idx)
        word = idx2word.get(idx, UNK_TOKEN)
        if strip_special:
            if word == END_TOKEN:
                break
            if word == PAD_TOKEN or word == START_TOKEN:
                continue
        words.append(word)
    return " ".join(words)


def build_training_pairs(sequences):
    if sequences.ndim != 2 or sequences.shape[1] < 2:
        raise ValueError(
            "sequences must be 2-D with at least 2 columns, got "
            + str(sequences.shape)
        )
    decoder_input = sequences[:, :-1].astype(np.int32)
    decoder_target = sequences[:, 1:].astype(np.int32)
    return decoder_input, decoder_target


def save_vocabulary(word2idx, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(word2idx, f, indent=2)


def load_vocabulary(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_flickr8k_split(split_file):
    with open(split_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def prepare_dataset(
    captions_by_image,
    image_ids,
    features,
    index_map,
    word2idx,
    max_len,
):
    feats = []
    caption_list = []

    for image_id in image_ids:
        if image_id not in captions_by_image:
            continue
        if image_id not in index_map:
            continue
        feat_row = features[index_map[image_id]]
        for caption in captions_by_image[image_id]:
            feats.append(feat_row)
            caption_list.append(caption)

    if len(feats) == 0:
        raise ValueError(
            "No (image, caption) pairs found — check that image_ids overlap "
            "with both captions_by_image and index_map."
        )

    feat_array = np.stack(feats, axis=0).astype(np.float32)
    sequences = tokenize_and_pad(caption_list, word2idx, max_len)
    decoder_input, decoder_target = build_training_pairs(sequences)

    return feat_array, decoder_input, decoder_target


def generate_splits(image_ids, train_ratio=0.8, val_ratio=0.1, seed=42):
    np.random.seed(seed)
    ids = list(image_ids)
    np.random.shuffle(ids)
    
    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train+n_val]
    test_ids = ids[n_train+n_val:]
    
    return train_ids, val_ids, test_ids
