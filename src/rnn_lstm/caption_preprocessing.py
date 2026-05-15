"""Caption preprocessing for Flickr8k image captioning.

Reads the Flickr8k caption file, cleans the text, builds a vocabulary,
and produces padded integer sequences for teacher-forced decoder training.
"""

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np


# special tokens — fixed positions so the model output indices are stable
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
    """Preprocess captions from a Flickr8k file, a dict, or a list of strings.

    - If given a file path, returns {image_filename: [processed_caption, ...]}.
    - If given a dict {image: [raw_caption, ...]}, returns the same shape but
      with each caption cleaned and wrapped.
    - If given a list of strings, returns a list of processed strings.
    """
    # file path case
    if isinstance(captions_source, (str, Path)):
        captions_by_image = load_flickr8k_captions(captions_source)
        if len(captions_by_image) > 0:
            # got Flickr8k style file
            return preprocess_captions(captions_by_image)

        # not Flickr8k format treat each line as one caption
        with open(captions_source, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        return preprocess_captions(lines)

    # dict case
    if isinstance(captions_source, dict):
        out = {}
        for image_id, captions in captions_source.items():
            out[image_id] = [preprocess_caption(c) for c in captions]
        return out

    # sequence-of-strings case
    return [preprocess_caption(c) for c in captions_source]


def load_flickr8k_captions(captions_path):
    """Read Flickr8k.token.txt — each line is `image#idx\\tcaption`."""
    captions_path = Path(captions_path)
    captions = {}

    with open(captions_path, "r", encoding="utf-8") as f:
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
    """Build a word->id map from the wrapped training captions.

    Words appearing fewer than ``min_freq`` times are dropped (they map to
    <unk> at encoding time). Special tokens occupy indices 0..3.
    """
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
    # sort by count desc, then alphabetically so the vocab is reproducible
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
    """Encode and pad a list of captions to shape (N, max_len)."""
    n = len(captions)
    out = np.full((n, max_len), PAD_ID, dtype=np.int32)
    for i, caption in enumerate(captions):
        ids = encode_caption(caption, word2idx)
        k = min(len(ids), max_len)
        out[i, :k] = ids[:k]
    return out


def decode_sequence(ids, idx2word, strip_special=True):
    """Turn an integer sequence back into a whitespace-joined string.

    Stops at the first <end> token when ``strip_special`` is True.
    """
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
    """Split a padded (N, L) batch into decoder input / target arrays.

    decoder_input  = sequences[:, :-1]  (<start> + body)
    decoder_target = sequences[:,  1:]  (body + <end>)
    """
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
    """Read a Flickr8k split file (one image filename per line)."""
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
    """Build (feature, decoder_input, decoder_target) arrays for one split.

    For every (image, caption) pair, the image feature row is repeated once
    so that all three arrays share the same first dimension.
    """
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
