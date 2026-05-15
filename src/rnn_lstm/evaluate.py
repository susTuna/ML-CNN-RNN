"""Evaluation pipeline for image caption decoders.

Supports three flavours of generation:
- Keras greedy: runs the trained Keras model autoregressively on a
  pre-extracted CNN feature.
- Scratch greedy: uses ImageCaptionerScratch with a pre-extracted feature.
- Beam search: same as scratch but with beam search (bonus).

All entry points return BLEU-4, METEOR, captions, and wall-clock time so
the notebook can build comparison tables.
"""

import time

import numpy as np

from .caption_preprocessing import (
    END_ID,
    END_TOKEN,
    PAD_ID,
    PAD_TOKEN,
    START_ID,
    START_TOKEN,
    decode_sequence,
)
from .metrics import compute_bleu4, compute_meteor, per_sample_bleu4


class EvaluationResult:
    """Container for one evaluation pass over a split."""

    def __init__(
        self,
        variant,
        decoder_kind,
        bleu4,
        meteor,
        elapsed_seconds,
        captions=None,
        references=None,
        image_ids=None,
    ):
        self.variant = variant
        self.decoder_kind = decoder_kind 
        self.bleu4 = bleu4
        self.meteor = meteor
        self.elapsed_seconds = elapsed_seconds
        self.captions = captions if captions is not None else []
        self.references = references if references is not None else []
        self.image_ids = image_ids if image_ids is not None else []

    def to_dict(self):
        return {
            "variant": self.variant,
            "decoder_kind": self.decoder_kind,
            "bleu4": self.bleu4,
            "meteor": self.meteor,
            "elapsed_seconds": self.elapsed_seconds,
            "num_samples": len(self.captions),
        }


def _generate_caption_keras(model, image_feature, idx2word, seq_len, max_len):
    """Greedy decode for a Keras pre-inject decoder.

    Feed the image feature plus an evolving caption prefix (starts with
    <start>, the rest is <pad>) and pick argmax of the position matching the
    most recent real token.
    """
    feat = image_feature.astype(np.float32)[None, :]  

    decoder_input = np.full((1, seq_len), PAD_ID, dtype=np.int32)
    decoder_input[0, 0] = START_ID

    generated = []
    limit = min(max_len, seq_len)
    for t in range(limit):
        preds = model.predict([feat, decoder_input], verbose=0)
        logits = preds[0, t]
        token = int(np.argmax(logits))
        if token == END_ID:
            break
        if token != PAD_ID:
            generated.append(token)
        if t + 1 < seq_len:
            decoder_input[0, t + 1] = token
        else:
            break

    return decode_sequence(generated, idx2word, strip_special=True)


def evaluate_keras_decoder(
    model,
    features,
    image_ids,
    references_by_image,
    idx2word,
    seq_len,
    max_len=20,
    variant="keras",
    progress_cb=None,
):
    """Evaluate a Keras decoder on a (features, image_ids) split.

    features[i] must align with image_ids[i].
    """
    captions = []
    refs = []

    start = time.time()
    for i, image_id in enumerate(image_ids):
        caption = _generate_caption_keras(
            model,
            image_feature=features[i],
            idx2word=idx2word,
            seq_len=seq_len,
            max_len=max_len,
        )
        captions.append(caption)
        refs.append(list(references_by_image.get(image_id, [])))
        if progress_cb is not None:
            progress_cb(i + 1, len(image_ids))
    elapsed = time.time() - start

    return EvaluationResult(
        variant=variant,
        decoder_kind="keras",
        bleu4=compute_bleu4(refs, captions),
        meteor=compute_meteor(refs, captions),
        elapsed_seconds=elapsed,
        captions=captions,
        references=refs,
        image_ids=list(image_ids),
    )


def _scratch_generate_from_feature(captioner, feature, max_len):
    """Run ImageCaptionerScratch greedy from a pre-extracted feature vector.

    Mirrors the captioner's own ``generate_caption_greedy`` but skips the CNN
    encoder step — used when features are cached on disk.
    """
    cnn_feat = feature.astype(np.float32)

    # project the CNN feature to embed_dim (if the captioner has a projector)
    if captioner.dense_proj is not None:
        x_proj = captioner.dense_proj.forward(cnn_feat).ravel()
    else:
        x_proj = cnn_feat

    states = captioner._init_hidden()
    _, states = captioner._step(x_proj, states)

    token = captioner.start_id
    generated = []

    for _ in range(max_len):
        x_t = captioner.embedding.forward(np.array(token, dtype=np.int32))
        h_out, states = captioner._step(x_t, states)
        logits = captioner.dense_out.forward(h_out).ravel()
        token = int(np.argmax(logits))
        if token == captioner.end_id:
            break
        if token != captioner.pad_id:
            generated.append(token)

    words = []
    for t in generated:
        # idx2word might be keyed by int or str depending on how it was loaded
        word = captioner.idx2word.get(t, captioner.idx2word.get(str(t), "<unk>"))
        if word == START_TOKEN or word == END_TOKEN or word == PAD_TOKEN:
            continue
        words.append(word)
    return " ".join(words)


def evaluate_scratch_decoder(
    captioner,
    features,
    image_ids,
    references_by_image,
    max_len=20,
    variant="scratch",
    progress_cb=None,
):
    """Evaluate ImageCaptionerScratch (greedy) on cached features."""
    captions = []
    refs = []

    start = time.time()
    for i, image_id in enumerate(image_ids):
        caption = _scratch_generate_from_feature(captioner, features[i], max_len)
        captions.append(caption)
        refs.append(list(references_by_image.get(image_id, [])))
        if progress_cb is not None:
            progress_cb(i + 1, len(image_ids))
    elapsed = time.time() - start

    return EvaluationResult(
        variant=variant,
        decoder_kind="scratch",
        bleu4=compute_bleu4(refs, captions),
        meteor=compute_meteor(refs, captions),
        elapsed_seconds=elapsed,
        captions=captions,
        references=refs,
        image_ids=list(image_ids),
    )


def _beam_from_feature(captioner, feature, max_len, beam_width):
    """Beam search over the scratch captioner's NumPy layers.

    Mirrors the algorithm in src/rnn_lstm/beam_search.py but starts from a
    pre-extracted feature instead of a raw image.
    """
    from src.shared.activations import softmax

    cnn_feat = feature.astype(np.float32)
    if captioner.dense_proj is not None:
        x_proj = captioner.dense_proj.forward(cnn_feat).ravel()
    else:
        x_proj = cnn_feat

    # warm the RNN/LSTM state with the projected image at t = -1
    states0 = captioner._init_hidden()
    _, states0 = captioner._step(x_proj, states0)

    # each beam: (sequence_so_far, log_prob, hidden_states, finished_flag)
    beams = [([captioner.start_id], 0.0, states0, False)]

    for _ in range(max_len):
        new_beams = []
        for seq, log_p, states, finished in beams:
            if finished:
                new_beams.append((seq, log_p, states, True))
                continue

            token = seq[-1]
            x_t = captioner.embedding.forward(np.array(token, dtype=np.int32))
            h_out, new_states = captioner._step(x_t, states)
            logits = captioner.dense_out.forward(h_out).ravel()
            probs = softmax(logits)
            log_probs = np.log(np.maximum(probs, 1e-12))

            top_k_idx = np.argpartition(-log_probs, beam_width)[:beam_width]
            for next_token in top_k_idx:
                next_token = int(next_token)
                new_log_p = log_p + float(log_probs[next_token])
                new_seq = seq + [next_token]
                is_finished = (next_token == captioner.end_id)
                new_beams.append((new_seq, new_log_p, new_states, is_finished))

        # length-normalised score so short beams don't always win
        new_beams.sort(key=lambda b: b[1] / max(len(b[0]), 1), reverse=True)
        beams = new_beams[:beam_width]

        if all(b[3] for b in beams):
            break

    best = max(beams, key=lambda b: b[1] / max(len(b[0]), 1))
    ids = best[0]

    words = []
    for t in ids:
        word = captioner.idx2word.get(t, captioner.idx2word.get(str(t), "<unk>"))
        if word == START_TOKEN or word == END_TOKEN or word == PAD_TOKEN:
            continue
        words.append(word)
    return " ".join(words)


def evaluate_beam_decoder(
    captioner,
    features,
    image_ids,
    references_by_image,
    max_len=20,
    beam_width=3,
    variant="beam",
    progress_cb=None,
):
    """Beam search version of evaluate_scratch_decoder."""
    captions = []
    refs = []

    start = time.time()
    for i, image_id in enumerate(image_ids):
        caption = _beam_from_feature(captioner, features[i], max_len, beam_width)
        captions.append(caption)
        refs.append(list(references_by_image.get(image_id, [])))
        if progress_cb is not None:
            progress_cb(i + 1, len(image_ids))
    elapsed = time.time() - start

    return EvaluationResult(
        variant=variant,
        decoder_kind="beam_k" + str(beam_width),
        bleu4=compute_bleu4(refs, captions),
        meteor=compute_meteor(refs, captions),
        elapsed_seconds=elapsed,
        captions=captions,
        references=refs,
        image_ids=list(image_ids),
    )


def select_qualitative_examples(result, n_high=4, n_mid=3, n_low=3):
    """Pick caption examples spanning high / medium / low BLEU-4.

    Returns a list of dicts ready to render in the notebook:
    {image_id, score, prediction, references}.
    """
    if len(result.captions) == 0:
        return []

    scores = per_sample_bleu4(result.references, result.captions)
    indexed = list(enumerate(scores))
    indexed.sort(key=lambda t: t[1], reverse=True)

    n_total = len(indexed)
    high = indexed[:n_high]
    if n_low > 0:
        low = indexed[-n_low:]
    else:
        low = []
    mid_start = max(0, n_total // 2 - n_mid // 2)
    mid = indexed[mid_start: mid_start + n_mid]

    chosen_idx = set()
    for i, _ in high:
        chosen_idx.add(i)
    for i, _ in mid:
        chosen_idx.add(i)
    for i, _ in low:
        chosen_idx.add(i)

    examples = []
    for i in sorted(chosen_idx):
        examples.append(
            {
                "image_id": result.image_ids[i] if len(result.image_ids) > 0 else None,
                "score": scores[i],
                "prediction": result.captions[i],
                "references": result.references[i],
            }
        )
    return examples


def compare_results(results):
    """Flatten a list of EvaluationResults into row dicts for tabulating."""
    return [r.to_dict() for r in results]


def sweep_max_caption_length(eval_fn, max_lens, **eval_kwargs):
    """Re-run eval_fn for each max_len and tag the variant.

    ``eval_fn`` is one of evaluate_keras_decoder, evaluate_scratch_decoder,
    or evaluate_beam_decoder. Other keyword arguments are forwarded.
    """
    results = []
    base_variant = eval_kwargs.pop("variant", "max_len_sweep")
    for L in max_lens:
        kwargs = dict(eval_kwargs)
        kwargs["max_len"] = L
        kwargs["variant"] = base_variant + "_L" + str(L)
        results.append(eval_fn(**kwargs))
    return results
