from __future__ import annotations

from typing import List, Tuple

import numpy as np

from src.rnn_lstm.decoder import ImageCaptionerScratch


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax over the last axis."""
    m = np.max(logits)
    return logits - m - np.log(np.sum(np.exp(logits - m)))


def beam_search_decode(
    captioner: ImageCaptionerScratch,
    image_path: str,
    k: int = 5,
    max_len: int = 20,
) -> str:
    """
    Generate an image caption using beam search.

    At each timestep every active hypothesis is expanded by the top-k next
    tokens; the global top-k hypotheses (by cumulative log-probability) are
    kept.  A hypothesis is *completed* when it emits <end> or reaches
    max_len.

    Parameters
    ----------
    captioner  : ImageCaptionerScratch
    image_path : str — path to the query image
    k          : int — beam width (number of hypotheses maintained)
    max_len    : int — hard cap on generated sequence length

    Returns
    -------
    str — best caption (words joined by spaces, special tokens stripped)
    """
    # 1. CNN feature + projection
    cnn_feat = captioner._extract_cnn_feature(image_path)

    if captioner.dense_proj is not None:
        x_proj = captioner.dense_proj.forward(cnn_feat).ravel()
    else:
        x_proj = cnn_feat

    # 2. Run projection step to seed the hidden state
    h_init = captioner._init_hidden()
    _, h_init = captioner._step(x_proj, h_init)

    # 3. Initialise k beams
    # Each beam: (cumulative_log_prob, token_id_list, hidden_states)
    beams: List[Tuple[float, List[int], list]] = [
        (0.0, [captioner.start_id], h_init)
    ]
    completed: List[Tuple[float, List[int]]] = []

    # 4. Beam expansion loop
    for _ in range(max_len):
        if not beams:
            break

        candidates: List[Tuple[float, List[int], list]] = []

        for log_prob, tokens, states in beams:
            last_token = tokens[-1]

            # Finished beam — move straight to completed
            if last_token == captioner.end_id:
                completed.append((log_prob, tokens))
                continue

            # One RNN/LSTM step
            x_t = captioner.embedding.forward(
                np.array(last_token, dtype=np.int32)
            )
            h_out, new_states = captioner._step(x_t, states)

            logits = captioner.dense_out.forward(h_out).ravel()
            log_p = _log_softmax(logits)  # (vocab_size,)

            # Top-k token expansions
            top_k_ids = np.argpartition(log_p, -k)[-k:]
            top_k_ids = top_k_ids[np.argsort(log_p[top_k_ids])[::-1]]

            for next_token in top_k_ids:
                new_log_prob = log_prob + float(log_p[next_token])
                candidates.append(
                    (new_log_prob, tokens + [int(next_token)], new_states)
                )

        if not candidates:
            break

        # Prune: keep top-k by cumulative log-probability
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:k]

        # Early exit if every active beam has just emitted <end>
        if all(t[-1] == captioner.end_id for _, t, _ in beams):
            completed.extend([(lp, t) for lp, t, _ in beams])
            beams = []

    # Drain any still-active beams
    completed.extend([(lp, t) for lp, t, _ in beams])

    if not completed:
        return ""

    # 5. Pick the highest-scoring complete hypothesis
    # Normalise by sequence length to avoid bias toward short sequences
    def _score(item: Tuple[float, List[int]]) -> float:
        log_prob, tokens = item
        # Length = number of tokens after <start>, excluding <end>
        n = max(len(tokens) - 1, 1)
        return log_prob / n

    best_log_prob, best_tokens = max(completed, key=_score)

    # 6. Decode token ids -> words
    words: List[str] = []
    for t in best_tokens[1:]:          # skip leading <start>
        if t == captioner.end_id:
            break
        if t != captioner.pad_id and t != captioner.start_id:
            word = captioner.idx2word.get(str(t), captioner.idx2word.get(t, "<unk>"))
            words.append(word)

    return " ".join(words)


def beam_search_batch(
    captioner: ImageCaptionerScratch,
    image_paths: List[str],
    k: int = 5,
    max_len: int = 20,
) -> List[str]:
    """
    Run beam search for a list of images (sequential).

    Parameters
    ----------
    captioner   : ImageCaptionerScratch
    image_paths : list of str
    k           : beam width
    max_len     : maximum caption length

    Returns
    -------
    list of str — one caption per image
    """
    return [beam_search_decode(captioner, p, k=k, max_len=max_len) for p in image_paths]
