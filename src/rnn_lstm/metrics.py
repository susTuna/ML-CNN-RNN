import numpy as np


def _tokenize(text):
    return text.strip().split()


def _ensure_nltk_data():
    import nltk

    needed = [
        ("wordnet", "corpora/wordnet"),
        ("omw-1.4", "corpora/omw-1.4"),
        ("punkt", "tokenizers/punkt"),
    ]
    for resource, locator in needed:
        try:
            nltk.data.find(locator)
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pass


def compute_bleu4(references, hypotheses):
    if len(references) != len(hypotheses):
        raise ValueError(
            "references (" + str(len(references)) + ") and hypotheses ("
            + str(len(hypotheses)) + ") must have the same length."
        )
    if len(hypotheses) == 0:
        return 0.0

    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

    refs_tok = []
    for refs in references:
        refs_tok.append([_tokenize(r) for r in refs])
    hyps_tok = [_tokenize(h) for h in hypotheses]

    smoother = SmoothingFunction().method1
    score = corpus_bleu(
        refs_tok,
        hyps_tok,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoother,
    )
    return float(score)


def compute_bleu_n(references, hypotheses, n=4):
    if n < 1:
        raise ValueError("n must be >= 1")
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length.")
    if len(hypotheses) == 0:
        return 0.0

    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

    refs_tok = []
    for refs in references:
        refs_tok.append([_tokenize(r) for r in refs])
    hyps_tok = [_tokenize(h) for h in hypotheses]
    weights = tuple([1.0 / n] * n)

    smoother = SmoothingFunction().method1
    score = corpus_bleu(refs_tok, hyps_tok, weights=weights, smoothing_function=smoother)
    return float(score)


def compute_meteor(references, hypotheses):
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length.")
    if len(hypotheses) == 0:
        return 0.0

    _ensure_nltk_data()
    from nltk.translate.meteor_score import meteor_score

    scores = []
    for refs, hyp in zip(references, hypotheses):
        ref_toks = [_tokenize(r) for r in refs]
        hyp_toks = _tokenize(hyp)
        try:
            s = float(meteor_score(ref_toks, hyp_toks))
        except Exception:
            s = 0.0
        scores.append(s)

    if len(scores) == 0:
        return 0.0
    return float(np.mean(scores))


def per_sample_bleu4(references, hypotheses):
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length.")

    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    smoother = SmoothingFunction().method1
    scores = []
    for refs, hyp in zip(references, hypotheses):
        ref_toks = [_tokenize(r) for r in refs]
        hyp_toks = _tokenize(hyp)
        try:
            s = sentence_bleu(
                ref_toks,
                hyp_toks,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=smoother,
            )
        except Exception:
            s = 0.0
        scores.append(float(s))
    return scores
