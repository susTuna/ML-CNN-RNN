# Tugas Besar 2 IF3270 - CNN & RNN/LSTM Image Captioning

**Mata Kuliah:** IF3270 Pembelajaran Mesin

Implementasi forward propagation CNN, Simple RNN, dan LSTM from scratch, serta pipeline image captioning end-to-end menggunakan arsitektur encoder-decoder (CNN + LSTM/RNN).

---

## Repository Structure

```
ML-CNN-RNN/
├── src/
│   ├── shared/                  # Shared utilities (image utils, activations, dense)
│   │   ├── image_utils.py
│   │   ├── activations.py
│   │   └── dense_layer.py
│   ├── cnn/                     # CNN module
│   │   ├── layers.py            # Conv2D, LocallyConnected2D, Pooling, Flatten (scratch)
│   │   ├── model.py             # Keras CNN model builder
│   │   ├── train.py             # Training + hyperparameter sweep
│   │   └── gradcam.py           # [BONUS] Grad-CAM + feature map visualization
│   ├── rnn_lstm/                # RNN/LSTM module
│   │   ├── layers.py            # Embedding, SimpleRNN cell, LSTM cell (scratch)
│   │   ├── model.py             # Keras decoder model builder
│   │   ├── caption_preprocessing.py
│   │   ├── train.py             # Training + hyperparameter sweep
│   │   ├── decoder.py           # Greedy decoder (scratch inference pipeline)
│   │   ├── beam_search.py       # [BONUS] Beam search decoder
│   │   └── evaluate.py          # BLEU-4, METEOR scoring
│   └── notebooks/
│       └── notebook.ipynb
├── data/
│   ├── intel/                   # Intel Image Classification dataset
│   └── flickr8k/
│       ├── images/
│       ├── features/            # Extracted CNN feature vectors (.npy)
│       └── captions/            # Preprocessed captions + vocab
├── models/
│   ├── cnn/                     # Saved CNN model weights
│   ├── rnn/                     # Saved RNN decoder weights
│   └── lstm/                    # Saved LSTM decoder weights
├── outputs/                     # Evaluation results, plots, generated captions
├── doc/                         # Laporan PDF
├── README.md
```

---

## Setup & Installation

```bash
uv sync
```

Download datasets:

- **Intel Image Classification** → `data/intel/` : [Kaggle Link](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
- **Flickr8k** → `data/flickr8k/images/` + captions file : [Kaggle Link](https://www.kaggle.com/datasets/adityajn105/flickr8k)

---

## How to Run

1. **Install dependencies:**

   ```bash
   uv sync
   ```

2. **Download the datasets** into `data/`:
   - Intel Image Classification → `data/intel/seg_train/seg_train/<class>/` and `data/intel/seg_test/seg_test/<class>/`
   - Flickr8k → `data/flickr8k/images/` (JPGs) and the caption file `captions.txt` in `data/flickr8k/`.

3. **Run the experiments:** open `src/notebooks/notebook.ipynb` in Jupyter and run all cells. The notebook covers all 8 sections:
   - 1. setup, 2. CNN training sweep (16 variants), 3. Keras vs scratch CNN comparison, 4. Grad-CAM (bonus),
   - 5. feature extraction + caption preprocessing, 6. RNN/LSTM training sweep (12 variants), 7. evaluation (BLEU-4/METEOR, Keras vs scratch, max_len sweep), 8. beam search + init-inject (bonus).
        Weights are written to `models/`, evaluation tables and plots to `outputs/`.

4. **Tests:**
   ```bash
   python -m pytest tests/
   ```

---

## Pembagian Tugas

<table align="center">
  <tr>
    <th align="center">User</th>
    <th align="center">Job</th>
  </tr>
  <tr>
    <td align="center">
      <a href="https://github.com/iannn23">
        <img src="https://avatars.githubusercontent.com/u/167867671?v=4" width="80px" style="border-radius: 50%;" alt="iannn23"/><br />
        <sub><b>Sebastian Enrico Nathanael</b></sub>
      </a>
    </td>
    <td align="center">CNN layers from scratch, training & hyperparameter sweep, evaluation, Grad-CAM, backward propagation</td>
  </tr>
  <tr>
    <td align="center">
      <a href="https://github.com/susTuna">
        <img src="https://avatars.githubusercontent.com/u/148179846?s=96&v=4" width="80px" style="border-radius: 50%;" alt="susTuna"/><br />
        <sub><b>Frederiko Eldad Mugiyono</b></sub>
      </a>
    </td>
    <td align="center">RNN/LSTM layers from scratch, decoder & beam search, batch inference, shared utilities, experiments</td>
  </tr>
  <tr>
    <td align="center">
      <a href="https://github.com/BenedictusNelson">
        <img src="https://avatars.githubusercontent.com/u/163625178?v=4" width="80px" style="border-radius: 50%;" alt="BenedictusNelson"/><br />
        <sub><b>Benedictus Nelson</b></sub>
      </a>
    </td>
    <td align="center">Training pipeline, caption preprocessing, evaluation BLEU/METEOR, init-inject, PDF report, experiments /td>
  </tr>
</table>
