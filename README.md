# Tugas Besar 2 IF3270 — CNN & RNN/LSTM Image Captioning

**Mata Kuliah:** IF3270 Pembelajaran Mesin  

Implementasi forward propagation CNN, Simple RNN, dan LSTM from scratch, serta pipeline image captioning end-to-end menggunakan arsitektur encoder-decoder (CNN + LSTM/RNN).

---

## Repository Structure

```
tubes2-if3270/
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
│       ├── 01_cnn_training.ipynb
│       ├── 02_cnn_scratch_eval.ipynb
│       ├── 03_rnn_lstm_training.ipynb
│       ├── 04_rnn_lstm_scratch_eval.ipynb
│       └── 05_bonus.ipynb
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
└── TASK_ASSIGNMENT.md
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

TODO: Fill this section

---

## Pembagian Tugas

Lihat `TASK_ASSIGNMENT.md` untuk detail pembagian per anggota.
