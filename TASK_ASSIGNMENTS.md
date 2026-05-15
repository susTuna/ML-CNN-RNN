# TASK ASSIGNMENT — Tugas Besar 2 IF3270
## CNN & RNN/LSTM Image Captioning

**Deadline:** Jumat, 15 Mei 2026  
**Tim:** 3 Orang  
**Total Pekerjaan:** CNN (image classification) + RNN/LSTM (image captioning) + Bonus (semua)

---

## Ringkasan Pembagian

| Anggota | Domain Utama | File Utama |
|---------|-------------|------------|
| **13523134** | CNN — layers from scratch, training, evaluasi, Grad-CAM bonus | `src/cnn/conv2d.py`, `src/cnn/locally_connected.py`, `src/cnn/pooling.py`, `src/cnn/flatten.py`, `src/cnn/scratch_model.py`, `src/cnn/model.py`, `src/cnn/train.py`, `src/shared/` |
| **13523147** | RNN/LSTM — layers from scratch, decoder, beam search bonus | `src/rnn_lstm/embedding.py`, `src/rnn_lstm/rnn.py`, `src/rnn_lstm/lstm.py`, `src/rnn_lstm/decoder.py`, `src/rnn_lstm/beam_search.py` |
| **13523150** | RNN/LSTM — training pipeline, caption preprocessing, evaluasi, laporan | `src/rnn_lstm/caption_preprocessing.py`, `src/rnn_lstm/model.py`, `src/rnn_lstm/model_init_inject.py`, `src/rnn_lstm/train.py`, `src/rnn_lstm/metrics.py`, `src/rnn_lstm/evaluate.py`, `doc/` |

> **Shared responsibilities:** `src/shared/` dikerjakan bersama di hari 1-2 sebagai fondasi bersama. Backward propagation bonus dibagi rata setelah forward propagation selesai.

**Notebook:** Semua eksperimen dan pengujian dikerjakan dalam satu file `src/notebooks/notebook.ipynb`, dengan section per domain.

---

## Struktur Notebook

```
src/notebooks/notebook.ipynb
  §1  Setup & Shared Utilities
  §2  CNN — Training (hyperparameter sweep)
  §3  CNN — From Scratch & Evaluation
  §4  CNN — Grad-CAM [Bonus]
  §5  RNN/LSTM — Feature Extraction + Caption Preprocessing
  §6  RNN/LSTM — Training (RNN & LSTM variants)
  §7  RNN/LSTM — From Scratch & Evaluation
  §8  RNN/LSTM — Beam Search / Init-Inject [Bonus]
```

13523134 mengerjakan §1–4, 13523147 membantu §7–8 (scratch inference), 13523150 mengerjakan §5–7 (training & evaluasi).

---

---

# 13523134 — CNN Specialist

## Tanggung Jawab Utama

Seluruh bagian CNN: utility functions, implementasi from scratch, training Keras, eksperimen hyperparameter, evaluasi, dan bonus Grad-CAM.

---

## [SHARED — Hari 1] Setup Bersama

Kerjakan bersama 13523147 dan 13523150:

**`src/shared/image_utils.py`**
- `load_image(path, target_size) -> np.ndarray` — PIL load, resize, normalize ke [0,1]
- `load_batch(paths, target_size) -> np.ndarray` — shape (N, H, W, C)
- `extract_and_save_features(paths, keras_encoder, out_path)` — ekstraksi fitur CNN frozen, simpan `.npy`

**`src/shared/activations.py`**
- `relu(x)`, `softmax(x)`, `sigmoid(x)`, `tanh(x)` — semua NumPy only

**`src/shared/dense_layer.py`**
- Class `DenseLayer` — reuse dari Tubes 1 (load bobot Keras, forward pass `W·x + b`, support aktivasi)

---

## [CNN Bagian 1] Utility Functions
**File:** `src/shared/image_utils.py`  
**Notebook:** `§1 Setup & Shared Utilities`

- [ ] `load_image(path, target_size=(224,224))`: `PIL.Image.open` → resize → `np.array` → `/255.0`, handle mode RGB
- [ ] `load_batch(paths, target_size)`: loop `load_image`, stack ke `(N, H, W, C)`
- [ ] `extract_and_save_features(paths, encoder_model, save_path)`: predict batch Keras frozen encoder, `np.save`, skip jika file sudah ada

---

## [CNN Bagian 2] Forward Propagation From Scratch
**File:** (decoupled — lihat bawah) — NumPy only, DILARANG import TF/Keras  
**Notebook:** `§3 CNN — From Scratch & Evaluation`  
**Referensi:** d2l.ai Conv chapter, CS231n notes

> **Struktur file (decoupled):**
> ```
> src/cnn/
> ├── conv2d.py             ← Conv2DLayer
> ├── locally_connected.py  ← LocallyConnected2DLayer
> ├── pooling.py            ← MaxPooling2DLayer, AveragePooling2DLayer,
> │                              GlobalAveragePooling2DLayer, GlobalMaxPooling2DLayer
> ├── flatten.py            ← FlattenLayer
> ├── scratch_model.py      ← CNNScratchModel
> ├── layers.py             ← shim: re-export semua kelas di atas
> ├── __init__.py           ← re-export package-level
> ├── model.py              ← Keras training model (13523134 Bagian 3)
> ├── train.py              ← training pipeline (13523134 Bagian 3)
> └── gradcam.py            ← Grad-CAM [BONUS]
> ```
> `layers.py` hanya berisi `from .conv2d import ...` dst.
> — tidak ada logika di sana — untuk menjaga kompatibilitas dengan spec.

### `Conv2DLayer` — `src/cnn/conv2d.py`
Load `layer.get_weights()` → `[kernel (kH,kW,C_in,C_out), bias (C_out,)]`
- [ ] Sliding window forward: setiap posisi `(i,j)` dan filter `k`: dot product patch vs `kernel[:,:,:,k]` + `bias[k]` → aktivasi
- [ ] Handle `padding='same'` dan `padding='valid'`, arbitrary `strides`
- [ ] `from_keras_layer(keras_layer)` classmethod

### `LocallyConnected2DLayer` — `src/cnn/locally_connected.py`
Load kernel shape `(out_rows*out_cols, kH*kW*C_in, C_out)` — non-shared, tiap posisi punya kernel sendiri
- [ ] Forward: posisi `(i,j)` → gunakan `kernel[i*out_cols+j]`
- [ ] `from_keras_layer(keras_layer)` classmethod

### `MaxPooling2DLayer` / `AveragePooling2DLayer` — `src/cnn/pooling.py`
- [ ] Sliding window, `pool_size` dan `strides`, `np.max`/`np.mean` per channel
- [ ] `from_keras_layer(keras_layer)` classmethod untuk masing-masing

### `GlobalAveragePooling2DLayer` / `GlobalMaxPooling2DLayer` — `src/cnn/pooling.py`
- [ ] `(H,W,C)` → `(C,)` via `np.mean(x, axis=(0,1))` atau `np.max`
- [ ] `from_keras_layer(keras_layer)` classmethod untuk masing-masing

### `FlattenLayer` — `src/cnn/flatten.py`
- [ ] `x.flatten(order='C')` — row-major, konsisten Keras
- [ ] `from_keras_layer(keras_layer)` classmethod

### `CNNScratchModel` — `src/cnn/scratch_model.py`
- [ ] `load_from_keras(keras_model)` — iterasi layer Keras, buat layer scratch yang sesuai
- [ ] `forward(x)` — jalankan layer secara sekuensial

---

## [CNN Bagian 3] Training Keras
**File:** `src/cnn/model.py`, `src/cnn/train.py`  
**Notebook:** `§2 CNN — Training`  
**Dataset:** Intel Image Classification (25k gambar, 6 kelas)

Arsitektur baseline: `Input → [Conv2D → MaxPool] × N → GlobalAvgPool → Dense(128) → Dense(6, softmax)`  
Loss: `SparseCategoricalCrossentropy`, Optimizer: `Adam`, Metrik: macro F1-score

Hyperparameter sweep (semua Conv2D shared):
- [ ] **Jumlah layer konvolusi** — 3 variasi, contoh: 1, 3, 5 conv blocks
- [ ] **Banyak filter per layer** — 3 variasi kombinasi, contoh: [32,64] vs [64,128,256] vs [16,32,64]
- [ ] **Ukuran filter (kernel size)** — 3 variasi: 3×3, 5×5, 7×7
- [ ] **Jenis pooling** — 2 variasi: MaxPooling2D vs AveragePooling2D

Simpan semua bobot ke `models/cnn/`, catat macro F1 dan training history `.json`.

---

## [CNN Bagian 4] Evaluasi
**Notebook:** `§3 CNN — From Scratch & Evaluation`

- [ ] Load arsitektur terbaik → jalankan `CNNScratchModel.forward()` pada test set → bandingkan macro F1 vs Keras
- [ ] Ganti Conv2D dengan `LocallyConnected2DLayer` → jalankan ulang
- [ ] Tabel: shared vs non-shared (F1, jumlah parameter, inference time)
- [ ] Plot training/validation loss setiap variasi hyperparameter + analisis kesimpulan

---

## [BONUS] Grad-CAM + Feature Map Visualization
**File:** `src/cnn/gradcam.py`  
**Notebook:** `§4 CNN — Grad-CAM`  
**Referensi:** Selvaraju et al. 2016

- [ ] `get_intermediate_feature_maps(model, image, layer_names)` — visualisasi activation maps
- [ ] `grad_cam(model, image, class_idx, last_conv_layer_name)` — gradient → weight feature maps → ReLU → overlay heatmap
- [ ] Tampilkan minimal 5 contoh gambar per kelas dengan heatmap Grad-CAM

---

## [BONUS] Backward Propagation — CNN
- [ ] `Conv2DLayer.backward(grad_output)` → `grad_input`, `grad_kernel`, `grad_bias`
- [ ] `MaxPooling2DLayer.backward()` / `AveragePooling2DLayer.backward()`
- [ ] `FlattenLayer.backward()`, `GlobalAveragePooling2DLayer.backward()`

---

## Output 13523134
```
src/shared/image_utils.py
src/shared/activations.py
src/shared/dense_layer.py
src/shared/__init__.py
src/cnn/conv2d.py             (Conv2DLayer)
src/cnn/locally_connected.py  (LocallyConnected2DLayer)
src/cnn/pooling.py            (MaxPooling2DLayer, AveragePooling2DLayer,
                               GlobalAveragePooling2DLayer, GlobalMaxPooling2DLayer)
src/cnn/flatten.py            (FlattenLayer)
src/cnn/scratch_model.py      (CNNScratchModel)
src/cnn/layers.py            ← shim re-export (backward compat)
src/cnn/__init__.py           (package-level re-exports)
src/cnn/model.py              (Keras training model)
src/cnn/train.py
src/cnn/gradcam.py            [BONUS]
models/cnn/*.weights.h5       (12+ model files)
outputs/cnn/                  (plots, F1 tables, Grad-CAM images)
notebook.ipynb §1, §2, §3, §4
```

---

---

# 13523147 — RNN/LSTM From-Scratch & Decoder Specialist

## Tanggung Jawab Utama

Implementasi from scratch semua layer RNN/LSTM, decoder inference pipeline (greedy + beam search), dan bonus batch inference.

---

## [SHARED — Hari 1] Setup Bersama

Sama dengan 13523134 — kerjakan `src/shared/` bersama.

---

## [RNN/LSTM Bagian 0] Forward Propagation From Scratch
**File:** (decoupled — lihat bawah) — NumPy only  
**Notebook:** `§7 RNN/LSTM — From Scratch & Evaluation`  
**Referensi:** d2l.ai RNN/LSTM from Scratch, Keras weight format docs

> **Struktur file (decoupled):**
> ```
> src/rnn_lstm/
> ├── embedding.py   ← EmbeddingLayer
> ├── rnn.py         ← SimpleRNNCell, SimpleRNNLayer
> ├── lstm.py        ← LSTMCell, LSTMLayer
> ├── layers.py      ← shim: re-export semua kelas di atas
> └── __init__.py    ← re-export package-level
> ```
> `layers.py` hanya berisi `from .embedding import ...` dst.
> — tidak ada logika di sana — untuk menjaga kompatibilitas dengan spec.

### `EmbeddingLayer` — `src/rnn_lstm/embedding.py`
Load `layer.get_weights()[0]` → matrix `(vocab_size, embed_dim)`
- [ ] `forward(token_ids)`: `return self.embedding_matrix[token_ids]`
- [ ] Support shape `(seq_len,)` dan `(batch, seq_len)`
- [ ] `from_keras_layer(keras_layer)` classmethod

### `SimpleRNNCell` — `src/rnn_lstm/rnn.py`
Load `layer.get_weights()` → `[W_x (input_dim, hidden_dim), W_h (hidden_dim, hidden_dim), b (hidden_dim,)]`
- [ ] `forward(x_t, h_prev)`: `return np.tanh(x_t @ W_x + h_prev @ W_h + b)`
- [ ] `from_keras_weights(keras_weights)` / `from_keras_layer(keras_layer)` classmethods

### `SimpleRNNLayer` — `src/rnn_lstm/rnn.py`
Wrapper iterasi timestep di atas satu atau lebih `SimpleRNNCell`:
- [ ] `forward(x_seq, h0=None)` — x_seq shape `(seq_len, input_dim)`
- [ ] Support `return_sequences=True` dan `False`
- [ ] Support stacking (deep RNN) — list of cells
- [ ] `from_keras_layer()` / `from_keras_layers()` classmethods

### `LSTMCell` — `src/rnn_lstm/lstm.py`
Load `layer.get_weights()` → `[kernel (input_dim, 4*hidden_dim), recurrent_kernel (hidden_dim, 4*hidden_dim), bias (4*hidden_dim,)]`  
Urutan gate Keras: `i, f, g, o`
- [ ] Split gates: `i = sigmoid(...)`, `f = sigmoid(...)`, `g = tanh(...)`, `o = sigmoid(...)`
- [ ] `c_t = f * c_prev + i * g`
- [ ] `h_t = o * tanh(c_t)`
- [ ] `forward(x_t, h_prev, c_prev) -> (h_t, c_t)` — supports batch `(N, dim)` naturally
- [ ] `from_keras_weights(keras_weights)` / `from_keras_layer(keras_layer)` classmethods

### `LSTMLayer` — `src/rnn_lstm/lstm.py`
Wrapper iterasi timestep di atas satu atau lebih `LSTMCell`:
- [ ] `forward(x_seq, h0=None, c0=None)` — x_seq shape `(seq_len, input_dim)`
- [ ] Support `return_sequences=True` dan `False`
- [ ] Support stacking (deep LSTM) — list of cells
- [ ] `from_keras_layer()` / `from_keras_layers()` classmethods

---

## [RNN/LSTM Bagian 4] Decoder Inference Pipeline From Scratch
**File:** `src/rnn_lstm/decoder.py`  
**Notebook:** `§7 RNN/LSTM — From Scratch & Evaluation`

```
ImageCaptionerScratch.generate_caption(image_path, max_len):
  1. load_image → preprocess
  2. extract CNN feature (Keras frozen encoder, numpy output)
  3. x_{-1} = dense_proj.forward(cnn_feat)   [pre-inject]
  4. h0=zeros, c0=zeros
  5. run x_{-1} through RNN/LSTM → updated hidden state
  6. token = <start>
  7. loop max_len:
     x_t = embedding.forward(token)
     h_t = rnn/lstm.forward(x_t, h_prev)
     logits = dense_out.forward(h_t)
     token = argmax(softmax(logits))
     if token == <end>: break
  8. decode token ids → words
```

- [ ] `load_from_keras(keras_model)` — auto-load semua bobot per layer
- [ ] `generate_caption_greedy(image_path, max_len) -> str`
- [ ] `generate_captions(image_paths, max_len) -> List[str]` — loop sequential

---

## [BONUS] Beam Search Decoder
**File:** `src/rnn_lstm/beam_search.py`  
**Notebook:** `§8 RNN/LSTM — Beam Search / Init-Inject`  
**Referensi:** d2l.ai Beam Search

- [ ] `beam_search_decode(captioner, image_path, k=5, max_len=20)`:
  - Maintain k hipotesis terbaik dengan log-probability
  - Setiap timestep: expand setiap hipotesis dengan top-k next tokens → prune ke k terbaik
  - Return sequence dengan log-prob tertinggi yang sudah lengkap (`<end>`)
- [ ] Test untuk `k=3` dan `k=5`
- [ ] Bandingkan BLEU-4 beam search vs greedy + contoh caption kualitatif

---

## [BONUS] Batch Inference From Scratch
- [ ] Modifikasi `Conv2DLayer.forward(x)`: handle shape `(N, H, W, C)`
- [ ] Modifikasi `EmbeddingLayer.forward(token_ids)`: handle `(N, seq_len)`
- [ ] Modifikasi `LSTMCell.forward(x_t, h_prev, c_prev)`: handle `(N, dim)`
- [ ] Tambahkan parameter `batch_size` ke semua entry-point inference
- [ ] Verifikasi output batch identik dengan loop sequential

---

## [BONUS] Backward Propagation — RNN/LSTM
- [ ] `SimpleRNNCell.backward(grad_h)` → `grad_x`, `grad_W_x`, `grad_W_h`, `grad_b`, `grad_h_prev`
- [ ] `LSTMCell.backward(grad_h, grad_c)` → semua gradient via chain rule melalui gates
- [ ] `EmbeddingLayer.backward(grad_out, token_ids)` → accumulate gradient ke embedding matrix

---

## Output 13523147
```
src/rnn_lstm/embedding.py            (EmbeddingLayer)
src/rnn_lstm/rnn.py                  (SimpleRNNCell, SimpleRNNLayer)
src/rnn_lstm/lstm.py                 (LSTMCell, LSTMLayer)
src/rnn_lstm/layers.py              ← shim re-export (backward compat)
src/rnn_lstm/__init__.py             (package-level re-exports)
src/rnn_lstm/decoder.py              (ImageCaptionerScratch, greedy decode)
src/rnn_lstm/beam_search.py          [BONUS]
notebook.ipynb §7, §8               (bersama 13523150)
```

---

---

# 13523150 — RNN/LSTM Training, Evaluation & Laporan

## Tanggung Jawab Utama

Caption preprocessing, Keras training pipeline, evaluasi (BLEU/METEOR), eksperimen hyperparameter, init-inject bonus, dan seluruh laporan PDF.

---

## [SHARED — Hari 1] Setup Bersama

Sama dengan 13523134 dan 13523147.

---

## [RNN/LSTM Bagian 1] Feature Extraction
**Notebook:** `§5 RNN/LSTM — Feature Extraction + Caption Preprocessing`  
Bergantung pada `extract_and_save_features` dari 13523134.

- [ ] Jalankan extraction seluruh Flickr8k (8.092 gambar) dengan InceptionV3 atau VGG16 tanpa top layer, bobot ImageNet, frozen
- [ ] Simpan ke `data/flickr8k/features/features.npy` (shape: `(8092, feature_dim)`)
- [ ] Simpan mapping `image_filename → index` ke `data/flickr8k/features/index_map.json`

---

## [RNN/LSTM Bagian 2] Caption Preprocessing
**File:** `src/rnn_lstm/caption_preprocessing.py`  
**Notebook:** `§5 RNN/LSTM — Feature Extraction + Caption Preprocessing`

- [ ] `preprocess_captions(captions_file)`: lowercase, hapus tanda baca (`re.sub`), tambahkan `<start>` dan `<end>`
- [ ] `build_vocabulary(captions_train, min_freq=2)`: hitung frekuensi, filter, tambah special tokens `<pad>=0, <start>=1, <end>=2, <unk>=3`
- [ ] `tokenize_and_pad(captions, word2idx, max_len=35)`: map kata → integer, padding `<pad>`, return `(N, max_len)`
- [ ] Simpan `vocab.json`, `sequences_train.npy`, dll. ke `data/flickr8k/captions/`

---

## [RNN/LSTM Bagian 3] Training Keras Decoder
**File:** `src/rnn_lstm/model.py`, `src/rnn_lstm/train.py`  
**Notebook:** `§6 RNN/LSTM — Training`

> **Struktur file (decoupled):**
> ```
> src/rnn_lstm/
> ├── model.py               ← build_decoder_pre_inject(...) — arsitektur utama
> ├── model_init_inject.py   ← build_decoder_init_inject(...) [BONUS]
> ├── train.py               ← training loop + history logging
> ├── metrics.py             ← compute_bleu4, compute_meteor (pure functions)
> └── evaluate.py            ← evaluation orchestration (imports metrics.py)
> ```

### `model.py` — Pre-Inject Decoder
Arsitektur decoder (pre-inject):
```
Input: [projected_cnn_feat, emb(<start>), emb(S_0), ..., emb(S_{N-1})]  (seq_len+1, embed_dim)
→ SimpleRNN atau LSTM (h0=zeros)
→ Dense(vocab_size, softmax)
Loss: SparseCategoricalCrossentropy, Optimizer: Adam, Teacher Forcing
```
- [ ] `build_decoder_pre_inject(vocab_size, embed_dim, hidden_dim, num_layers, rnn_type)` → Keras model

### `train.py` — Training Pipeline
- [ ] `train_model(model, train_data, val_data, epochs, ...)` → history dict
- [ ] Simpan bobot ke `models/rnn/` atau `models/lstm/` sebagai `.weights.h5` + history `.json`

Hyperparameter sweep — SAMA untuk RNN dan LSTM:
- [ ] **Jumlah layer recurrent** — 3 variasi: 1, 2, 3 layers
- [ ] **Ukuran hidden state** — 2 variasi: 128, 512

Minimal 6 variasi × 2 decoder = 12 training runs.

---

## [RNN/LSTM Bagian 5] Eksperimen & Evaluasi
**File:** `src/rnn_lstm/metrics.py`, `src/rnn_lstm/evaluate.py`  
**Notebook:** `§7 RNN/LSTM — From Scratch & Evaluation` (bersama 13523147)

### `metrics.py` — Pure Metric Functions
- [ ] `compute_bleu4(references, hypotheses)` — gunakan `sacrebleu` atau `nltk`
- [ ] `compute_meteor(references, hypotheses)` — gunakan `nltk.translate.meteor_score`

### `evaluate.py` — Evaluation Orchestration
Menggunakan `metrics.py` + `decoder.py` dari 13523147.

### Eksperimen

**a) Variasi jumlah layer dan hidden state:**
- [ ] Jalankan semua 12 variasi → BLEU-4 dan METEOR pada test set untuk RNN dan LSTM
- [ ] Plot training/validation loss per epoch setiap variasi
- [ ] Tabel perbandingan + kesimpulan

**b) Keras vs From Scratch:**
- [ ] Pilih 1 variasi terbaik per decoder
- [ ] Keras inference → catat BLEU-4 dan inference time
- [ ] Scratch inference (dari 13523147) → catat BLEU-4 dan inference time
- [ ] Analisis perbedaan

**c) RNN vs LSTM:**
- [ ] Bandingkan BLEU-4, METEOR, inference time
- [ ] **Qualitative analysis:** minimal 10 gambar — caption ground truth, RNN output, LSTM output (pilih contoh score tinggi/sedang/rendah)
- [ ] Analisis: vanishing gradient, long-term memory, mengapa satu arsitektur lebih baik

**d) Pengaruh max caption length:**
- [ ] Ambil arsitektur terbaik, variasikan `max_len`: minimal 3 variasi (contoh: 10, 20, 35)
- [ ] Catat BLEU-4 setiap `max_len` + kesimpulan

---

## [BONUS] Init-Inject Architecture
**File:** `src/rnn_lstm/model_init_inject.py`  
**Notebook:** `§8 RNN/LSTM — Beam Search / Init-Inject`  
**Referensi:** Tanti et al. 2017

- [ ] `build_decoder_init_inject(vocab_size, embed_dim, hidden_dim, feat_dim, rnn_type)`: image feature sebagai `h0 = Dense(feat_dim → hidden_dim)(image_feature)`, bukan sebagai input awal sequence
- [ ] Latih satu versi RNN dan satu LSTM dengan init-inject
- [ ] Bandingkan BLEU-4 dengan pre-inject: tabel + analisis

---

## [Laporan PDF]
**Folder:** `doc/`

- [ ] **Cover** — Nama, NIM, kelas, judul
- [ ] **Deskripsi Persoalan**
- [ ] **Pembahasan & Penjelasan Implementasi** — deskripsi class, atribut, method; penjelasan forward propagation CNN, RNN, LSTM dengan rumus
- [ ] **Hasil Pengujian CNN** — shared vs non-shared, pengaruh jumlah layer/filter/kernel size/pooling, Keras vs scratch
- [ ] **Hasil Pengujian RNN/LSTM** — perbandingan jumlah layer & hidden state (RNN+LSTM), RNN vs LSTM, Keras vs scratch, pengaruh max caption length
- [ ] **Kesimpulan dan Saran**
- [ ] **Pembagian Tugas**
- [ ] **Referensi**
- [ ] **Lampiran: Form Penggunaan AI**

---

## Output 13523150
```
src/rnn_lstm/caption_preprocessing.py  (preprocess_captions, build_vocabulary, tokenize_and_pad)
src/rnn_lstm/model.py                  (build_decoder_pre_inject)
src/rnn_lstm/model_init_inject.py      (build_decoder_init_inject) [BONUS]
src/rnn_lstm/train.py                  (train_model, history logging)
src/rnn_lstm/metrics.py                (compute_bleu4, compute_meteor)
src/rnn_lstm/evaluate.py               (evaluation orchestration)
models/rnn/*.weights.h5                (6+ model files)
models/lstm/*.weights.h5               (6+ model files)
outputs/rnn/, outputs/lstm/            (plots, BLEU tables, caption examples)
doc/laporan.pdf
notebook.ipynb §5, §6, §7             (bersama 13523147 untuk §7)
```

---

---

# Timeline & Milestones

| Hari | Tanggal | Milestone |
|------|---------|-----------|
| 1–2 | 5–6 Mei | Setup repo + shared utilities selesai. 13523134 mulai CNN layers. 13523147 mulai RNN/LSTM layers. 13523150 setup dataset + caption preprocessing. |
| 3–4 | 7–8 Mei | 13523134: CNN from scratch done. 13523147: EmbeddingLayer + LSTMCell done. 13523150: feature extraction + preprocessing selesai. |
| 5–6 | 9–10 Mei | 13523134: CNN training sweep selesai. 13523147: greedy decoder + beam search done. 13523150: 12 training runs selesai. |
| 7–8 | 11–12 Mei | 13523134: evaluasi + Grad-CAM. 13523147: batch inference done. 13523150: evaluasi BLEU/METEOR + semua eksperimen. |
| 9–10 | 13–14 Mei | 13523134+B: backward propagation. 13523150: laporan draft. Review bersama + merge notebook. |
| 11 | 15 Mei | Final check + submission via Edunex (NIM terkecil). |

---

# Dependency Graph

```
shared/image_utils.py ──┬──→ cnn/train.py
shared/activations.py ──┤    rnn_lstm/train.py
shared/dense_layer.py ──┘    rnn_lstm/decoder.py

cnn/conv2d.py ──────────────────────┐
cnn/locally_connected.py ───────────┤
cnn/pooling.py ─────────────────────┼──→ cnn/scratch_model.py → notebook §3
cnn/flatten.py ─────────────────────┘
cnn/layers.py  (shim) ──────────────→ cnn/scratch_model.py  [alias]
cnn/model.py + cnn/train.py ────────→ notebook §2
cnn/gradcam.py ─────────────────────→ notebook §4 [BONUS]

rnn_lstm/embedding.py ──────────────┐
rnn_lstm/rnn.py ────────────────────┼──→ rnn_lstm/decoder.py
rnn_lstm/lstm.py ───────────────────┘
rnn_lstm/layers.py  (shim) ─────────→ rnn_lstm/decoder.py  [alias]
rnn_lstm/decoder.py ────────────────→ notebook §7
rnn_lstm/beam_search.py ────────────→ notebook §8 [BONUS]
rnn_lstm/caption_preprocessing.py ──→ rnn_lstm/train.py
rnn_lstm/model.py ──────────────────→ rnn_lstm/train.py → notebook §6
rnn_lstm/model_init_inject.py ──────→ rnn_lstm/train.py → notebook §8 [BONUS]
rnn_lstm/metrics.py ────────────────→ rnn_lstm/evaluate.py
rnn_lstm/evaluate.py ───────────────→ notebook §7
```

---

# Interface Contracts (antar Dev)

### 13523134 → 13523147 & C: `extract_and_save_features`
```
Output .npy shape: (N_images, feature_dim)
Output index_map.json: {"filename.jpg": 0, ...}
```

### 13523134 → 13523147: `CNNScratchModel`
```
model.forward(x)  → x shape (H,W,C) atau (N,H,W,C) [batch BONUS]
Returns: np.ndarray logits
```

### 13523147 → 13523150: `ImageCaptionerScratch`
```
captioner.generate_caption(image_path, max_len) -> str
captioner.generate_captions(image_paths, max_len) -> List[str]
```

### 13523150 → 13523147: Model weight files
```
models/lstm/variant_name.weights.h5
models/rnn/variant_name.weights.h5
models/lstm/variant_name_config.json   ← arsitektur info untuk auto-load
```