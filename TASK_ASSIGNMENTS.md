# TASK ASSIGNMENT — Tugas Besar 2 IF3270
## CNN & RNN/LSTM Image Captioning

**Deadline:** Jumat, 15 Mei 2026  
**Tim:** 3 Orang  
**Total Pekerjaan:** CNN (image classification) + RNN/LSTM (image captioning) + Bonus (ALL)

---

## Ringkasan Pembagian

| Anggota | Domain Utama | File Utama |
|---------|-------------|------------|
| **Dev A** | CNN — layers from scratch, training, evaluasi, Grad-CAM bonus | `src/cnn/`, `src/shared/` |
| **Dev B** | RNN/LSTM — layers from scratch, decoder, beam search bonus | `src/rnn_lstm/layers.py`, `src/rnn_lstm/decoder.py`, `src/rnn_lstm/beam_search.py` |
| **Dev C** | RNN/LSTM — training pipeline, caption preprocessing, evaluasi, laporan | `src/rnn_lstm/model.py`, `src/rnn_lstm/train.py`, `src/rnn_lstm/evaluate.py`, `doc/` |

> **Shared responsibilities:** `src/shared/` dikerjakan bersama di hari 1-2 sebagai fondasi bersama. Backward propagation bonus dibagi rata setelah forward propagation selesai.

---

# DEV A — CNN Specialist

## Tanggung Jawab Utama

Seluruh bagian CNN: utility functions, implementasi from scratch, training Keras, eksperimen hyperparameter, evaluasi, dan bonus Grad-CAM.

---

## [SHARED — Hari 1] Setup Bersama

Kerjakan bersama Dev B dan Dev C:

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

- [ ] `load_image(path, target_size=(224,224))`: `PIL.Image.open` → resize → `np.array` → `/255.0`, handle mode RGB
- [ ] `load_batch(paths, target_size)`: loop `load_image`, stack ke `(N, H, W, C)`
- [ ] `extract_and_save_features(paths, encoder_model, save_path)`: predict batch Keras frozen encoder, `np.save`, skip jika file sudah ada

---

## [CNN Bagian 2] Forward Propagation From Scratch
**File:** `src/cnn/layers.py` — NumPy only, DILARANG import TF/Keras  
**Referensi:** d2l.ai Conv chapter, CS231n notes

### `Conv2DLayer`
Load `layer.get_weights()` → `[kernel (kH,kW,C_in,C_out), bias (C_out,)]`
- [ ] Sliding window forward: setiap posisi `(i,j)` dan filter `k`: dot product patch vs `kernel[:,:,:,k]` + `bias[k]` → aktivasi
- [ ] Handle `padding='same'` dan `padding='valid'`, arbitrary `strides`

### `LocallyConnected2DLayer`
Load kernel shape `(out_rows*out_cols, kH*kW*C_in, C_out)` — non-shared, tiap posisi punya kernel sendiri
- [ ] Forward: posisi `(i,j)` → gunakan `kernel[i*out_cols+j]`

### `MaxPooling2DLayer` / `AveragePooling2DLayer`
- [ ] Sliding window, `pool_size` dan `strides`, `np.max`/`np.mean` per channel

### `GlobalAveragePooling2DLayer` / `GlobalMaxPooling2DLayer`
- [ ] `(H,W,C)` → `(C,)` via `np.mean(x, axis=(0,1))` atau `np.max`

### `FlattenLayer`
- [ ] `x.flatten(order='C')` — row-major, konsisten Keras

### `CNNScratchModel`
- [ ] `load_from_keras(keras_model)` — iterasi layer Keras, buat layer scratch yang sesuai
- [ ] `forward(x)` — jalankan layer secara sekuensial

---

## [CNN Bagian 3] Training Keras
**File:** `src/cnn/model.py`, `src/cnn/train.py`  
**Notebook:** `01_cnn_training.ipynb`  
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
**Notebook:** `02_cnn_scratch_eval.ipynb`

- [ ] Load arsitektur terbaik → jalankan `CNNScratchModel.forward()` pada test set → bandingkan macro F1 vs Keras
- [ ] Ganti Conv2D dengan `LocallyConnected2DLayer` → jalankan ulang
- [ ] Tabel: shared vs non-shared (F1, jumlah parameter, inference time)
- [ ] Plot training/validation loss setiap variasi hyperparameter + analisis kesimpulan

---

## [BONUS] Grad-CAM + Feature Map Visualization
**File:** `src/cnn/gradcam.py`  
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

## Output Dev A
```
src/shared/image_utils.py
src/shared/activations.py
src/shared/dense_layer.py
src/cnn/layers.py
src/cnn/model.py
src/cnn/train.py
src/cnn/gradcam.py          [BONUS]
src/notebooks/01_cnn_training.ipynb
src/notebooks/02_cnn_scratch_eval.ipynb
models/cnn/*.weights.h5     (12+ model files)
outputs/cnn/                (plots, F1 tables, Grad-CAM images)
```

---

---

# DEV B — RNN/LSTM From-Scratch & Decoder Specialist

## Tanggung Jawab Utama

Implementasi from scratch semua layer RNN/LSTM, decoder inference pipeline (greedy + beam search), dan bonus batch inference.

---

## [SHARED — Hari 1] Setup Bersama

Sama dengan Dev A — kerjakan `src/shared/` bersama.

---

## [RNN/LSTM Bagian 0] Forward Propagation From Scratch
**File:** `src/rnn_lstm/layers.py` — NumPy only  
**Referensi:** d2l.ai RNN/LSTM from Scratch, Keras weight format docs

### `EmbeddingLayer`
Load `layer.get_weights()[0]` → matrix `(vocab_size, embed_dim)`
- [ ] `forward(token_ids)`: `return self.embedding_matrix[token_ids]`
- [ ] Support shape `(seq_len,)` dan `(batch, seq_len)`

### `SimpleRNNCell`
Load `layer.get_weights()` → `[W_x (input_dim, hidden_dim), W_h (hidden_dim, hidden_dim), b (hidden_dim,)]`
- [ ] `forward(x_t, h_prev)`: `return np.tanh(x_t @ W_x + h_prev @ W_h + b)`

### `LSTMCell`
Load `layer.get_weights()` → `[kernel (input_dim, 4*hidden_dim), recurrent_kernel (hidden_dim, 4*hidden_dim), bias (4*hidden_dim,)]`
Urutan gate Keras: `i, f, c, o`
- [ ] Split gates: `i = sigmoid(...)`, `f = sigmoid(...)`, `g = tanh(...)`, `o = sigmoid(...)`
- [ ] `c_t = f * c_prev + i * g`
- [ ] `h_t = o * tanh(c_t)`
- [ ] `forward(x_t, h_prev, c_prev) -> (h_t, c_t)`

### `SimpleRNNLayer` / `LSTMLayer`
Wrapper iterasi timestep:
- [ ] `forward(x_seq, h0=None, c0=None)` — x_seq shape `(seq_len, input_dim)`
- [ ] Support `return_sequences=True` dan `False`
- [ ] Support stacking (deep RNN/LSTM) — list of cells

---

## [RNN/LSTM Bagian 4] Decoder Inference Pipeline From Scratch
**File:** `src/rnn_lstm/decoder.py`

```
ImageCaptionerScratch:
  generate_caption(image_path, max_len=20):
    1. load_image → preprocess
    2. extract CNN feature (Keras frozen encoder, numpy output)
    3. x_{-1} = dense_proj.forward(cnn_feat)   [pre-inject]
    4. h0=zeros, c0=zeros
    5. run x_{-1} through LSTM → updated hidden state
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
**Referensi:** d2l.ai Beam Search

- [ ] `beam_search_decode(captioner, image_path, k=5, max_len=20)`:
  - Maintain k hipotesis terbaik dengan log-probability
  - Setiap timestep: expand setiap hipotesis dengan top-k next tokens → prune ke k terbaik
  - Return sequence dengan log-prob tertinggi yang sudah lengkap (`<end>`)
- [ ] Test untuk `k=3` dan `k=5`
- [ ] Bandingkan BLEU-4 beam search vs greedy + contoh caption kualitatif

---

## [BONUS] Batch Inference From Scratch
**Koordinasi:** Dev A (CNN layers) dan Dev C (evaluasi)

- [ ] Modifikasi `Conv2DLayer.forward(x)`: handle shape `(N, H, W, C)`
- [ ] Modifikasi `EmbeddingLayer.forward(token_ids)`: handle `(N, seq_len)`
- [ ] Modifikasi `LSTMCell.forward(x_t, h_prev, c_prev)`: handle `(N, dim)` — matrix ops handle ini secara natural
- [ ] Tambahkan parameter `batch_size` ke semua entry-point inference
- [ ] Verifikasi output batch identik dengan loop sequential

---

## [BONUS] Backward Propagation — RNN/LSTM
- [ ] `SimpleRNNCell.backward(grad_h)` → `grad_x`, `grad_W_x`, `grad_W_h`, `grad_b`, `grad_h_prev` — BPTT satu step
- [ ] `LSTMCell.backward(grad_h, grad_c)` → semua gradient via chain rule melalui gates
- [ ] `EmbeddingLayer.backward(grad_out, token_ids)` → accumulate gradient ke embedding matrix

---

## Output Dev B
```
src/rnn_lstm/layers.py               (EmbeddingLayer, SimpleRNNCell, LSTMCell, wrappers)
src/rnn_lstm/decoder.py              (ImageCaptionerScratch, greedy decode)
src/rnn_lstm/beam_search.py          [BONUS]
src/notebooks/04_rnn_lstm_scratch_eval.ipynb  (bersama Dev C)
```

---

---

# DEV C — RNN/LSTM Training, Evaluation & Laporan

## Tanggung Jawab Utama

Caption preprocessing, Keras training pipeline, evaluasi (BLEU/METEOR), eksperimen hyperparameter, init-inject bonus, dan seluruh laporan PDF.

---

## [SHARED — Hari 1] Setup Bersama

Sama dengan Dev A dan Dev B.

---

## [RNN/LSTM Bagian 1] Feature Extraction
Bergantung pada `extract_and_save_features` dari Dev A.

- [ ] Jalankan extraction seluruh Flickr8k (8.092 gambar) dengan InceptionV3 atau VGG16 tanpa top layer, bobot ImageNet, frozen
- [ ] Simpan ke `data/flickr8k/features/features.npy` (shape: `(8092, feature_dim)`)
- [ ] Simpan mapping `image_filename → index` ke `data/flickr8k/features/index_map.json`

---

## [RNN/LSTM Bagian 2] Caption Preprocessing
**File:** `src/rnn_lstm/caption_preprocessing.py`

- [ ] `preprocess_captions(captions_file)`: lowercase, hapus tanda baca (`re.sub`), tambahkan `<start>` dan `<end>`
- [ ] `build_vocabulary(captions_train, min_freq=2)`: hitung frekuensi, filter, tambah special tokens `<pad>=0, <start>=1, <end>=2, <unk>=3`
- [ ] `tokenize_and_pad(captions, word2idx, max_len=35)`: map kata → integer, padding `<pad>`, return `(N, max_len)`
- [ ] Simpan `vocab.json`, `sequences_train.npy`, dll. ke `data/flickr8k/captions/`

---

## [RNN/LSTM Bagian 3] Training Keras Decoder
**File:** `src/rnn_lstm/model.py`, `src/rnn_lstm/train.py`  
**Notebook:** `03_rnn_lstm_training.ipynb`

Arsitektur decoder (pre-inject):
```
Input: [projected_cnn_feat, emb(<start>), emb(S_0), ..., emb(S_{N-1})]  (seq_len+1, embed_dim)
→ SimpleRNN atau LSTM (h0=zeros)
→ Dense(vocab_size, softmax)
Loss: SparseCategoricalCrossentropy, Optimizer: Adam, Teacher Forcing
```

Hyperparameter sweep — SAMA untuk RNN dan LSTM:
- [ ] **Jumlah layer recurrent** — 3 variasi: 1, 2, 3 layers
- [ ] **Ukuran hidden state** — 2 variasi: 128, 512

Minimal 6 variasi × 2 decoder = 12 training runs. Simpan ke `models/rnn/` dan `models/lstm/` sebagai `.weights.h5` + history `.json`.

---

## [RNN/LSTM Bagian 5] Eksperimen & Evaluasi
**File:** `src/rnn_lstm/evaluate.py`  
**Notebook:** `04_rnn_lstm_scratch_eval.ipynb` (bersama Dev B)

### Metrik
- [ ] `compute_bleu4(references, hypotheses)` — gunakan `sacrebleu` atau `nltk`
- [ ] `compute_meteor(references, hypotheses)` — gunakan `nltk.translate.meteor_score`

### Eksperimen

**a) Variasi jumlah layer dan hidden state:**
- [ ] Jalankan semua 12 variasi → BLEU-4 dan METEOR pada test set untuk RNN dan LSTM
- [ ] Plot training/validation loss per epoch setiap variasi
- [ ] Tabel perbandingan + kesimpulan: pengaruh jumlah layer dan hidden state size

**b) Keras vs From Scratch:**
- [ ] Pilih 1 variasi terbaik per decoder
- [ ] Keras inference → catat BLEU-4 dan inference time
- [ ] Scratch inference (dari Dev B) → catat BLEU-4 dan inference time
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
**Tambahkan ke:** `src/rnn_lstm/model.py`  
**Referensi:** Tanti et al. 2017

- [ ] `build_decoder_init_inject(...)`: image feature sebagai initial hidden state `h0 = Dense(feat_dim → hidden_dim)(image_feature)`, bukan sebagai input awal sequence
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

## Output Dev C
```
src/rnn_lstm/caption_preprocessing.py
src/rnn_lstm/model.py               (pre-inject + init-inject [BONUS])
src/rnn_lstm/train.py
src/rnn_lstm/evaluate.py
src/notebooks/03_rnn_lstm_training.ipynb
src/notebooks/04_rnn_lstm_scratch_eval.ipynb  (bersama Dev B)
models/rnn/*.weights.h5             (6+ model files)
models/lstm/*.weights.h5            (6+ model files)
outputs/rnn/, outputs/lstm/         (plots, BLEU tables, caption examples)
doc/laporan.pdf
```

---

---

# Timeline & Milestones

| Hari | Tanggal | Milestone |
|------|---------|-----------|
| 1–2 | 5–6 Mei | Setup repo + shared utilities selesai. Dev A mulai CNN layers. Dev B mulai RNN/LSTM layers. Dev C setup dataset + caption preprocessing. |
| 3–4 | 7–8 Mei | Dev A: CNN from scratch done. Dev B: EmbeddingLayer + LSTMCell done. Dev C: feature extraction + preprocessing selesai. |
| 5–6 | 9–10 Mei | Dev A: CNN training sweep selesai. Dev B: greedy decoder + beam search done. Dev C: 12 training runs selesai. |
| 7–8 | 11–12 Mei | Dev A: evaluasi + Grad-CAM. Dev B: batch inference done. Dev C: evaluasi BLEU/METEOR + semua eksperimen. |
| 9–10 | 13–14 Mei | Dev A+B: backward propagation. Dev C: laporan draft. Review bersama. |
| 11 | 15 Mei | Final check + submission via Edunex (NIM terkecil). |

---

# Dependency Graph

```
shared/image_utils.py ──┬──→ cnn/train.py
shared/activations.py ──┤    rnn_lstm/train.py
shared/dense_layer.py ──┘    rnn_lstm/decoder.py

cnn/layers.py ──────────────────────→ 02_cnn_scratch_eval.ipynb
cnn/model.py + cnn/train.py ────────→ 01_cnn_training.ipynb
cnn/gradcam.py ─────────────────────→ 05_bonus.ipynb [BONUS]

rnn_lstm/layers.py ─────────────────→ rnn_lstm/decoder.py
rnn_lstm/decoder.py ────────────────→ 04_rnn_lstm_scratch_eval.ipynb
rnn_lstm/beam_search.py ────────────→ 05_bonus.ipynb [BONUS]
rnn_lstm/caption_preprocessing.py ──→ rnn_lstm/train.py
rnn_lstm/model.py + train.py ───────→ 03_rnn_lstm_training.ipynb
rnn_lstm/evaluate.py ───────────────→ 04_rnn_lstm_scratch_eval.ipynb
```

---

# Interface Contracts (antar Dev)

### Dev A → Dev B & C: `extract_and_save_features`
```
Output .npy shape: (N_images, feature_dim)
Output index_map.json: {"filename.jpg": 0, ...}
```

### Dev A → Dev B: `CNNScratchModel`
```
model.forward(x)  → x shape (H,W,C) atau (N,H,W,C) [batch BONUS]
Returns: np.ndarray logits
```

### Dev B → Dev C: `ImageCaptionerScratch`
```
captioner.generate_caption(image_path, max_len) -> str
captioner.generate_captions(image_paths, max_len) -> List[str]
```

### Dev C → Dev B: Model weight files
```
models/lstm/variant_name.weights.h5
models/rnn/variant_name.weights.h5
models/lstm/variant_name_config.json   ← arsitektur info untuk auto-load
```