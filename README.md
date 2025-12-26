# Đồ án CS221: VIHATET5: Enhancing Hate Speech Detection in Vietnamese With a Unified Text-to-Text Transformer Model

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Transformers](https://img.shields.io/badge/library-transformers-orange.svg)](https://github.com/huggingface/transformers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Một hệ thống toàn diện cho bài toán phát hiện ngôn ngữ thù ghét (Hate Speech) và bình luận độc hại (Toxic Speech) tiếng Việt, sử dụng các kiến trúc SOTA như **PhoBERT/ViSoBERT** và **T5/ViT5**.

> 📄 **Paper**: [ViHATE T5: Enhancing Hate Speech Detection in Vietnamese With a Unified Text-to-Text Transformer Model](https://aclanthology.org/2024.findings-acl.355.pdf) (ACL 2024 Findings)

---

## 📌 Tổng quan dự án

Dự án cung cấp 3 pipeline chính cho phép bạn đi từ dữ liệu thô đến mô hình hoàn chỉnh:
1.  **Pre-training**: Tiếp tục huấn luyện T5 với cơ chế *Span Corruption* trên dữ liệu tiếng Việt.
2.  **T5 Fine-tuning**: Huấn luyện Seq2Seq cho bài toán phân loại đa tập dữ liệu.
3.  **BERT Classification**: Huấn luyện các mô hình Encoder-only (PhoBERT, ViSoBERT) truyền thống.

---

## 👥 Thành viên nhóm

| STT | Họ và Tên | MSSV |
| :---: | :--- | :---: |
| 1 | Trịnh Trân Trân | 23521624 |
| 2 | Phạm Thị Ngọc Bích | 23520148 |
| 3 | Nguyễn Minh Bảo | 23520123 |

---

## 🛠 Cài đặt & Chuẩn bị

### 1. Khởi tạo môi trường
```bash
# Khởi tạo venv
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# Windows: .venv\Scripts\activate

# Cài đặt thư viện
pip install -r requirements.txt
```

### 2. Đăng nhập HuggingFace (Cần thiết để tải/đẩy mô hình)
```bash
huggingface-cli login
# Hoặc thiết lập biến môi trường HF_TOKEN
```

---

## 📊 Dữ liệu (Datasets)

Hệ thống hỗ trợ nạp dữ liệu tự động từ HuggingFace hoặc file local:

| Tên Dataset | Loại | Mô tả |
| :--- | :--- | :--- |
| **ViHSD** | Multi-class | 3 nhãn: CLEAN, OFFENSIVE, HATE |
| **ViCTSD** | Binary | Phát hiện độc hại (Toxic/None) |
| **ViHOS** | Hate Spans | Phát hiện vùng thù ghét |
| **VOZ-HSD** | Binary | Dữ liệu lớn (balanced, hate_only, full) |
| **Custom HF** | Tùy chọn | Bất kỳ dataset nào trên HuggingFace (tự nhận diện cột) |

---

## 📦 Các Model & Dataset đã huấn luyện

> **Collection đầy đủ**: Tất cả các model và dataset của dự án được tổng hợp tại [CS221 - UIT Collection](https://huggingface.co/collections/Minhbao5xx2/cs221-uit) trên HuggingFace.

Dưới đây là các tài nguyên chính được phát triển trong dự án này:

*   **Model Gán nhãn (Labeling)**: [CS221_Labeling_visobert](https://huggingface.co/Minhbao5xx2/CS221_Labeling_visobert) - Model dựa trên ViSoBERT được dùng để gán nhãn tự động cho tập dữ liệu lớn.
*   **Dataset đã gán nhãn**: [re_VOZ-HSD](https://huggingface.co/datasets/Minhbao5xx2/re_VOZ-HSD) - Tập dữ liệu VOZ với hơn 12 triệu dòng đã được xử lý và gán nhãn.
*   **Model Fine-tuned (3-datasets - Hate Only)**: [Hate_only_ViT5](https://huggingface.co/Minhbao5xx2/Hate_only_ViT5) - Mô hình ViT5-base được fine-tune đồng thời trên 3 tập dữ liệu (ViHSD, ViCTSD, ViHOS) khởi tạo từ checkpoint "hate-only".
*   **Model Fine-tuned (3-datasets - Balanced)**: [balance_Vi_T5](https://huggingface.co/Minhbao5xx2/balance_Vi_T5) - Mô hình ViT5-base được fine-tune đồng thời trên 3 tập dữ liệu khởi tạo từ checkpoint "balanced".
*   **Model Fine-tuned (Multi-dataset version)**: [vit5_multi_dataset](https://huggingface.co/Minhbao5xx2/vit5_multi_dataset) - Một phiên bản khác của ViT5-base được huấn luyện bằng pipeline `src/train_t5.py`.
*   **Model Pre-trained (Hate Only)**: [pre_train_ViT5_hate_only](https://huggingface.co/Minhbao5xx2/pre_train_ViT5_hate_only) - Mô hình ViT5 được pre-train bằng cơ chế Span Corruption trên **100,000 mẫu** từ tập dữ liệu VOZ "hate-only".
*   **Model Pre-trained (Balanced)**: [balance_pre_train_Vi_T5](https://huggingface.co/Minhbao5xx2/balance_pre_train_Vi_T5) - Mô hình ViT5 được pre-train bằng cơ chế Span Corruption trên **200,000 mẫu** từ tập dữ liệu VOZ "balanced".

---

## ⚙️ Cấu hình Model & Training Pipeline

Dưới đây là cấu hình chi tiết cho từng giai đoạn huấn luyện trong dự án:

### 1️⃣ **Giai đoạn Pre-training T5 (Span Corruption)**

**Mục tiêu**: Tiếp tục huấn luyện mô hình T5 với cơ chế Span Corruption trên dữ liệu tiếng Việt để tăng khả năng hiểu ngữ cảnh.

**Model Base**: `VietAI/vit5-base` hoặc `google/mt5-base`

**Cấu hình chính**:
```python
# Model & Tokenizer
model_name = "VietAI/vit5-base"
max_length = 256
noise_density = 0.15
mean_noise_span_length = 3.0

# Training Arguments
per_device_train_batch_size = 128  # Tùy GPU
gradient_accumulation_steps = 1
learning_rate = 5e-3
num_train_epochs = 10
warmup_steps = 2000
weight_decay = 0.001
bf16 = True  # Bật mixed precision cho H200/A100

# Optimizer
optim = "adamw_torch"
gradient_checkpointing = True
```

**Dataset**: 
- `Minhbao5xx2/re_VOZ-HSD` (split: `hate_only` hoặc `balanced`)
- Số lượng samples: 100K (hate-only) hoặc 200K (balanced)

**Output**: Checkpoint được lưu tại `vihate_t5_pretrain/` hoặc `--output_dir` tùy chỉnh.

---

### 2️⃣ **Giai đoạn Fine-tuning T5 (Seq2Seq Classification)**

**Mục tiêu**: Fine-tune mô hình T5 (từ checkpoint pre-trained hoặc base) trên các tập dữ liệu hate speech detection.

**Model Base**: 
- Checkpoint từ giai đoạn 1: `vihate_t5_pretrain/final`
- Hoặc trực tiếp: `VietAI/vit5-base`

**Cấu hình chính**:
```python
# Model & Tokenizer
pre_trained_checkpoint = "vihate_t5_pretrain/final"  # hoặc "VietAI/vit5-base"
max_length = 256
target_max_length = 10  # Độ dài label (CLEAN, HATE, OFFENSIVE...)

# Training Arguments
per_device_train_batch_size = 32
per_device_eval_batch_size = 32
gradient_accumulation_steps = 1
learning_rate = 2e-4
num_train_epochs = 4
warmup_ratio = 0.0
weight_decay = 0.01
lr_scheduler_type = "linear"
bf16 = True

# Evaluation
evaluation_strategy = "epoch"
save_strategy = "epoch"
load_best_model_at_end = True
metric_for_best_model = "f1_macro"
```

**Dataset**: 
- `ViHSD`, `ViCTSD`, `ViHOS` (tự động load từ HuggingFace)
- Hoặc tập dữ liệu tùy chỉnh

**Output**: Model được lưu tại `outputs/` hoặc `--output_dir` tùy chỉnh.

---

### 3️⃣ **Giai đoạn Training BERT-based Models (Classification)**

**Mục tiêu**: Huấn luyện các mô hình encoder-only (PhoBERT, ViSoBERT) cho bài toán phân loại truyền thống.

**Cấu hình chính**:
```python
# Model & Tokenizer
model_name = "uitnlp/visobert"
max_length = 256
num_labels = 3  # Tùy dataset (ViHSD: 3, ViCTSD: 2, ViHOS: 2)

# Training Arguments
per_device_train_batch_size = 16
per_device_eval_batch_size = 32
gradient_accumulation_steps = 1
learning_rate = 2e-5
num_train_epochs = 10
warmup_ratio = 0.1
weight_decay = 0.01
patience = 3  # Early stopping

# Optimizer
optim = "adamw_torch"
```

**Dataset**: 
- `ViHSD`, `ViCTSD`, `ViHOS`
- Tự động xử lý label encoding

**Output**: Model được lưu tại `outputs/` hoặc `--output_dir` tùy chỉnh.

---

### 4️⃣ **Giai đoạn Auto-Labeling (Optional)**

**Mục tiêu**: Sử dụng mô hình đã huấn luyện để gán nhãn tự động cho tập dữ liệu lớn.

**Model**: `Minhbao5xx2/CS221_Labeling_visobert`

**Cấu hình chính**:
```python
# Model & Tokenizer
model_name = "Minhbao5xx2/CS221_Labeling_visobert"
max_length = 256
batch_size = 128 
```

**Dataset Input**: Dữ liệu thô (CSV, JSON, Parquet)

**Output**: Dataset đã gán nhãn được đẩy lên HuggingFace Hub.

---

### 📊 **So sánh cấu hình giữa các giai đoạn**

| Giai đoạn | Model Base | Batch Size | Learning Rate | Epochs | Optimizer |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Pre-training T5** | vit5-base | 128 | 5e-3 | 10 | adamw_torch |
| **Fine-tuning T5** | Pre-trained checkpoint | 32 | 2e-4 | 4 | adamw_torch |
| **Training BERT** | phobert/visobert | 16 | 2e-5 | 10 | adamw_torch |
| **Auto-Labeling** | visobert (fine-tuned) | 128 | - | - | - |

---

## 🚀 Hướng dẫn sử dụng (Scripts)

### 1. Pre-training T5 (Span Corruption)
```bash
bash scripts/run_pretrain_t5.sh \
    --dataset_name "Minhbao5xx2/re_VOZ-HSD" \
    --split_name "hate_only" \
    --batch_size 128 \
    --epochs 10 \
    --lr 5e-3
```
*Lưu ý: Mặc định tối ưu cho H200. Với GPU nhỏ, giảm `batch_size` và tăng `gradient_accumulation_steps`.*

### 2. Fine-tuning T5 (Phân loại Seq2Seq)
```bash
bash scripts/run_train_t5.sh \
    --pre_trained_ckpt "vihate_t5_pretrain/final" \
    --batch_size 32 \
    --num_epochs 4 \
    --learning_rate 2e-4 \
    --max_length 256
```

### 3. Huấn luyện BERT/PhoBERT (Classification)
```bash
bash scripts/run_train_bert.sh \
    --dataset "ViHSD" \
    --model_name "vinai/phobert-base" \
    --epochs 10 \
    --batch_size 16
```

---

## ⚙️ Chi tiết tham số (CLI Arguments)

### **Script: run_train_t5.sh & run_pretrain_t5.sh**
| Tham số | Mô tả | T5 Fine-tune | T5 Pre-train |
| :--- | :--- | :--- | :--- |
| `--dataset_name` / `--dataset` | Tên dataset (HF hoặc Local) | ✅ | ✅ |
| `--pre_trained_ckpt` | Model gốc (ViT5, checkpoint...) | ✅ | - |
| `--batch_size` | Batch size mỗi GPU | `32` | `128` |
| `--num_epochs` / `--epochs` | Số epoch huấn luyện | `4` | `10` |
| `--learning_rate` / `--lr` | Tốc độ học (Learning Rate) | `2e-4` | `5e-3` |
| `--max_length` | Độ dài sequence tối đa | `256` | - |
| `--gradient_accumulation_steps`| Tích lũy gradient | `1` | `1` |
| `--weight_decay` | Suy giảm trọng số | `0.01` | `0.001` |
| `--warmup_ratio` / `--warmup_steps`| Tỉ lệ/Số bước khởi động | `0.0` | `2000` |
| `--seed` | Random seed | `42` | - |
---

## 📊 Kết quả Auto-Labeling VOZ-HSD Dataset

### Labeling Performance (ViSoBERT Model)

Mô hình **CS221_Labeling_visobert** được sử dụng để tự động gán nhãn cho tập dữ liệu VOZ-HSD:

| Metric | Kết quả |
| :--- | :---: |
| **Tổng samples đã gán nhãn** | 10,747,733 |
| **Agreement với manual labels** | **97.5%** ✅ |
| **Accuracy** | 97.5% |
| **Processing Time** | Batch processing on H200 GPU |

> **Nhận xét**: Mô hình ViSoBERT đạt độ chính xác cao **97.5%** so với manual labels của tác giả gốc, chứng minh tính hiệu quả của phương pháp auto-labeling. Tập dữ liệu được xử lý hoàn toàn và sẵn sàng để sử dụng cho pre-training và fine-tuning các mô hình T5.
---

## 📊 Kết quả thực nghiệm (Table 3 - Paper)

Dưới đây là kết quả chi tiết trên các tập dữ liệu test của các mô hình BERT-based đã huấn luyện:

### Kết quả chi tiết theo Dataset

#### ViHSD Dataset
| Model | Accuracy | Macro F1 |
| :--- | :---: | :---: |
| **ViSoBERT** | 0.8842 | 0.6871 |
| **DistilBERT** (multilingual) | 0.8615 | 0.6224 |
| **BERT** (multilingual, cased) | 0.8665 | 0.6427 |
| **PhoBERT v2** | 0.8725 | 0.6583 |
| **PhoBERT** | 0.8632 | 0.6360 |
| **viBERT** | 0.8596 | 0.6149 |
| **XLM-RoBERTa** | 0.8692 | 0.6544 |
| **BERT** (multilingual, uncased) | 0.8561 | 0.6161 |

#### ViCTSD Dataset
| Model | Accuracy | Macro F1 |
| :--- | :---: | :---: |
| **ViSoBERT** | 0.9035 | 0.7045 |
| **XLM-RoBERTa** | 0.9015 | 0.7153 |
| **PhoBERT v2** | 0.9023 | 0.7139 |
| **PhoBERT** | 0.9078 | 0.7131 |
| **BERT** (multilingual, cased) | 0.8983 | 0.6710 |
| **BERT** (multilingual, uncased) | 0.8993 | 0.6796 |
| **DistilBERT** | 0.8962 | 0.6850 |
| **viBERT** | 0.8881 | 0.6765 |

#### ViHOS Dataset
| Model | Accuracy | Macro F1 |
| :--- | :---: | :---: |
| **ViSoBERT** | 0.9016 | 0.8578 |
| **XLM-RoBERTa** | 0.8834 | 0.8133 |
| **PhoBERT v2** | 0.8492 | 0.7351 |
| **PhoBERT** | 0.8465 | 0.7281 |
| **BERT** (multilingual, cased) | 0.8601 | 0.7637 |
| **BERT** (multilingual, uncased) | 0.8520 | 0.7393 |
| **DistilBERT** | 0.8585 | 0.7615 |
| **viBERT** | 0.8463 | 0.7291 |

### Trung bình F1 Macro theo Model (across 3 datasets)
| Model | ViHSD F1 | ViCTSD F1 | ViHOS F1 | **Average F1** |
| :--- | :---: | :---: | :---: | :---: |
| **ViSoBERT** | 0.6871 | 0.7045 | 0.8578 | **0.7498** |
| **PhoBERT v2** | 0.6583 | 0.7139 | 0.7351 | **0.7024** |
| **PhoBERT** | 0.6360 | 0.7131 | 0.7281 | **0.6924** |
| **XLM-RoBERTa** | 0.6544 | 0.7153 | 0.8133 | **0.7277** |
| **BERT** (cased) | 0.6427 | 0.6710 | 0.7637 | **0.6925** |
| **BERT** (uncased) | 0.6161 | 0.6796 | 0.7393 | **0.6783** |
| **DistilBERT** | 0.6224 | 0.6850 | 0.7615 | **0.6896** |
| **viBERT** | 0.6149 | 0.6765 | 0.7291 | **0.6735** |
| **Overall Average** | **0.6412** | **0.6949** | **0.7660** | **0.7007** |

---

## 📊 Kết quả T5 Fine-tuning (Table 4 - Paper)

Dưới đây là kết quả chi tiết của các mô hình T5 được fine-tune trên 3 tập dữ liệu:

### Kết quả chi tiết theo Dataset

#### T5 Models Results
| Model | Dataset | Accuracy | F1 Weighted | F1 Macro |
| :--- | :--- | :---: | :---: | :---: |
| **ViT5 (Base)** | ViHSD | 0.8777 | 0.8787 | 0.6625 |
| **ViT5 (Base)** | ViCTSD | 0.9080 | 0.9178 | 0.7163 |
| **ViT5 (Base)** | ViHOS | 0.9075 | 0.9000 | 0.8612 |
| **mT5 (Base)** | ViHSD | 0.8746 | 0.8877 | 0.6246 |
| **mT5 (Base)** | ViCTSD | 0.8932 | 0.9024 | 0.7053 |
| **mT5 (Base)** | ViHOS | 0.9075 | 0.8957 | 0.8501 |
| **ViHateT5 (Ours)** | ViHSD | **0.8815** | **0.8849** | **0.6698** |
| **ViHateT5 (Ours)** | ViCTSD | **0.9105** | **0.9158** | **0.7189** |
| **ViHateT5 (Ours)** | ViHOS | **0.9081** | **0.9055** | **0.8616** |

### Trung bình F1 Macro theo Model T5 (across 3 datasets)
| Model | ViHSD F1 | ViCTSD F1 | ViHOS F1 | **Average F1** |
| :--- | :---: | :---: | :---: | :---: |
| **ViHateT5 (Ours)** | **0.6698** | **0.7189** | **0.8616** | **0.7501** ⭐ |
| **ViT5 (Base)** | 0.6625 | 0.7163 | 0.8612 | 0.7467 |
| **mT5 (Base)** | 0.6246 | 0.7053 | 0.8501 | 0.7267 |

---

## 📊 Kết quả ViHateT5 Pre-trained Impact (Table 5 - Paper)

Dưới đây là kết quả ảnh hưởng của pre-training với các tỉ lệ dữ liệu khác nhau trên hiệu suất của ViHateT5:

### Pre-trained trên 100K samples (Hate-Only)
| Dataset | Accuracy | F1 Weighted | F1 Macro |
| :--- | :---: | :---: | :---: |
| **ViHSD** | 0.8789 | 0.8784 | 0.6808 |
| **ViCTSD** | 0.9070 | 0.9283 | 0.6586 |
| **ViHOS** | 0.9039 | 0.8981 | 0.8541 |

### Pre-trained trên 200K samples (Balanced)
| Dataset | Accuracy | F1 Weighted | F1 Macro |
| :--- | :---: | :---: | :---: |
| **ViHSD** | 0.8815 | 0.8849 | 0.6698 |
| **ViCTSD** | 0.9105 | 0.9158 | 0.7189 |
| **ViHOS** | 0.9081 | 0.9055 | 0.8616 |

### Trung bình F1 Macro theo Pre-training Checkpoint
| Pre-training Setup | ViHSD F1 | ViCTSD F1 | ViHOS F1 | **Average F1** |
| :--- | :---: | :---: | :---: | :---: |
| **ViHateT5 (Ours) - Pre-trained (200K, Balanced)** | **0.6698** | **0.7189** | **0.8616** | **0.7501** ⭐ |
| **Pre-trained (100K, Hate-Only)** | 0.6808 | 0.6586 | 0.8541 | 0.7312 |

---

## **Table 6 — BERT-based models comparison (sorted by Macro F1)**

### Multilingual Pre-trained Models
| Model | Accuracy | F1 Weighted | F1 Macro |
| :--- | :---: | :---: | :---: |
| xlm-roberta-base | 0.9189 | 0.7722 | 0.8028 |
| xlm-roberta-large | 0.9204 | 0.7755 | 0.7968 |
| google-bert/bert-base-multilingual-uncased | 0.9102 | 0.7557 | 0.7784 |
| distilbert-base-multilingual-cased | 0.9115 | 0.7459 | 0.7754 |
| google-bert/bert-base-multilingual-cased | 0.9094 | 0.7548 | 0.7740 |

### Monolingual Pre-trained Models
| Model | Accuracy | F1 Weighted | F1 Macro |
| :--- | :---: | :---: | :---: |
| **uitnlp/visobert** | 0.9296 | 0.8051 | **0.8128** |
| vinai/phobert-base-v2 | 0.9216 | 0.7888 | 0.7810 |
| FPTAI/vibert-base-cased | 0.9117 | 0.7385 | 0.7771 |
| vinai/phobert-base | 0.9231 | 0.7562 | 0.7764 |
| vinai/phobert-large | 0.9245 | 0.7895 | 0.7832 |
---

## 📈 Kết quả & Output

Sau khi chạy training, kết quả sẽ được lưu vào thư mục `outputs/` hoặc `vihate_t5_pretrain/`:

-   **Model Checkpoints**: File trọng số (`.bin` / `.safetensors`) và cấu hình.
-   **`run_summary.csv`**: Tổng hợp kết quả tốt nhất (F1, Accuracy, Loss).
-   **`epoch_metrics.csv`**: Chi tiết các chỉ số qua từng epoch.
-   **`results/evaluation_results.csv`**: Kết quả đánh giá trên các tập test riêng biệt.

---

## 💡 Tối ưu hóa hiệu năng (Hardware Tips)

> **Lưu ý**: Tất cả các kết quả thực nghiệm trong dự án này đều được thực hiện trên GPU **NVIDIA H200** (được cung cấp bởi FPT thông qua voucher) và **P100**.

Tùy vào cấu hình phần cứng, bạn nên điều chỉnh các tham số sau để đạt tốc độ cao nhất:

-   **GPU H200 (141GB)**: Có thể dùng `batch_size=128` cho pre-training.
-   **GPU A100/A800 / P100**: Khuyến nghị `batch_size=128-256`.
-   **GPU Phổ thông (8GB-16GB)**: 
    -   Bật `gradient_checkpointing=True`.
    -   Sử dụng `gradient_accumulation_steps` (ví dụ: 8 hoặc 16) để bù đắp batch size nhỏ.
    -   Giảm `max_length` xuống 128 hoặc 256.

---

## 📚 Citation

Nếu bạn sử dụng code, dataset hoặc model trong nghiên cứu, vui lòng cite paper sau:

```bibtex
@inproceedings{nguyen2024vihate,
  title={ViHATE T5: Enhancing Hate Speech Detection in Vietnamese With a Unified Text-to-Text Transformer Model},
  author={Nguyen, Luan Thanh},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2024},
  pages={5948--5961},
  year={2024},
  url={https://aclanthology.org/2024.findings-acl.355.pdf}
}
```

**Paper**: [ViHATE T5: Enhancing Hate Speech Detection in Vietnamese With a Unified Text-to-Text Transformer Model](https://aclanthology.org/2024.findings-acl.355.pdf) (ACL 2024 Findings)

---

© 2024 Vietnamese Hate Speech Team. Dự án phục vụ mục đích nghiên cứu.






