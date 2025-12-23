# 🇻🇳 Vietnamese Hate Speech Detection Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Transformers](https://img.shields.io/badge/library-transformers-orange.svg)](https://github.com/huggingface/transformers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Một hệ thống toàn diện cho bài toán phát hiện ngôn ngữ thù ghét (Hate Speech) và bình luận độc hại (Toxic Speech) tiếng Việt, sử dụng các kiến trúc SOTA như **PhoBERT/ViSoBERT** và **T5/ViT5**.

---

## 📌 Tổng quan dự án

Dự án cung cấp 3 pipeline chính cho phép bạn đi từ dữ liệu thô đến mô hình hoàn chỉnh:
1.  **Pre-training**: Tiếp tục huấn luyện T5 với cơ chế *Span Corruption* trên dữ liệu tiếng Việt.
2.  **T5 Fine-tuning**: Huấn luyện Seq2Seq cho bài toán phân loại đa tập dữ liệu.
3.  **BERT Classification**: Huấn luyện các mô hình Encoder-only (PhoBERT, ViSoBERT) truyền thống.

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

## 🚀 Hướng dẫn sử dụng (Scripts)

### 1. Pre-training T5 (Span Corruption)
```bash
bash scripts/run_pretrain_t5.sh \
    --dataset_name "Minhbao5xx2/re_VOZ-HSD" \
    --split_name "hate_only" \
    --batch_size 512 \
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
| `--batch_size` | Batch size mỗi GPU | `32` | `512` |
| `--num_epochs` / `--epochs` | Số epoch huấn luyện | `4` | `10` |
| `--learning_rate` / `--lr` | Tốc độ học (Learning Rate) | `2e-4` | `5e-3` |
| `--max_length` | Độ dài sequence tối đa | `256` | - |
| `--gradient_accumulation_steps`| Tích lũy gradient | `1` | `1` |
| `--weight_decay` | Suy giảm trọng số | `0.01` | `0.001` |
| `--warmup_ratio` / `--warmup_steps`| Tỉ lệ/Số bước khởi động | `0.0` | `2000` |
| `--seed` | Random seed | `42` | - |

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
| **BERT** (multilingual, cased) | 0.8800 | 0.6886 |
| **BERT** (multilingual, uncased) | 0.8820 | 0.6569 |
| **DistilBERT** | 0.8640 | 0.6634 |
| **XLM-RoBERTa** | 0.8990 | 0.7231 |
| **PhoBERT** | 0.8750 | 0.7210 |
| **PhoBERT v2** | 0.8890 | 0.7304 |
| **viBERT** | 0.8920 | 0.6946 |
| **ViSoBERT** | 0.9050 | 0.7483 |

#### ViHOS Dataset
| Model | Accuracy | Macro F1 |
| :--- | :---: | :---: |
| **ViSoBERT** | 0.9231 | 0.9230 |
| **viBERT** | 0.8590 | 0.8589 |
| **BERT** (multilingual, uncased) | 0.8707 | 0.8706 |
| **BERT** (multilingual, cased) | 0.8834 | 0.8832 |
| **XLM-RoBERTa** | 0.8879 | 0.8878 |
| **PhoBERT v2** | 0.9033 | 0.9031 |
| **PhoBERT** | 0.8906 | 0.8903 |
| **DistilBERT** | 0.8707 | 0.8706 |

### Trung bình F1 Macro theo Model (across 3 datasets)
| Model | ViHSD F1 | ViCTSD F1 | ViHOS F1 | **Average F1** |
| :--- | :---: | :---: | :---: | :---: |
| **ViSoBERT** | 0.6871 | 0.7483 | 0.9230 | **0.7861** |
| **PhoBERT v2** | 0.6583 | 0.7304 | 0.9031 | **0.7639** |
| **PhoBERT** | 0.6360 | 0.7210 | 0.8903 | **0.7491** |
| **XLM-RoBERTa** | 0.6544 | 0.7231 | 0.8878 | **0.7551** |
| **BERT** (cased) | 0.6427 | 0.6886 | 0.8832 | **0.7382** |
| **viBERT** | 0.6149 | 0.6946 | 0.8589 | **0.7228** |
| **BERT** (uncased) | 0.6161 | 0.6569 | 0.8706 | **0.7145** |
| **DistilBERT** | 0.6224 | 0.6634 | 0.8706 | **0.7188** |
| **Overall Average** | **0.6412** | **0.7033** | **0.8911** | **0.7452** |

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
| **ViHateT5** | ViHSD | 0.8876 | 0.8914 | 0.6867 |
| **ViHateT5** | ViCTSD | 0.9178 | 0.9080 | 0.7163 |
| **ViHateT5** | ViHOS | 0.9020 | 0.9100 | 0.8637 |

### Trung bình F1 Macro theo Model T5 (across 3 datasets)
| Model | ViHSD F1 | ViCTSD F1 | ViHOS F1 | **Average F1** |
| :--- | :---: | :---: | :---: | :---: |
| **ViHateT5** | 0.6867 | 0.7163 | 0.8637 | **0.7556** |
| **ViT5 (Base)** | 0.6625 | 0.7163 | 0.8612 | **0.7467** |
| **mT5 (Base)** | 0.6246 | 0.7053 | 0.8501 | **0.7267** |
| **Overall Average** | **0.6579** | **0.7126** | **0.8583** | **0.7430** |

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
| **ViHSD** | 0.8843 | 0.8919 | 0.6621 |
| **ViCTSD** | 0.8630 | 0.8550 | 0.6921 |
| **ViHOS** | 0.9103 | 0.9027 | 0.8598 |

### Trung bình F1 Macro theo Pre-training Checkpoint
| Pre-training Setup | ViHSD F1 | ViCTSD F1 | ViHOS F1 | **Average F1** |
| :--- | :---: | :---: | :---: | :---: |
| **Pre-trained (100K, Hate-Only)** | 0.6808 | 0.6586 | 0.8541 | **0.7312** |
| **Pre-trained (200K, Balanced)** | 0.6621 | 0.6921 | 0.8598 | **0.7380** |
| **Fine-tuned from scratch (ViHateT5)** | 0.6867 | 0.7163 | 0.8637 | **0.7556** |
| **Overall Average** | **0.6765** | **0.6890** | **0.8592** | **0.7416** |

---

## 📈 Kết quả & Output

Sau khi chạy training, kết quả sẽ được lưu vào thư mục `outputs/` hoặc `vihate_t5_pretrain/`:

-   **Model Checkpoints**: File trọng số (`.bin` / `.safetensors`) và cấu hình.
-   **`run_summary.csv`**: Tổng hợp kết quả tốt nhất (F1, Accuracy, Loss).
-   **`epoch_metrics.csv`**: Chi tiết các chỉ số qua từng epoch.
-   **`results/evaluation_results.csv`**: Kết quả đánh giá trên các tập test riêng biệt.

---

## 💡 Tối ưu hóa hiệu năng (Hardware Tips)

> **Lưu ý**: Tất cả các kết quả thực nghiệm trong dự án này đều được thực hiện trên GPU **NVIDIA H200** và **P100**.

Tùy vào cấu hình phần cứng, bạn nên điều chỉnh các tham số sau để đạt tốc độ cao nhất:

-   **GPU H200 (141GB)**: Có thể dùng `batch_size=512` cho pre-training.
-   **GPU A100/A800 / P100**: Khuyến nghị `batch_size=128-256`.
-   **GPU Phổ thông (8GB-16GB)**: 
    -   Bật `gradient_checkpointing=True`.
    -   Sử dụng `gradient_accumulation_steps` (ví dụ: 8 hoặc 16) để bù đắp batch size nhỏ.
    -   Giảm `max_length` xuống 128 hoặc 256.

---

## 📂 Cấu trúc thư mục

```text
.
├── src/                    # Mã nguồn chính (Python)
├── scripts/                # Bash scripts chạy nhanh
├── outputs/                # Lưu trữ model checkpoints
├── results/                # Lưu trữ kết quả đánh giá (CSV)
└── requirements.txt        # Danh sách thư viện cần thiết
```

---

## ⚠️ Giải quyết sự cố thường gặp

1.  **Lỗi OOM**: Giảm `batch_size`, tăng `gradient_accumulation_steps`, hoặc giảm `max_length`.
2.  **ModuleNotFoundError**: `pip install -r requirements.txt` và chạy từ thư mục gốc.
3.  **Tốc độ chậm**: Kiểm tra `dataloader_num_workers` và sử dụng GPU phù hợp.

---
© 2024 Vietnamese Hate Speech Team. Dự án phục vụ mục đích nghiên cứu.
