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

## 🚀 Hướng dẫn sử dụng (Scripts)

Chúng tôi cung cấp các script bash trong thư mục `scripts/` để chạy nhanh với các tham số tối ưu.

### 1. Pre-training T5 (Span Corruption)
Sử dụng khi bạn muốn mô hình T5 hiểu sâu hơn về ngữ cảnh dữ liệu đặc thù của mình.

```bash
bash scripts/run_pretrain_t5.sh \
    --dataset_name "Minhbao5xx2/re_VOZ-HSD" \
    --split_name "hate_only" \
    --max_samples 50000
```
*Lưu ý: Script được tối ưu mặc định cho GPU H200. Nếu dùng GPU nhỏ hơn, hãy điều chỉnh `batch_size` trong code.*

### 2. Fine-tuning T5 (Phân loại Seq2Seq)
Huấn luyện mô hình sinh ra nhãn văn bản (ví dụ: "HATE", "CLEAN").

```bash
bash scripts/run_train_t5.sh \
    --pre_trained_ckpt "VietAI/vit5-base" \
    --batch_size 32 \
    --num_epochs 4 \
    --gpu "0"
```

### 3. Huấn luyện BERT/PhoBERT (Classification)
Cách tiếp cận truyền thống sử dụng Classification Head.

```bash
bash scripts/run_train_bert.sh \
    --dataset "ViHSD" \
    --model_name "vinai/phobert-base" \
    --epochs 10 \
    --patience 3
```

---

## ⚙️ Chi tiết tham số (CLI Arguments)

### Các tham số chung cho các Script:
| Tham số | Mô tả | Mặc định |
| :--- | :--- | :--- |
| `--dataset` | Tên dataset hoặc đường dẫn HF | `ViHSD` |
| `--model_name` | Model checkpoint từ HuggingFace | `vinai/phobert-base` |
| `--batch_size` | Kích thước batch huấn luyện | `16` |
| `--epochs` | Số lượng epoch huấn luyện | `10` |
| `--learning_rate`| Tốc độ học | `2e-5` |
| `--output_dir` | Thư mục lưu kết quả | Tự động sinh |

---

## 📈 Kết quả & Output

Sau khi chạy training, kết quả sẽ được lưu vào thư mục `outputs/` hoặc `vihate_t5_pretrain/`:

-   **Model Checkpoints**: File trọng số (`.bin` / `.safetensors`) và cấu hình.
-   **`run_summary.csv`**: Tổng hợp kết quả tốt nhất (F1, Accuracy, Loss).
-   **`epoch_metrics.csv`**: Chi tiết các chỉ số qua từng epoch.
-   **`results/evaluation_results.csv`**: (Dành riêng cho T5) Kết quả đánh giá trên các tập test riêng biệt.

---

## 💡 Tối ưu hóa hiệu năng (Hardware Tips)

Tùy vào cấu hình phần cứng, bạn nên điều chỉnh các tham số sau để đạt tốc độ cao nhất:

-   **GPU H200 (141GB)**: Có thể dùng `batch_size=512` cho pre-training.
-   **GPU A100/A800**: Khuyến nghị `batch_size=128-256`.
-   **GPU Phổ thông (8GB-16GB)**: 
    -   Bật `gradient_checkpointing=True`.
    -   Sử dụng `gradient_accumulation_steps` để bù đắp batch size nhỏ.
    -   Giảm `max_length` xuống 128 nếu bị OOM.

---

## 📂 Cấu trúc thư mục

```text
.
├── src/                    # Mã nguồn chính (Python)
│   ├── pre_train_t5.py    # Script pre-training
│   ├── train_t5.py         # Script fine-tuning T5
│   ├── train_bert.py       # Script huấn luyện BERT
│   └── data_loader.py      # Xử lý nạp dữ liệu
├── scripts/                # Bash scripts chạy nhanh
├── outputs/                # Lưu trữ model checkpoints
├── results/                # Lưu trữ kết quả đánh giá (CSV)
└── requirements.txt        # Danh sách thư viện cần thiết
```

---

## ⚠️ Giải quyết sự cố thường gặp

1.  **Lỗi OOM (Out of Memory)**: Giảm `batch_size` hoặc `max_length`.
2.  **Không tìm thấy module**: Đảm bảo bạn đã `pip install -r requirements.txt` và chạy script từ thư mục gốc.
3.  **Lỗi nạp Dataset**: Kiểm tra kết nối mạng và đảm bảo tên dataset trên HuggingFace là chính xác.

---
© 2024 Vietnamese Hate Speech Team. Dự án phục vụ mục đích nghiên cứu.
