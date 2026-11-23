# FER2013 Emotion Recognition

Project CNN phân loại 7 cảm xúc từ ảnh grayscale 48×48 bằng PyTorch.

## Cấu trúc

- `data/`: `train.csv` và `test.csv` (dữ liệu cục bộ, không commit).
- `models/`: kiến trúc baseline hiện tại.
- `configs/`: cấu hình baseline; tạo config mới khi bắt đầu thí nghiệm.
- `train.py`, `evaluate.py`: huấn luyện và đánh giá checkpoint.
- `notebooks/`: khám phá dữ liệu và trình bày kết quả.
- `results/<experiment>/<run>/`: checkpoint, config đã dùng và lịch sử train.
- `runs/<experiment>/<run>/`: log TensorBoard.

## Chạy baseline

Từ thư mục gốc dự án:

```powershell
python train.py --config configs/baseline.yaml
```

Kiểm tra nhanh một epoch trước khi chạy đầy đủ:

```powershell
python train.py --config configs/baseline.yaml --smoke
```

Lượt chạy lưu cấu hình, `history.csv`, `training.json` và `best.pth` vào thư mục timestamp riêng. Khi bắt đầu thí nghiệm mới, hãy tạo kiến trúc và config riêng. Chọn mô hình bằng validation; chỉ đánh giá test cho checkpoint cuối cùng:

```powershell
python evaluate.py results/EXP001_Baseline/<run-timestamp>
```

Lệnh đánh giá trên test cần `emotion` trong `test.csv` hoặc `fer2013.csv` để đối chiếu nhãn. Nếu test không có nhãn và thiếu file gốc, đánh giá validation bằng:

```powershell
python evaluate.py results/EXP001_Baseline/<run-timestamp> --split validation
```

Lệnh đánh giá ghi `evaluation.json` và các hình phân tích vào cùng thư mục run. Thay `<run-timestamp>` bằng tên thư mục thực tế. Kết quả validation và test được ghi nhãn riêng.

## Quy tắc so sánh

Khi tạo thí nghiệm mới, giữ nguyên split, seed, augmentation, optimizer, batch size và số epoch; mỗi lượt chỉ thay đổi một yếu tố chính. Khi tổng hợp báo cáo, ghi lại validation loss tốt nhất, validation accuracy tại epoch đó, macro F1 trên test cho mô hình cuối, số tham số và thời gian huấn luyện.

| Experiment | Model | Val loss | Val accuracy | Test macro F1 | Parameters | Thời gian |
|---|---|---:|---:|---:|---:|---:|
| EXP001_Baseline | `baseline` | | | | | |

Điền số liệu từ `training.json`, `history.csv` và `evaluation.json` sau khi hoàn tất baseline.
