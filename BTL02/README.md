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

Lệnh đánh giá ghi `evaluation.json` và các hình phân tích vào cùng thư mục run. Thay `<run-timestamp>` bằng tên thư mục thực tế.

## Quy tắc so sánh

Khi tạo thí nghiệm mới, giữ nguyên split, seed, augmentation, optimizer, batch size và số epoch; mỗi lượt chỉ thay đổi một yếu tố chính. Khi tổng hợp báo cáo, ghi lại validation loss tốt nhất, validation accuracy tại epoch đó, macro F1 trên test cho mô hình cuối, số tham số và thời gian huấn luyện.

| Experiment | Model | Val loss | Val accuracy | Test macro F1 | Parameters | Thời gian |
|---|---|---:|---:|---:|---:|---:|
| EXP001_Baseline | `baseline` | | | | | |
| EXP002_CNN_MLP | `cnn_mlp` | 1.198144 | 55.29% | 0.5200 | 5,355,015 | Resume: 2190.980 giây |

Điền số liệu từ `training.json`, `history.csv` và `evaluation.json` sau khi hoàn tất baseline.


## CNN + MLP và Training Utilities

Mẫu `2. CNN + MLP/1_MNIST_CNN.ipynb` được chuyển sang PyTorch cho FER2013 tại
`notebooks/03_cnn_mlp.ipynb`. Kiến trúc: Conv 64/128/256, pooling, dropout,
MLP 512/512/7. Input 48×48; output logits, không Softmax trước CrossEntropyLoss.

```powershell
python -m pip install -r requirements.txt
python train.py --config configs/cnn_mlp.yaml --smoke
python train.py --config configs/cnn_mlp.yaml
python evaluate.py results/EXP002_CNN_MLP/<run-timestamp>
python -m tensorboard.main --logdir runs
```

`--smoke` chạy một epoch trên toàn bộ train/validation; không đánh giá test.
Dữ liệu đặt tại `data/train.csv` và `data/test.csv`, thêm `fer2013.csv` nếu test thiếu nhãn.
Checkpoint và kết quả cũ giữ nguyên. Các hàm dùng chung nằm trong `training_utils/`;
import cũ từ `train` vẫn dùng được. EarlyStopping, ReduceLROnPlateau và augmentation
được điều khiển bằng YAML, mặc định tắt khi so sánh với baseline. Muốn thử, sao chép
config với tên experiment mới và thay một yếu tố mỗi lần.
