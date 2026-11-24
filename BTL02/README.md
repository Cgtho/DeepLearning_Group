# FER2013 Emotion Recognition

Dự án phân loại 7 cảm xúc từ ảnh FER2013 grayscale 48 × 48 bằng PyTorch.

## Cấu trúc hiện tại

- `data/`: dữ liệu cục bộ `train.csv`, `test.csv`; `fer2013.csv` chỉ cần để đánh giá test có nhãn.
- `models/baseline_cnn.py`: CNN baseline.
- `models/exp02_more_convolution/`: DeeperCNN, Improved V2 và Improved V3.
- `configs/`: cấu hình baseline và ba biến thể EXP02.
- `train.py`, `fer_data.py`: pipeline huấn luyện và dữ liệu.
- `evaluate.py`, `visualize.py`: đánh giá checkpoint, lưu metric và hình.
- `notebooks/01_data_exploration.ipynb`: khám phá dữ liệu.
- `notebooks/02_baseline_cnn.ipynb`: kết quả baseline.
- `notebooks/03_exp02_more_convolution.ipynb`: so sánh baseline và EXP02 trên validation.
- `results/<experiment>/<run>/`: checkpoint, cấu hình, lịch sử và kết quả đánh giá.
- `runs/<experiment>/<run>/`: log TensorBoard.

## Baseline

Chạy từ thư mục gốc dự án:

```powershell
python train.py --config configs/baseline.yaml
```

Thêm `--smoke` để chạy một epoch kiểm tra pipeline. Mỗi run được lưu trong thư mục timestamp, gồm `best.pth`, `config.yaml`, `history.csv` và `training.json`.

## EXP002: tăng số lớp convolution

Notebook EXP002 so sánh baseline với ba biến thể:

- **DeeperCNN:** 4 lớp Conv, thêm một Conv 128 kênh so với baseline.
- **Improved V2:** 6 lớp Conv, hai Conv trong mỗi khối 32/64/128 kênh.
- **Improved V3:** 8 lớp Conv, thêm khối 256 kênh và một lần pooling nữa.

Các cấu hình giữ split 80/20, seed 42, batch size 64, Adam, learning rate 0,001 và 30 epoch. Chạy từng cấu hình bằng:

```powershell
python train.py --config configs/exp02_more_conv.yaml
python train.py --config configs/exp02_improved_v2.yaml
python train.py --config configs/exp02_improved_v3.yaml
```

Các kết quả hiện có được tạo trong cùng split và thông số huấn luyện, nhưng baseline ngày 20/11 chạy CPU/PyTorch 2.14 còn các biến thể EXP02 chạy RTX 3050/PyTorch 2.11. Vì vậy không nên so sánh trực tiếp thời gian chạy; nếu cần đối chứng chặt hơn, hãy chạy lại baseline trong cùng môi trường GPU.

Notebook `03_exp02_more_convolution.ipynb` dùng các đường dẫn trong `RUN_DIRS` để tái sử dụng run đã hoàn tất; đặt một giá trị thành `None` nếu muốn huấn luyện lại biến thể đó. Đường dẫn run được in tương đối từ thư mục dự án.

Notebook tính validation accuracy, macro F1 và confusion matrix cho baseline cùng từng biến thể. Kết quả đánh giá và hình được lưu vào thư mục run tương ứng. Dùng validation loss, accuracy và macro F1 để chọn mô hình; chỉ đánh giá test sau khi đã chốt mô hình cuối.

## Validation và test

Đánh giá một checkpoint trên validation từ terminal:

```powershell
python evaluate.py results/EXP02/improved_cnn_v2/<run-timestamp> --split validation
```

`test.csv` hiện chỉ có ảnh, không có nhãn. Để tính test accuracy/F1 và confusion matrix, cần đặt `fer2013.csv` vào `data/`; code kiểm tra ảnh và thứ tự test có khớp trước khi lấy nhãn. Sau khi chọn checkpoint cuối, chạy:

```powershell
python evaluate.py results/EXP02/improved_cnn_v2/<run-timestamp>
```

## TensorBoard

```powershell
tensorboard --logdir runs --port 6006
```

Mở `http://localhost:6006` trong trình duyệt. Nhấn `Ctrl+C` để dừng TensorBoard.
