# FER2013 Emotion Recognition

Dự án phân loại 7 cảm xúc từ ảnh FER2013 grayscale 48 × 48 bằng PyTorch.

## Cấu trúc hiện tại

- `data/`: dữ liệu cục bộ `train.csv`, `test.csv`; `fer2013.csv` chỉ cần để đánh giá test có nhãn.
- `models/baseline_cnn.py`: CNN baseline.
- `models/exp02_more_convolution/`: DeeperCNN, Improved V2 và Improved V3.
- `models/exp03_batch_normalization/`: các kiến trúc BatchNorm của EXP03.
- `models/exp05_dropout/`: baseline có Dropout ở lớp fully connected.
- `configs/`: cấu hình baseline và các lượt EXP02–EXP06.
- `train.py`, `fer_data.py`: pipeline huấn luyện và dữ liệu.
- `evaluate.py`, `visualize.py`: đánh giá checkpoint, lưu metric và hình.
- `notebooks/01_data_exploration.ipynb`: khám phá dữ liệu.
- `notebooks/02_baseline_cnn.ipynb`: kết quả baseline.
- `notebooks/03_exp02_more_convolution.ipynb`: so sánh baseline và EXP02 trên validation.
- `notebooks/04_exp03_batch_normalization.ipynb`: so sánh BatchNorm trên validation.
- `notebooks/05_exp04_data_augmentation.ipynb`: so sánh baseline có/không augmentation trên validation.
- `notebooks/06_exp05_dropout.ipynb`: so sánh baseline có/không Dropout trên validation.
- `notebooks/07_exp06_learning_rate.ipynb`: so sánh ba learning rate của Adam trên validation.
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

## EXP003: Batch Normalization

EXP03 khảo sát tác động của Batch Normalization bằng cách thêm `BatchNorm2d` sau các lớp convolution và `BatchNorm1d` sau lớp fully connected ẩn. Thực nghiệm gồm bốn kiến trúc có BatchNorm tương ứng với baseline, DeeperCNN, Improved V2 và Improved V3. Các cấu hình dùng split 80/20, seed 42, batch size 64, Adam với learning rate 0,001 và 30 epoch.

Chạy từng cấu hình từ thư mục gốc dự án:

```powershell
python train.py --config configs/exp03_baseline_bn.yaml
python train.py --config configs/exp03_deeper_cnn_bn.yaml
python train.py --config configs/exp03_improved_v2_bn.yaml
python train.py --config configs/exp03_improved_v3_bn.yaml
```

Notebook `04_exp03_batch_normalization.ipynb` dùng các run trong `RUN_DIRS` để đánh giá trên validation; có thể điền đường dẫn run đã có để đọc lại kết quả hoặc để `None` để huấn luyện lại theo cấu hình. Notebook so sánh validation accuracy, macro F1 và confusion matrix giữa các kiến trúc có BatchNorm. Dùng các chỉ số validation để so sánh; chưa cần đánh giá test trong giai đoạn thực nghiệm.

## EXP004: data augmentation

EXP04 đo tác động của augmentation bằng cách so sánh cùng một baseline CNN có và không có augmentation. Split 80/20, seed 42, Adam `lr=0.001`, batch size 64 và 30 epoch được giữ nguyên. Chỉ tập train được biến đổi ngẫu nhiên (xoay tối đa 15°, dịch tối đa 5% chiều rộng/cao và lật ngang xác suất 0,5); validation không bị biến đổi.

Hai cấu hình EXP04 chọn `device: cuda` để buộc chạy trên GPU; cần dùng môi trường PyTorch có hỗ trợ CUDA và driver tương thích.

```powershell
python train.py --config configs/exp04_no_augmentation.yaml
python train.py --config configs/exp04_data_augmentation.yaml
```

Notebook `05_exp04_data_augmentation.ipynb` có thể huấn luyện hai cấu hình khi `RUN_DIRS` để `None`, hoặc đọc lại các run có sẵn bằng cách điền đường dẫn tương đối. Notebook đánh giá trên validation, so sánh accuracy/macro F1 và hiển thị learning curves, confusion matrix cùng filter của lớp convolution đầu tiên. Chưa dùng tập test để chọn mô hình.

## EXP005: Dropout

EXP05 giữ nguyên baseline CNN, không augmentation và không BatchNorm; chỉ thêm `Dropout(p=0.5)` sau ReLU của lớp fully connected ẩn. Các thiết lập split, seed, batch size, optimizer, learning rate và số epoch giữ như baseline.

```powershell
python train.py --config configs/exp05_dropout.yaml --smoke
python train.py --config configs/exp05_dropout.yaml
```

Run đối chứng không Dropout dùng baseline không augmentation tại `results/EXP001_Baseline/20261120_122107_175923`. Notebook `06_exp05_dropout.ipynb` đánh giá cả hai trên validation rồi so sánh accuracy, macro F1, F1 từng lớp, confusion matrix và learning curves. Môi trường Python mặc định hiện không có CUDA; để so sánh thời gian công bằng, chạy cả hai cấu hình trên cùng một kernel/phần cứng.

## EXP006: ảnh hưởng của learning rate

EXP06 so sánh baseline CNN được huấn luyện bằng Adam với learning rate `0.0003`, `0.001` và `0.003`. Split 80/20, seed 42, batch size 64, 30 epoch, kiến trúc và các thiết lập khác được giữ nguyên; không dùng augmentation, BatchNorm hoặc Dropout. Cả ba config chọn `device: cuda` để dùng cùng GPU.

```powershell
python train.py --config configs/exp06_lr_0003.yaml
python train.py --config configs/exp06_lr_001.yaml
python train.py --config configs/exp06_lr_003.yaml
```

Notebook `07_exp06_learning_rate.ipynb` kiểm tra các biến kiểm soát, có thể chạy các lượt còn thiếu hoặc đọc lại các run đã hoàn thành, rồi so sánh validation loss, accuracy, macro F1, F1 từng lớp, confusion matrix và learning curves. Chỉ dùng validation để so sánh; không dùng test để chọn learning rate.

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
