# Báo cáo chuyển đổi CNN nhận diện giới tính từ Keras sang PyTorch

## 1. Mục tiêu

Báo cáo mô tả việc chuyển notebook nhận diện giới tính từ Keras sang PyTorch, đồng thời ghi lại cách xử lý dữ liệu, kiến trúc mô hình, quy trình huấn luyện và kết quả đã chạy. Bản PyTorch giữ nguyên bài toán phân loại ảnh khuôn mặt grayscale thành hai lớp `female` và `male`.

Notebook được chuyển đổi là [2_Human_Gender_CNN.ipynb](./2_Human_Gender_CNN.ipynb). Bố cục báo cáo tham khảo tài liệu chuyển đổi giới tính hiện có trong thư mục CNN + MLP và báo cáo chuyển CNN nhận diện biểu cảm trong thư mục này; nội dung dưới đây dựa trên mã và kết quả riêng của notebook hiện tại.

## 2. Dữ liệu và tiền xử lý

Dữ liệu gồm 11.938 ảnh huấn luyện và 5.968 ảnh validation. Mỗi ảnh có kích thước 64 × 64, một kênh grayscale; nhãn ban đầu được lưu dưới dạng one-hot hai lớp.

| Tập | Tệp ảnh | Tệp nhãn | Kích thước ban đầu |
| --- | --- | --- | --- |
| Train | `64_64_11938_4098_train_x_onehot.npy` | `64_64_11938_4098_train_y_onehot.npy` | `(11938, 64, 64, 1)` / `(11938, 2)` |
| Validation | `64_64_5968_4098_val_x_onehot.npy` | `64_64_5968_4098_val_y_onehot.npy` | `(5968, 64, 64, 1)` / `(5968, 2)` |

Các thay đổi tiền xử lý chính:

- Giữ phép xoay `np.rot90(..., k=3, axes=(1, 2))` như notebook Keras.
- Chuyển ảnh từ NHWC `[N, H, W, C]` sang NCHW `[N, C, H, W]`, định dạng đầu vào của `nn.Conv2d`.
- Chuyển ảnh sang `float32`; giữ nguyên dải pixel gốc 0–255, không chuẩn hóa thêm.
- Chuyển nhãn one-hot sang chỉ số lớp bằng `argmax(axis=1)`, sau đó dùng kiểu `torch.long` cho `nn.CrossEntropyLoss`. Lớp 0 là `female`, lớp 1 là `male`.

## 3. Chuyển kiến trúc mô hình

Mô hình PyTorch giữ số khối, số lớp và kích thước các tầng như notebook Keras:

| Thành phần | Keras | PyTorch |
| --- | --- | --- |
| Đầu vào | `Input(shape=(64, 64, 1))`, NHWC | Tensor `[N, 1, 64, 64]`, NCHW |
| Tích chập | `Conv2D`, kernel 3 × 3, `padding="same"`, ReLU | `nn.Conv2d(..., kernel_size=3, padding=1)` và `nn.ReLU()` |
| Pooling | `MaxPooling2D(2, 2)` | `nn.MaxPool2d(2)` |
| Dropout | 0,2 sau từng khối tích chập và tầng ẩn | `nn.Dropout(0.2)` tại các vị trí tương ứng |
| MLP | Hai tầng Dense 1024 nút, ReLU | Hai `nn.Linear(1024, 1024)` và ReLU |
| Phân loại | Dense 2 nút, softmax | `nn.Linear(1024, 2)` trả về logits |
| Hàm mất mát | Categorical cross-entropy | `nn.CrossEntropyLoss()` |

Mạng có năm khối tích chập với số kênh lần lượt là 16, 32, 64, 128 và 256. Số lớp tích chập trong từng khối là 2, 3, 3, 3 và 3. Mỗi khối kết thúc bằng max pooling và dropout 0,2. Sau năm lần pooling, kích thước không gian giảm từ 64 × 64 xuống 2 × 2; phép flatten tạo ra `256 × 2 × 2 = 1024` đặc trưng.

Phần fully connected gồm hai tầng 1024 nút, mỗi tầng có ReLU và dropout 0,2, sau đó là tầng đầu ra hai logits. Vì `CrossEntropyLoss` nhận logits và tự thực hiện log-softmax, mô hình không đặt softmax ở tầng cuối. Các tầng `Conv2d` và `Linear` được khởi tạo Xavier uniform với bias bằng 0, tương ứng cấu hình mặc định Glorot của mô hình Keras. Tổng số tham số có thể học là 4.063.570.

## 4. Huấn luyện và công cụ hỗ trợ

Notebook dùng `TensorDataset` và `DataLoader`, với batch size 128 và xáo trộn dữ liệu huấn luyện. Adam dùng learning rate ban đầu 0,001 và `eps=1e-7`.

Vòng huấn luyện được dùng lại từ [shared_training.py](./shared_training.py), thay cho `model.compile()` và `model.fit()`:

1. Ở chế độ train, mỗi batch xóa gradient cũ, tính logits và loss, gọi `backward()` rồi cập nhật trọng số bằng optimizer.
2. Ở chế độ validation, mô hình chuyển sang `eval()` và không tính gradient.
3. Loss và accuracy được tính trên toàn bộ tập theo số lượng mẫu.

Các tiện ích Keras được chuyển tương ứng như sau:

- **ModelCheckpoint:** lưu `state_dict` tại `checkpoints/best_gender_cnn.pth` khi validation loss đạt giá trị tốt nhất.
- **EarlyStopping:** dừng sau 5 epoch liên tiếp không cải thiện validation loss.
- **ReduceLROnPlateau:** giảm learning rate theo hệ số 0,1 sau 5 epoch không cải thiện, với learning rate tối thiểu 0,0001.
- **TensorBoard:** ghi loss, accuracy và learning rate vào thư mục `logs/`.

Hàm `fit` dùng chung tự nạp lại checkpoint có validation loss tốt nhất trước khi trả về. Vì vậy, trọng số sau huấn luyện không nhất thiết là trọng số của epoch cuối cùng.

## 5. Lỗi môi trường và cách xử lý

Trong môi trường Anaconda trên Windows, kernel gặp lỗi OpenMP do có các bản `libiomp5md.dll` trong Anaconda MKL và thư mục thư viện của PyTorch. Notebook đặt `MKL_THREADING_LAYER='SEQUENTIAL'` trước khi import NumPy và PyTorch để tránh nạp thêm OpenMP runtime từ MKL. Cách này không bật `KMP_DUPLICATE_LIB_OK`, tùy chọn không an toàn được đề cập trong một báo cáo cũ.

Ngoài ra, môi trường ban đầu có TensorBoard 2.6.0 không tương thích với protobuf 6.33.5. TensorBoard được nâng cấp lên 2.21.0 trong kernel notebook; dependency `tensorboard` đã có trong [requirements.txt](../requirements.txt). Sau khi đổi thiết lập hoặc package, cần restart kernel và chạy lại notebook từ đầu.

## 6. Kiểm thử và kết quả

### Kiểm thử một epoch

Trước khi chạy đầy đủ, quy trình được chạy thử một epoch trên toàn bộ train và validation. Kết quả:

| Chỉ số | Giá trị |
| --- | ---: |
| Thời gian trên CPU | 1,66 phút |
| Train loss | 0,5537 |
| Train accuracy | 78,45% |
| Validation loss | 0,5116 |
| Validation accuracy | 78,85% |

Checkpoint của lần chạy thử được lưu trong thư mục tạm và không ghi đè checkpoint huấn luyện chính.

### Lần huấn luyện đầy đủ đã lưu trong notebook

Notebook đã chạy 42 epoch trên CPU rồi dừng sớm. Validation loss tốt nhất đạt tại epoch 37; tổng thời gian huấn luyện khoảng 72,13 phút.

| Chỉ số | Giá trị |
| --- | ---: |
| Epoch tốt nhất | 37 |
| Epoch kết thúc | 42 |
| Validation loss tốt nhất | 0,3238 |
| Validation accuracy ở epoch tốt nhất | 88,05% |
| Validation loss khi nạp checkpoint và đánh giá lại | 0,3238 |
| Validation accuracy khi đánh giá lại | 88,05% |

Tập validation được dùng để theo dõi early stopping, chọn checkpoint và đánh giá lại; do đó accuracy 88,05% không phải kết quả trên một tập test độc lập. Kết quả có thể thay đổi theo môi trường phần cứng và lần chạy.

## 7. Kết luận

Notebook PyTorch duy trì cấu trúc CNN + MLP, phép tiền xử lý và các thiết lập huấn luyện chính của bản Keras. Các khác biệt quan trọng nằm ở thứ tự tensor NCHW, nhãn dạng chỉ số lớp, logits kết hợp `CrossEntropyLoss`, và vòng lặp huấn luyện tường minh thông qua các helper PyTorch. Kiểm thử một epoch xác nhận vòng huấn luyện hoạt động; lần chạy đầy đủ đã lưu đạt 88,05% accuracy trên validation.

## Tài liệu tham khảo trong dự án

- [Notebook PyTorch nhận diện giới tính](./2_Human_Gender_CNN.ipynb)
- [Helper huấn luyện dùng chung](./shared_training.py)
- [Kiểm thử helper huấn luyện](./test_shared_training.py)
- [Báo cáo chuyển đổi giới tính có sẵn](../2.%20CNN%20%2B%20MLP/BAO_CAO_CHUYEN_NHAN_DIEN_GIOI_TINH_KERAS_SANG_PYTORCH.txt)
- [Báo cáo chuyển CNN nhận diện biểu cảm](./Bao_cao_chuyen_Keras_sang_PyTorch.md)
