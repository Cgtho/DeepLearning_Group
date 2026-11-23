# Báo cáo chuyển notebook nhận diện biểu cảm khuôn mặt từ Keras sang PyTorch

## 1. Chuẩn bị và nạp dữ liệu

Hai tệp đầu vào được tạo từ `fer2013.csv`. Cột `Usage` xác định tập của mỗi ảnh. Giá trị trong cột `pixels` được tách thành 2.304 cột, sau đó thêm 7 cột one-hot từ chỉ số trong cột `emotion`. Hai tệp được lưu dưới dạng CSV không có dòng tiêu đề, cạnh notebook.

| Tập dữ liệu | Tệp đầu vào | Số ảnh |
| --- | --- | ---: |
| Training | `fer2013_training_onehot.csv` | 28.709 |
| PublicTest | `fer2013_publictest_onehot.csv` | 3.589 |

Hàm `load_fer2013` đọc pixel dưới dạng `float32` và dùng `reshape(-1, 1, 48, 48)` để tạo tensor theo thứ tự `[N, C, H, W]` của PyTorch. Bước này thay cho `expand_dims(axis=4)` trong bản Keras, vốn gây lỗi vì trục đó nằm ngoài phạm vi của mảng `[N, 48, 48]`. Giá trị pixel được giữ trong khoảng 0–255. Với nhãn, `argmax(axis=1)` chuyển mỗi hàng one-hot thành một chỉ số lớp kiểu `torch.long`. Các lớp theo thứ tự từ 0 đến 6 là `angry`, `disgust`, `fear`, `happy`, `sad`, `surprise` và `neutral`.

## 2. Chuyển mô hình CNN + MLP

Bản PyTorch khai báo các lớp theo thứ tự trong `EmotionCNN(nn.Sequential)`. Sáu lớp tích chập có số kênh đầu ra lần lượt là 64, 64, 128, 128, 218 và 218. Sau hai khối tích chập và pooling, kích thước không gian đi theo `48 → 46 → 23 → 21 → 19 → 9`. Vì vậy, `nn.Flatten()` tạo vector có `218 × 9 × 9 = 17.658` đặc trưng cho lớp kết nối đầy đủ đầu tiên.

| Thành phần Keras | Cách chuyển sang PyTorch |
| --- | --- |
| `Conv2D` với kernel 3 × 3 và ReLU | `nn.Conv2d` rồi `nn.ReLU`. Dùng `padding=1` cho `same` và `padding=0` cho `valid`. |
| `MaxPooling2D` với cửa sổ 2 × 2 | `nn.MaxPool2d(kernel_size=2)` với bước trượt 2. |
| `Dropout` | `nn.Dropout(0.25)` sau mỗi khối pooling và `nn.Dropout(0.5)` sau lớp ẩn của MLP. |
| `Flatten` và `Dense(1024, activation="relu")` | `nn.Flatten()`, `nn.Linear(218 * 9 * 9, 1024)` và `nn.ReLU()`. |
| `Dense(7, activation="softmax")` và categorical cross-entropy | `nn.Linear(1024, 7)` trả về 7 logits. `nn.CrossEntropyLoss` nhận trực tiếp logits và chỉ số lớp, rồi thực hiện log-softmax bên trong. |

Các lớp tích chập và tuyến tính dùng Xavier uniform để khởi tạo trọng số và đặt bias bằng 0, tương ứng với Glorot uniform và bias 0 trong bản Keras.

## 3. Tăng cường dữ liệu và huấn luyện

`AugmentedFER2013` thay cho `ImageDataGenerator`. Khi `DataLoader` lấy ảnh train, `RandomAffine.get_params` lấy tham số xoay trong khoảng ±45° và dịch ngang, dọc trong phạm vi 20% kích thước tương ứng. Ảnh được đệm bằng giá trị pixel ở mép trước khi áp dụng affine với nội suy song tuyến tính, sau đó cắt về 48 × 48 và lật ngang với xác suất 50%. Tập PublicTest dùng `TensorDataset` và được đọc trực tiếp để đánh giá.

Notebook thay `model.compile()` và `fit_generator()` bằng `nn.CrossEntropyLoss`, Adam với learning rate `1e-4` và vòng lặp tối đa 100 epoch. Mỗi batch có tối đa 128 ảnh. Hàm `run_epoch` đặt mô hình ở chế độ train khi nhận optimizer, sau đó xóa gradient cũ, chạy dự đoán, tính loss, gọi `backward()` và cập nhật trọng số. Khi đánh giá, hàm chuyển mô hình sang chế độ eval và tắt tính gradient. Loss được tính trung bình theo số mẫu, còn accuracy là tỷ lệ dự đoán đúng. [Mã hàm huấn luyện dùng chung](shared_training.py)

Cuối mỗi epoch, notebook ghi loss và accuracy của train và PublicTest vào `history`. Bộ đếm dừng sớm trở về 0 khi `val_loss` giảm xuống dưới giá trị tốt nhất đã ghi nhận. Nếu `val_loss` không cải thiện trong 5 epoch liên tiếp, vòng lặp kết thúc. Phần đánh giá dùng trọng số từ epoch cuối đã chạy, theo cấu hình của bản Keras.

## 4. Đánh giá và kết quả

Notebook dùng PublicTest cho cả validation và đánh giá cuối, theo cách chia tập của bài gốc. Với một ảnh ngẫu nhiên, `unsqueeze(0)` thêm chiều batch, rồi `model.eval()` và `torch.inference_mode()` được dùng để dự đoán. `argmax(dim=1)` chọn lớp có logit lớn nhất để hiển thị cùng ảnh và nhãn thật.

| Chỉ số | Kết quả lưu trong notebook |
| --- | ---: |
| Epoch đạt validation loss thấp nhất | 62 |
| Validation loss thấp nhất | 1,0117 |
| Epoch dừng sớm | 67 |
| Loss khi đánh giá cuối trên PublicTest | 1,0214 |
| Accuracy khi đánh giá cuối trên PublicTest | 61,97% |

Các số liệu trên được lấy từ đầu ra của một lần chạy đã lưu trong notebook. Accuracy 61,97% được đo trên PublicTest, tập đã dùng để theo dõi dừng sớm. [Kết quả lưu trong notebook](3_Emotion_Reg_CNN.ipynb)
