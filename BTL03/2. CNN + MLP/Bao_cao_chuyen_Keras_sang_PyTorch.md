# Chuyển mô hình CNN + MLP nhận diện biểu cảm từ Keras sang PyTorch

Notebook `3_Emotion_Reg_CNN.ipynb` dùng ảnh khuôn mặt xám 48 × 48 của FER2013 để phân loại bảy biểu cảm. Phần CNN trích xuất đặc trưng từ ảnh, sau đó phần MLP sử dụng các đặc trưng này để tính điểm cho từng lớp. Đây là phiên bản huấn luyện cố định 15 epoch trên ảnh gốc. [Notebook PyTorch](3_Emotion_Reg_CNN.ipynb)

## 1. Chuyển ảnh và nhãn thành tensor

Notebook đọc lại hai tệp CSV đã chuẩn bị trong thư mục `../4. Training Utilities/`. Tệp `fer2013_training_onehot.csv` chứa 28.709 ảnh train. Tệp `fer2013_publictest_onehot.csv` chứa 3.589 ảnh, dùng để tính chỉ số sau mỗi epoch và đánh giá cuối cùng theo cách chia tập của bản Keras.

Mỗi dòng gồm 2.304 giá trị pixel và 7 giá trị nhãn one-hot. Mô hình Keras khai báo đầu vào theo thứ tự chiều cao, chiều rộng, kênh. Trong PyTorch, `Conv2d` nhận chiều kênh ngay sau chiều batch. Vì vậy, hàm `load_fer2013` đọc pixel dưới dạng `float32` và dùng `reshape(-1, 1, 48, 48)`. Thang pixel 0–255 được giữ như trong tệp nguồn.

Với nhãn, `argmax(axis=1)` lấy vị trí của giá trị 1 trong từng hàng one-hot rồi chuyển kết quả sang `torch.long`. Ví dụ, nhãn `[0, 0, 0, 1, 0, 0, 0]` trở thành chỉ số `3`, tương ứng với `happy`. Thứ tự lớp từ 0 đến 6 lấy theo hàm `get_emotion` của bản gốc, lần lượt là `angry`, `disgust`, `fear`, `happy`, `sad`, `surprise` và `neutral`.

## 2. Chuyển kiến trúc CNN + MLP

| Thành phần Keras | Cài đặt PyTorch | Đầu ra của một ảnh |
| --- | --- | --- |
| Hai lớp Conv2D 64 bộ lọc, ReLU, max pooling và dropout 0,25 | Hai lớp `nn.Conv2d` 3 × 3 có padding lần lượt là 1 và 0, mỗi lớp có `nn.ReLU()`. Tiếp theo là `nn.MaxPool2d(2)` và `nn.Dropout(0.25)`. | `[64, 23, 23]` |
| Bốn lớp Conv2D có 128, 128, 218 và 218 bộ lọc, ReLU, max pooling và dropout 0,25 | Dùng padding lần lượt 1, 0, 1, 0 cho bốn lớp tích chập. Sau pooling 2 × 2 là `nn.Dropout(0.25)`. | `[218, 9, 9]` |
| Flatten, Dense 1.024 nút với ReLU và dropout 0,5 | `nn.Flatten()` ghép 218 × 9 × 9 = 17.658 đặc trưng thành một vector. `nn.Linear(17658, 1024)` nối với `nn.ReLU()` và `nn.Dropout(0.5)`. | `[1024]` |
| Dense 7 nút với softmax | `nn.Linear(1024, 7)` trả về 7 điểm số (logits). | `[7]` |

Keras dùng `categorical_crossentropy` trên đầu ra softmax. Bản PyTorch truyền logits và chỉ số lớp trực tiếp vào `nn.CrossEntropyLoss()`, vì hàm này đã thực hiện log-softmax khi tính loss. Các lớp tích chập và tuyến tính dùng Xavier uniform cho trọng số và khởi tạo bias bằng 0. Toàn bộ mô hình có **19.028.287 tham số học được**.

## 3. Thay vòng lặp huấn luyện và dự đoán

Trong Keras, `model.fit()` thực hiện cả cập nhật trọng số và tính chỉ số validation. Bản PyTorch dùng hai `DataLoader` với batch size 128. Loader train xáo trộn ảnh sau mỗi lượt duyệt, còn loader PublicTest đọc ảnh theo thứ tự. Cả hai dùng trực tiếp ảnh trong CSV.

Mỗi epoch gọi `run_epoch` hai lần. Lần train nhận optimizer Adam với learning rate `1e-4`, bật dropout, tính gradient bằng `loss.backward()` và cập nhật trọng số bằng `optimizer.step()`. Lần validation tắt dropout và tính loss, accuracy trên PublicTest. Notebook lưu bốn chỉ số vào `history` qua đủ 15 epoch. [Hàm run_epoch trong shared_training.py](<../4. Training Utilities/shared_training.py>)

Sau epoch 15, notebook đánh giá trọng số cuối trên toàn bộ PublicTest. Để dự đoán một ảnh, `sample.unsqueeze(0)` tạo batch `[1, 1, 48, 48]`. Trong `torch.inference_mode()`, mô hình trả về `[1, 7]`. Lệnh `argmax(dim=1)` chọn lớp có điểm cao nhất để hiển thị cùng ảnh và nhãn thật.

## 4. Kiểm tra bản chuyển

Đã đọc đủ hai CSV bằng hàm trong notebook và kiểm tra kích thước tensor. Vòng lặp được chạy thử trên CPU với 8 ảnh train và 4 ảnh PublicTest trong 15 epoch để kiểm tra cập nhật trọng số, ghi lịch sử, đánh giá và dự đoán.

Đầu ra lưu trong **notebook Keras gốc** ghi nhận loss 1,4070 và accuracy 59,54% trên PublicTest. Để đo các chỉ số tương ứng của PyTorch trên toàn bộ dữ liệu, chạy các cell của notebook theo thứ tự và đọc `Test loss`, `Test accuracy` ở phần đánh giá cuối.
