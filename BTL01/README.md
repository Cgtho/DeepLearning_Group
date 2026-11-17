# MLP xấp xỉ hàm Gaussian hai chiều

## Nội dung
- `main_mlp_approximation.ipynb`: notebook triển khai và khảo sát MLP bằng PyTorch.
- `data/`: các tập Train Pool, Validation và Test được notebook sinh ra.
- `logs/`: log từng lần chạy, thống kê A/B/C và loss theo epoch.
- `figures/`: biểu đồ dữ liệu, kiến trúc, loss và dự đoán.
- `checkpoints/`: checkpoint tốt nhất theo Validation loss.

## Môi trường
Cần Python, Jupyter Notebook và các thư viện trong `requirements.txt`. Notebook tự chọn CUDA nếu có.

## Cài đặt và chạy
```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter notebook
```
Mở notebook và chọn **Kernel → Restart & Run All** (hoặc **Run All**) để sinh dữ liệu, chạy thực nghiệm và tạo các thư mục đầu ra. Chạy từ thư mục gốc của dự án để các đường dẫn tương đối hoạt động đúng.

## Quy trình đánh giá
- A/B/C: chỉ sử dụng Train và Validation; mỗi cấu hình chạy với seed `42, 100, 2026`.
- Cấu hình cuối: `[32, 32]`, `tanh`; lựa chọn dựa trên Validation và thời gian.
- Test: chỉ đánh giá một lần sau khi chọn cấu hình cuối.

## Thông tin nhóm
3123411094		Nguyễn Đình Hiệp (25%): 
Xây dựng luồng huấn luyện (train_and_evaluate_mlp) và thực hiện Thực nghiệm A (khảo sát quy mô dữ liệu A)

3123411218		Nguyễn Huỳnh Tấn Phát (25%): 
Quản lý tiến độ nhóm, thực hiện Thực nghiệm B (khảo sát độ rộng B) và phân tích khả năng biểu diễn. 

3123411274		Dương Minh Thiện (25%): 
Thực hiện Thực nghiệm C (khảo sát độ sâu C), đánh giá mô hình cuối trên tập Test và trực quan hóa kết quả.

3123411345		Ngô Trường Vũ (25%): 
Xử lý tập dữ liệu (Gaussian 2D, chuẩn hóa Z-score) và thiết kế lớp mô hình cốt lõi DynamicMLP.