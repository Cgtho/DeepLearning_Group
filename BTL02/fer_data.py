import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

EMOTIONS = ("Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral")


def load_images(df):
    """Đọc chuỗi pixel thành mảng uint8 có dạng (N, 48, 48, 1)."""
    pixels = np.empty((len(df), 48 * 48), dtype=np.uint8)
    for i, text in enumerate(df["pixels"]):
        values = np.fromstring(text, dtype=np.uint8, sep=" ")
        if values.size != 48 * 48:
            raise ValueError(f"Dòng {i} có {values.size} pixel, cần 2304.")
        pixels[i] = values
    return pixels.reshape(-1, 48, 48, 1)


class EmotionDataset(Dataset):
    def __init__(self, images, labels=None, transform=None):
        self.images = torch.from_numpy(images).permute(0, 3, 1, 2)
        self.labels = None if labels is None else torch.from_numpy(labels.astype(np.int64))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        if self.transform is not None:
            image = self.transform(image)
        if self.labels is None:
            return image
        return image, self.labels[index]


def split_train_valid(df):
    return train_test_split(
        np.arange(len(df)), test_size=0.2, random_state=42, stratify=df["emotion"]
    )


def make_dataset(frame):
    labels = frame["emotion"].to_numpy()
    if not np.isin(labels, np.arange(7)).all():
        raise ValueError("FER2013 yêu cầu nhãn số nguyên từ 0 đến 6.")
    images = load_images(frame).astype(np.float32)
    images /= 255.0
    return EmotionDataset(images, labels)


def training_loaders(data_dir, batch_size, seed):
    frame = pd.read_csv(data_dir / "train.csv")
    train_idx, valid_idx = split_train_valid(frame)
    generator = torch.Generator().manual_seed(seed)
    train = DataLoader(make_dataset(frame.iloc[train_idx]), batch_size=batch_size,
                       shuffle=True, generator=generator)
    valid = DataLoader(make_dataset(frame.iloc[valid_idx]), batch_size=batch_size)
    return train, valid


def test_loader(data_dir, batch_size):
    test = pd.read_csv(data_dir / "test.csv")
    if "emotion" not in test:
        # Chỉ lấy nhãn khi toàn bộ ảnh và thứ tự khớp với phần test của bản gốc.
        reference_path = data_dir / "fer2013.csv"
        if not reference_path.is_file():
            raise FileNotFoundError(
                "test.csv không có cột emotion và thiếu fer2013.csv để khôi phục nhãn. "
                "Hãy dùng --split validation để đánh giá trên tập validation có nhãn, "
                "hoặc đặt fer2013.csv vào thư mục data để đánh giá test."
            )
        pieces = []
        for chunk in pd.read_csv(reference_path, chunksize=2048):
            pieces.append(chunk.loc[chunk["Usage"].isin(["PublicTest", "PrivateTest"])])
        reference = pd.concat(pieces, ignore_index=True)
        if not test["pixels"].equals(reference["pixels"]):
            raise ValueError("Ảnh test.csv không khớp với fer2013.csv. Kiểm tra nguồn nhãn.")
        test["emotion"] = reference["emotion"]
    return DataLoader(make_dataset(test), batch_size=batch_size)
