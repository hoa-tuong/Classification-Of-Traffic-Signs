from pathlib import Path
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset


class TrafficSignDataset(Dataset):
    """Dataset load anh bien bao tu file CSV hoac danh sach duong dan anh."""

    def __init__(self, data_frame: pd.DataFrame, img_col: str = "path", label_col: str = "label", transform=None):
        self.df = data_frame.reset_index(drop=True)
        self.img_col = img_col
        self.label_col = label_col
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        img_path = self.df.loc[idx, self.img_col]
        label = int(self.df.loc[idx, self.label_col])

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

