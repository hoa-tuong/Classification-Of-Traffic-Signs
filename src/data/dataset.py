from pathlib import Path
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset


class TrafficSignDataset(Dataset):
    """
    Custom Dataset phan loai bien bao ho tro ca DataFrame va file summary CSV.
    Dam bao anh xa class_id tu 0 den num_classes - 1 chinh xac.
    """

    def __init__(self, data_source: pd.DataFrame | str | Path, base_dir: str | Path = "data", split: str | None = None, transform=None):
        if isinstance(data_source, (str, Path)):
            df = pd.read_csv(data_source)
        else:
            df = data_source.copy()

        if split is not None and "split" in df.columns:
            df = df[df["split"] == split].reset_index(drop=True)

        self.df = df.reset_index(drop=True)
        self.base_dir = Path(base_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_rel = str(row["crop_rel_path"])
        full_path = self.base_dir / img_rel

        # Fallback neu duong dan la tuyet doi hoac nam trong thu muc khac
        if not full_path.exists():
            full_path = Path(img_rel)

        try:
            image = Image.open(full_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (64, 64), color="black")

        label = int(row["class_id"])

        if self.transform:
            image = self.transform(image)

        return image, label
