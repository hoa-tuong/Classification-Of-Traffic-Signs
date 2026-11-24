from pathlib import Path
from torch.utils.data import DataLoader
from src.data.dataset import TrafficSignDataset
from src.data.transforms import get_train_transforms, get_val_transforms


def create_dataloaders(
    summary_csv: str | Path,
    base_dir: str | Path = "data",
    img_size: tuple[int, int] = (64, 64),
    batch_size: int = 64,
    num_workers: int = 0
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Khoi tao 3 DataLoader: Train, Validation va Test tu file dataset_summary.csv.

    Args:
        summary_csv: Duong dan toi file dataset_summary.csv.
        base_dir: Thu muc goc chua du lieu (mac dinh: 'data').
        img_size: Kich thuoc anh (mac dinh: (64, 64)).
        batch_size: Kich thuoc moi batch du lieu.
        num_workers: So worker nạp du lieu (mac dinh: 0 cho Windows).

    Returns:
        tuple (train_loader, val_loader, test_loader)
    """
    summary_path = Path(summary_csv)
    base_path = Path(base_dir)

    train_transform = get_train_transforms(img_size)
    val_transform = get_val_transforms(img_size)

    train_dataset = TrafficSignDataset(summary_path, base_dir=base_path, split="train", transform=train_transform)
    val_dataset = TrafficSignDataset(summary_path, base_dir=base_path, split="val", transform=val_transform)
    test_dataset = TrafficSignDataset(summary_path, base_dir=base_path, split="test", transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def get_test_loader(
    summary_csv: str | Path,
    base_dir: str | Path = "data",
    img_size: tuple[int, int] = (64, 64),
    batch_size: int = 64,
    num_workers: int = 0
) -> DataLoader:
    """Khoi tao DataLoader danh rieng cho tap Test phuc vu danh gia mo hinh."""
    val_transform = get_val_transforms(img_size)
    test_dataset = TrafficSignDataset(summary_csv, base_dir=base_dir, split="test", transform=val_transform)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
