from torchvision import transforms


def get_train_transforms(img_size: tuple[int, int] = (64, 64)):
    """Tap hop cac phep bien doi tang cuong du lieu (Data Augmentation) voi nen xam trung tinh 128."""
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomRotation(degrees=20, fill=128),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.92, 1.08), fill=128),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def get_val_transforms(img_size: tuple[int, int] = (64, 64)):
    """Phep chuan hoa danh cho tap validation va test (khong xoay/bien dang)."""
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


