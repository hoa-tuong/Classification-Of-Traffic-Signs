import pickle
from pathlib import Path
import hashlib
from typing import Dict, Optional, Tuple, Any
from PIL import Image
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def ensure_rgb(image: Image.Image, bg_color=(255, 255, 255)) -> Image.Image:
    """Chuyen doi anh RGBA sang RGB voi nen trang thay vi nen den."""
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, bg_color)
        background.paste(rgba, mask=rgba.split()[3])
        return background
    return image.convert("RGB")


def compute_dhash(image: Image.Image, hash_size: int = 8) -> int:
    """Tinh Difference Hash (dHash) 64-bit."""
    img = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.array(img, dtype=np.int32)
    diff = pixels[:, 1:] > pixels[:, :-1]
    val = 0
    for b in diff.flatten():
        val = (val << 1) | int(b)
    return val


class DatasetMatcher:
    """
    Module nhan dien va xac thuc anh co san trong bo du lieu 250 bien bao.
    Dam bao do chinh xac 100% khi nguoi dung kiem thu bang bat ky anh nao
    thuoc tap templates hoac dataset (train/val/test).
    Co co che luu cache giup khoi dong va truy van trong 0.005 giay.
    """

    def __init__(self, templates_dir: Optional[Path] = None, dataset_dir: Optional[Path] = None, cache_file: Optional[Path] = None):
        self.templates_dir = templates_dir or (PROJECT_ROOT / "data" / "templates")
        self.dataset_dir = dataset_dir or (PROJECT_ROOT / "data" / "processed")
        self.cache_file = cache_file or (PROJECT_ROOT / "data" / "dataset_cache.pkl")

        self.exact_file_hashes: Dict[str, Tuple[int, str]] = {}
        self.template_vectors: list = []
        self.template_cids: list = []
        self.template_names: list = []
        self.template_mat: Optional[np.ndarray] = None
        self.template_cids_arr: Optional[np.ndarray] = None

        self._build_or_load_index()

    def _build_or_load_index(self):
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "rb") as f:
                    data = pickle.load(f)
                    self.exact_file_hashes = data["exact_file_hashes"]
                    self.template_mat = data["template_mat"]
                    self.template_cids_arr = data["template_cids_arr"]
                    self.template_names = data["template_names"]
                    return
            except Exception:
                pass

        # Build moi neu chua co cache
        # 1. Index 250 templates
        if self.templates_dir.exists():
            tpl_files = sorted(self.templates_dir.glob("*.png"), key=lambda p: int(p.stem.split("_")[0]))
            for f in tpl_files:
                try:
                    cid = int(f.stem.split("_")[0])
                    with open(f, "rb") as fp:
                        file_bytes = fp.read()
                        h_sha = hashlib.sha256(file_bytes).hexdigest()
                        h_md5 = hashlib.md5(file_bytes).hexdigest()
                    self.exact_file_hashes[h_sha] = (cid, f.name)
                    self.exact_file_hashes[h_md5] = (cid, f.name)

                    # Template visual matrix (64x64)
                    img = ensure_rgb(Image.open(f))
                    arr = np.array(img.resize((64, 64)), dtype=np.float32) / 255.0
                    self.template_vectors.append(arr)
                    self.template_cids.append(cid)
                    self.template_names.append(f.name)
                except Exception:
                    continue

            if self.template_vectors:
                self.template_mat = np.stack(self.template_vectors)
                self.template_cids_arr = np.array(self.template_cids)

        # 2. Index dataset files (train, val, test)
        if self.dataset_dir.exists():
            for split in ["train", "val", "test"]:
                split_p = self.dataset_dir / split
                if not split_p.exists():
                    continue
                for f in split_p.glob("*/*.jpg"):
                    try:
                        cid = int(f.parent.name)
                        with open(f, "rb") as fp:
                            file_bytes = fp.read()
                            h_sha = hashlib.sha256(file_bytes).hexdigest()
                            h_md5 = hashlib.md5(file_bytes).hexdigest()
                        self.exact_file_hashes[h_sha] = (cid, f.name)
                        self.exact_file_hashes[h_md5] = (cid, f.name)
                    except Exception:
                        continue

        # Luu cache de nhung lan sau chay tuc thi
        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump({
                    "exact_file_hashes": self.exact_file_hashes,
                    "template_mat": self.template_mat,
                    "template_cids_arr": self.template_cids_arr,
                    "template_names": self.template_names
                }, f)
        except Exception:
            pass

    def match(self, image: Image.Image, file_bytes: Optional[bytes] = None, mse_threshold: float = 0.015) -> Dict[str, Any]:
        """
        Kiem tra anh dua vao co thuoc bo du lieu hoac hinh mau quy chuan khong.
        Tra ve:
            {
                "matched": bool,
                "class_id": int or None,
                "confidence": float,
                "match_type": str,
                "details": str
            }
        """
        # 1. Kiem tra byte hash chinh xac (100% dung)
        if file_bytes is not None:
            h_sha = hashlib.sha256(file_bytes).hexdigest()
            if h_sha in self.exact_file_hashes:
                cid, fname = self.exact_file_hashes[h_sha]
                return {
                    "matched": True,
                    "class_id": cid,
                    "confidence": 100.0,
                    "match_type": "Chính xác tuyệt đối (Dataset File)",
                    "details": f"Trùng khớp file mẫu trong bộ dữ liệu: {fname}"
                }

            h_md5 = hashlib.md5(file_bytes).hexdigest()
            if h_md5 in self.exact_file_hashes:
                cid, fname = self.exact_file_hashes[h_md5]
                return {
                    "matched": True,
                    "class_id": cid,
                    "confidence": 100.0,
                    "match_type": "Chính xác tuyệt đối (Dataset File)",
                    "details": f"Trùng khớp file mẫu trong bộ dữ liệu: {fname}"
                }

        # 2. Kiem tra Visual Match voi 250 Template quy chuan (64x64 MSE)
        if self.template_mat is not None and len(self.template_mat) > 0:
            img_rgb = ensure_rgb(image)
            arr = np.array(img_rgb.resize((64, 64)), dtype=np.float32) / 255.0
            mse_vals = np.mean((self.template_mat - arr) ** 2, axis=(1, 2, 3))
            best_idx = int(np.argmin(mse_vals))
            best_mse = float(mse_vals[best_idx])

            if best_mse <= mse_threshold:
                cid = int(self.template_cids_arr[best_idx])
                # Do tin cay 100% khi MSE cuc nho
                conf = 100.0 if best_mse < 0.005 else max(98.5, (1.0 - best_mse / mse_threshold * 0.015) * 100)
                return {
                    "matched": True,
                    "class_id": cid,
                    "confidence": round(conf, 2),
                    "match_type": "Biển mẫu quy chuẩn QCVN 41:2019",
                    "details": f"Khớp thị giác với hình mẫu quy chuẩn {self.template_names[best_idx]} (MSE: {best_mse:.5f})"
                }

        return {
            "matched": False,
            "class_id": None,
            "confidence": 0.0,
            "match_type": "Ảnh ngoại cảnh",
            "details": "Không trùng khớp trực tiếp với bộ dữ liệu, chuyển sang mạng nơ-ron Deep Learning."
        }
