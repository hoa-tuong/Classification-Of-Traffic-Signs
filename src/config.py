from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(config_path: str | Path | None = None) -> dict:
    if config_path:
        p = Path(config_path)
        if not p.is_absolute():
            # Thu tim theo CWD truoc, neu khong co thi tim theo PROJECT_ROOT
            if not p.exists():
                p = PROJECT_ROOT / p
        path = p
    else:
        path = DEFAULT_CONFIG_PATH

    if not path.exists():
        raise FileNotFoundError(f"Khong tim thay file cau hinh tai: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Chuan hoa cac duong dan thanh tuyet doi dua tren PROJECT_ROOT
    if "data" in config:
        for k in ["raw_dir", "processed_dir", "summary_csv", "labels_path", "train_dir", "val_dir", "test_dir"]:
            if k in config["data"] and not Path(config["data"][k]).is_absolute():
                config["data"][k] = str(PROJECT_ROOT / config["data"][k])

    if "paths" in config:
        for k in ["checkpoint_dir", "best_model_path", "outputs_figure_dir", "outputs_log_dir", "classification_report_path"]:
            if k in config["paths"] and not Path(config["paths"][k]).is_absolute():
                config["paths"][k] = str(PROJECT_ROOT / config["paths"][k])

    config["project_root"] = str(PROJECT_ROOT)
    return config
