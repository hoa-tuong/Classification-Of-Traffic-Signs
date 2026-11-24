import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "TrafficSign",
    log_file: str | Path | None = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Cau hinh he thong ghi log chuan (logging) cho du an.
    Xuat thong tin ra ca man hinh console va luu vao file log neu duoc chi dinh.

    Args:
        name: Ten dinh danh cua logger.
        log_file: Duong dan toi file log (vi du: 'outputs/logs/train.log').
        level: Muc do ghi log (mac dinh: logging.INFO).

    Returns:
        logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Tranh tao trung lap cac handler neu da khoi tao truoc do
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
