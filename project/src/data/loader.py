import pandas as pd
from pathlib import Path
from src.utils.logging import setup_logger

logger = setup_logger("DataLoader")

def load_data(file_path: Path) -> pd.DataFrame:
    """Загружает исторические котировки из CSV."""
    if not file_path.exists():
        logger.error(f"Файл не найден: {file_path}")
        raise FileNotFoundError(f"Нет данных по пути: {file_path}")
        
    df = pd.read_csv(file_path)
    logger.info(f"Данные успешно загружены. Размер датасета: {df.shape}")
    return df