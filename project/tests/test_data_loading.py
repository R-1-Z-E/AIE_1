import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.data.loader import load_data

def test_load_data_success(tmp_path):
    """Проверка успешной загрузки данных из существующего файла."""
    df = pd.DataFrame({"close": [100, 101, 102]})
    file_path = tmp_path / "dummy_data.csv"
    df.to_csv(file_path, index=False)
    
    loaded_df = load_data(file_path)
    
    assert isinstance(loaded_df, pd.DataFrame)
    assert len(loaded_df) == 3

def test_load_data_file_not_found():
    """Проверка обработки ошибки при отсутствии файла."""
    fake_path = Path("fake_dir/fake_data.csv")
    try:
        load_data(fake_path)
        assert False, "Функция должна была выдать FileNotFoundError"
    except FileNotFoundError:
        assert True