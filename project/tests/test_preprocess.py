import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.features.preprocess import add_features

def test_add_features():
    """Проверка корректной генерации признаков (sanity-check)."""
    data = {
        "open": range(1, 40),
        "high": range(2, 41),
        "low": range(0, 39),
        "close": range(1, 40)
    }
    raw_df = pd.DataFrame(data)
    
    processed_df = add_features(raw_df)
    
    expected_cols = ['sma_7', 'sma_30', 'volatility', 'momentum_3d', 'target']
    for col in expected_cols:
        assert col in processed_df.columns, f"Отсутствует признак {col}"
        
    assert processed_df.isna().sum().sum() == 0
    assert set(processed_df['target'].unique()).issubset({0, 1})