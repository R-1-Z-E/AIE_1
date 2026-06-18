import pandas as pd
from src.utils.logging import setup_logger

logger = setup_logger("DataPreprocess")

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Генерирует технические индикаторы и целевую переменную (target)."""
    logger.info("Генерация признаков: SMA, Volatility, Momentum...")
    df = df.copy()
    
    # Расчёт математических признаков
    df['sma_7'] = df['close'].rolling(window=7).mean()
    df['sma_30'] = df['close'].rolling(window=30).mean()
    df['volatility'] = (df['high'] - df['low']) / df['open']
    df['momentum_3d'] = df['close'].pct_change(periods=3)
    
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    df_clean = df.dropna().copy()
    logger.info(f"Признаки сгенерированы. Итоговый размер: {df_clean.shape}")
    
    return df_clean