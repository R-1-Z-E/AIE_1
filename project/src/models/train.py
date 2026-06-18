from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

from src.utils.logging import setup_logger
from src.data.loader import load_data
from src.features.preprocess import add_features

logger = setup_logger("ModelTraining")

def run_training():
    PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_PATH = PROJECT_DIR / "data" / "trading_data.csv"
    SAVE_PATH = PROJECT_DIR / "artifacts" / "model.pkl"
    
    logger.info("--- Старт пайплайна обучения ---")
    
    raw_data = load_data(DATA_PATH)
    processed_data = add_features(raw_data)
    
    FEATURES = ['sma_7', 'sma_30', 'volatility', 'momentum_3d']
    X = processed_data[FEATURES]
    y = processed_data['target']
    
    logger.info("Обучение RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=200, 
        max_depth=5, 
        random_state=42, 
        class_weight="balanced"
    )
    model.fit(X, y)
    
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    dump(model, SAVE_PATH)
    logger.info(f"Финальная модель успешно сохранена в: {SAVE_PATH}")

if __name__ == "__main__":
    run_training()