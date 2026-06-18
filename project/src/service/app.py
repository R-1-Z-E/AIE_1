import pandas as pd
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from joblib import load

from src.utils.logging import setup_logger

logger = setup_logger("TradingAPI")
model = None

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = PROJECT_DIR / "artifacts" / "model.pkl"

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if MODEL_PATH.exists():
        model = load(MODEL_PATH)
        logger.info(f"Модель успешно загружена из {MODEL_PATH}")
    else:
        logger.error(f"Модель не найдена по пути: {MODEL_PATH}. Сначала запустите обучение!")
    yield
    logger.info("Сервис остановлен.")

app = FastAPI(title="Trading Signal API", lifespan=lifespan)

class TradeRequest(BaseModel):
    sma_7: float
    sma_30: float
    volatility: float
    momentum_3d: float

@app.get("/health")
def health_check():
    """Проверка доступности сервиса и состояния модели."""
    if model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    return {"status": "ok"}

@app.post("/predict")
def predict_signal(request: TradeRequest):
    """Возвращает торговый сигнал на основе входящего вектора признаков."""
    if model is None:
        raise HTTPException(status_code=503, detail="Модель недоступна")
        
    input_data = pd.DataFrame([request.model_dump()])
    probability_up = model.predict_proba(input_data)[0][1]
    
    action = "BUY" if probability_up > 0.5 else "WAIT"
    return {
        "action": action, 
        "probability": round(float(probability_up), 4)
    }