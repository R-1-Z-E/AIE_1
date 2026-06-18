import logging

def setup_logger(name: str = "TradingService") -> logging.Logger:
    """Настройка единого логгера для всего проекта."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(name)