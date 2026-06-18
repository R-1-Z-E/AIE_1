import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

def test_train_module_imports():
    """Проверка импортов и доступности пайплайна обучения."""
    try:
        from src.models.train import run_training
        assert callable(run_training)
    except ImportError as e:
        assert False, f"Ошибка импорта в модуле обучения: {e}"