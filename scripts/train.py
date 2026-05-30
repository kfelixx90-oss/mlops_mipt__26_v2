"""
Тренировка модели классификации жестов.
"""

import sys
from pathlib import Path

# Добавляем корень проекта в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
import hydra
from omegaconf import DictConfig


# Импорты из ваших модулей (раскомментируйте, когда создадите их)
# from pedestrian_classifier.data.datamodule import GestureDataModule
# from pedestrian_classifier.models.gesture_cnn import GestureClassifier


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: DictConfig):
    """Основная функция тренировки."""
    print("Запуск тренировки...")
    print(f"Конфигурация: {cfg}")

    # Пример: создание фейковых данных для теста
    print("Создание фейковых данных для теста...")
    X_train = torch.randn(100, 3, 224, 224)
    y_train = torch.randint(0, 5, (100,))
    X_val = torch.randn(20, 3, 224, 224)
    y_val = torch.randint(0, 5, (20,))

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.training.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.training.batch_size
    )

    # Простая модель для теста
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, 3),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(16, 5),
    )

    # Логирование в MLflow
    mlflow_logger = MLFlowLogger(
        experiment_name="gesture_classification", tracking_uri=cfg.logging.mlflow_uri
    )

    # Тренер
    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator="auto",
        logger=mlflow_logger,
        enable_progress_bar=True,
    )

    # Тренировка
    print("Начало обучения...")
    trainer.fit(model, train_loader, val_loader)
    print("Обучение завершено!")

    # Сохранение модели
    torch.save(model.state_dict(), "model.pth")
    print("Модель сохранена в model.pth")


if __name__ == "__main__":
    train()
