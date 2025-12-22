"""Модуль для обучения модели."""

from pathlib import Path
from typing import Tuple

import git
import mlflow
import mlflow.pytorch
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader

from emotion_recognition.data.dataset import EmotionDataset, split_dataset
from emotion_recognition.models.cnn import EmotionCNN


def get_git_commit_id() -> str:
    """Получение текущего git commit id."""
    try:
        repo = git.Repo(search_parent_directories=True)
        return repo.head.object.hexsha[:8]
    except Exception:
        return "unknown"


def ensure_data_available(data_dir: str) -> None:
    data_path = Path(data_dir)
    if not data_path.exists() or not list(data_path.rglob("*.wav")):
        print("Пожалуйста, убедитесь, что данные доступны или настройте DVC.")
        raise Exception("Данные не найдены")


def setup_project_directories(project_root: Path, cfg: DictConfig) -> Tuple[Path, Path, Path]:
    """
    Создает необходимые директории проекта и возвращает их пути.

    Args:
        project_root: Корневая директория проекта
        cfg: Конфигурация Hydra

    Returns:
        Кортеж (models_dir, plots_dir, data_dir) с абсолютными путями
    """
    models_dir = project_root / cfg.paths.models_dir
    plots_dir = project_root / cfg.paths.plots_dir
    data_dir = project_root / cfg.paths.data_dir

    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    return models_dir, plots_dir, data_dir


def setup(cfg: DictConfig, data_dir: Path) -> None:
    """
    Выполняет начальную настройку: устанавливает seed и проверяет наличие данных.

    Args:
        cfg: Конфигурация Hydra
        data_dir: Путь к директории с данными
    """
    # Установка seed для воспроизводимости
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Проверка и загрузка данных
    ensure_data_available(str(data_dir))


def create_dataset_and_loader(
    data_dir: Path,
    cfg: DictConfig,
    split: str,
    indices: np.ndarray,
    shuffle: bool = False,
) -> Tuple[EmotionDataset, DataLoader]:
    """
    Создает dataset и DataLoader для указанного split.

    Args:
        data_dir: Путь к директории с данными
        cfg: Конфигурация Hydra
        split: Название split (train/val/test)
        indices: Индексы файлов для этого split
        shuffle: Нужно ли перемешивать данные

    Returns:
        Кортеж (dataset, loader)
    """
    dataset = EmotionDataset(
        data_dir=str(data_dir),
        sample_rate=cfg.data.sample_rate,
        n_mels=cfg.data.n_mels,
        n_fft=cfg.data.n_fft,
        hop_length=cfg.data.hop_length,
        duration=cfg.data.duration,
        split=split,
        indices=indices,
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=shuffle,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
    )

    return dataset, loader


def train() -> None:
    """Обучение модели."""
    project_root = Path(__file__).resolve().parent.parent.parent
    config_dir = project_root / "configs"

    # Инициализация Hydra с абсолютным путем к конфигам
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="config")

        models_dir, plots_dir, data_dir = setup_project_directories(project_root, cfg)

        setup(cfg, data_dir)

        full_dataset = EmotionDataset(
            data_dir=str(data_dir),
            sample_rate=cfg.data.sample_rate,
            n_mels=cfg.data.n_mels,
            n_fft=cfg.data.n_fft,
            hop_length=cfg.data.hop_length,
            duration=cfg.data.duration,
        )

        # train/val/test
        seed = cfg.get("seed", 42)
        train_indices, val_indices, test_indices = split_dataset(
            full_dataset,
            train_split=cfg.data.train_split,
            val_split=cfg.data.val_split,
            seed=seed,
        )

        train_dataset, train_loader = create_dataset_and_loader(
            data_dir, cfg, "train", train_indices, shuffle=True
        )
        val_dataset, val_loader = create_dataset_and_loader(
            data_dir, cfg, "val", val_indices, shuffle=False
        )
        test_dataset, test_loader = create_dataset_and_loader(
            data_dir, cfg, "test", test_indices, shuffle=False
        )

        # Создание модели
        model = EmotionCNN(
            num_classes=cfg.model.num_classes,
            input_channels=cfg.model.input_channels,
            conv_layers=cfg.model.conv_layers,
            dropout=cfg.model.dropout,
            fc_units=cfg.model.fc_units,
            learning_rate=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )

        # mlflow run
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        mlflow.set_experiment(cfg.mlflow.experiment_name)

        mlflow.start_run()
        run_id = mlflow.active_run().info.run_id
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))
        mlflow.log_param("git_commit_id", get_git_commit_id())

        # MLFlowLogger будет использовать существующий run
        mlflow_logger = MLFlowLogger(
            experiment_name=cfg.mlflow.experiment_name,
            tracking_uri=cfg.mlflow.tracking_uri,
            run_id=run_id,
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath=str(models_dir),
            filename="best_model-{epoch:02d}-{val_loss:.2f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        )

        early_stopping = EarlyStopping(
            monitor=cfg.training.early_stopping.monitor,
            mode=cfg.training.early_stopping.mode,
            patience=cfg.training.early_stopping.patience,
        )

        # Trainer
        trainer = Trainer(
            max_epochs=cfg.training.num_epochs,
            logger=mlflow_logger,
            callbacks=[checkpoint_callback, early_stopping],
            accelerator="auto",
            devices="auto",
        )

        trainer.fit(model, train_loader, val_loader)
        trainer.test(model, test_loader)

        # Сохранение модели в MLflow
        best_model_path = checkpoint_callback.best_model_path
        if best_model_path:
            best_model = EmotionCNN.load_from_checkpoint(best_model_path, weights_only=False)
            mlflow.pytorch.log_model(
                pytorch_model=best_model,
                artifact_path="model",
                registered_model_name="emotion_cnn",
            )

        # Закрываем run
        mlflow.end_run()

        print(f"Обучение завершено. Лучшая модель сохранена: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    train()
