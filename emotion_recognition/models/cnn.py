"""CNN модель для распознавания эмоций."""

from typing import Any, Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, F1Score


class EmotionCNN(pl.LightningModule):
    """CNN модель для классификации эмоций по мел-спектрограммам."""

    def __init__(
        self,
        num_classes: int = 8,
        input_channels: int = 1,
        conv_layers: list = None,
        dropout: float = 0.5,
        fc_units: int = 256,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
    ):
        """
        Инициализация модели.

        Args:
            num_classes: Количество классов эмоций
            input_channels: Количество входных каналов
            conv_layers: Список конфигураций сверточных слоев
            dropout: Dropout rate
            fc_units: Количество нейронов в полносвязном слое
            learning_rate: Learning rate
            weight_decay: Weight decay для оптимизатора
        """
        super().__init__()
        self.save_hyperparameters()

        if conv_layers is None:
            conv_layers = [
                {"filters": 32, "kernel_size": 3, "stride": 1, "padding": 1},
                {"filters": 64, "kernel_size": 3, "stride": 1, "padding": 1},
                {"filters": 128, "kernel_size": 3, "stride": 1, "padding": 1},
            ]

        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Сверточные слои
        conv_modules = []
        in_channels = input_channels
        for layer_config in conv_layers:
            conv_modules.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=layer_config["filters"],
                    kernel_size=layer_config["kernel_size"],
                    stride=layer_config["stride"],
                    padding=layer_config["padding"],
                )
            )
            conv_modules.append(nn.BatchNorm2d(layer_config["filters"]))
            conv_modules.append(nn.ReLU())
            conv_modules.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = layer_config["filters"]

        self.conv_layers = nn.Sequential(*conv_modules)

        # Вычисление размера после сверточных слоев будет выполнено динамически
        # Временный размер для инициализации
        self.fc_input_size = None
        self.fc_units = fc_units

        # Полносвязные слои
        self.fc1 = None
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc_units, num_classes)

        # Метрики
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Входной тензор (batch, channels, mel_bins, time_frames)

        Returns:
            Логиты классов
        """
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)

        # Динамическое создание FC слоя при первом проходе
        if self.fc1 is None:
            self.fc_input_size = x.size(1)
            self.fc1 = nn.Linear(self.fc_input_size, self.fc_units).to(x.device)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def training_step(self, batch: tuple, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Шаг обучения."""
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = self.train_accuracy(preds, y)
        f1 = self.train_f1(preds, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_f1", f1, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Шаг валидации."""
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = self.val_accuracy(preds, y)
        f1 = self.val_f1(preds, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1", f1, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch: tuple, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Шаг тестирования."""
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = self.test_accuracy(preds, y)
        f1 = self.test_f1(preds, y)

        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_acc", acc, on_step=False, on_epoch=True)
        self.log("test_f1", f1, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self) -> Dict[str, Any]:
        """Настройка оптимизатора."""
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }
