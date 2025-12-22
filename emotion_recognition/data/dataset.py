"""Dataset для загрузки и препроцессинга аудио данных."""

from pathlib import Path
from typing import Optional, Tuple

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset


class EmotionDataset(Dataset):
    """Dataset для распознавания эмоций по аудио."""

    # Маппинг эмоций RAVDESS
    EMOTION_MAP = {
        "01": "neutral",
        "02": "calm",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fearful",
        "07": "disgust",
        "08": "surprised",
    }

    def __init__(
        self,
        data_dir: str,
        sample_rate: int = 16000,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        duration: float = 3.0,
        split: str = "train",
        indices: Optional[np.ndarray] = None,
    ):
        """
        Инициализация dataset.

        Args:
            data_dir: Путь к директории с данными
            sample_rate: Частота дискретизации
            n_mels: Количество мел-фильтров
            n_fft: Размер окна FFT
            hop_length: Шаг окна
            duration: Длительность аудио в секундах
            split: Разделение (train/val/test)
            indices: Индексы файлов для этого split
        """
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.duration = duration
        self.split = split

        # Собираем все аудио файлы
        self.audio_files = list(self.data_dir.rglob("*.wav"))
        if not self.audio_files:
            raise ValueError(f"Не найдено аудио файлов в {data_dir}")

        # Фильтруем по индексам если указаны
        if indices is not None:
            self.audio_files = [self.audio_files[i] for i in indices]

        # Создаем список меток
        self.labels = []
        self.label_to_idx = {label: idx for idx, label in enumerate(self.EMOTION_MAP.values())}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

        for audio_file in self.audio_files:
            emotion_code = audio_file.stem.split("-")[2]
            emotion = self.EMOTION_MAP.get(emotion_code, "neutral")
            self.labels.append(self.label_to_idx[emotion])

        self.num_classes = len(self.label_to_idx)

    def __len__(self) -> int:
        return len(self.audio_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Получение одного примера.

        Args:
            idx: Индекс примера

        Returns:
            Кортеж (спектрограмма, метка)

        Аудио файл (.wav)
        ↓
        Загрузка через librosa
        ↓
        [Амплитуды звука] (48000 чисел)
        ↓
        Обрезка/паддинг до 3 секунд
        ↓
        Преобразование в мел-спектрограмму
        ↓
        [Мел-спектрограмма] (64 × ~188 частотных полос × время)
        ↓
        Логарифмическое масштабирование (dB)
        ↓
        Нормализация (mean=0, std=1)
        ↓
        Преобразование в тензор + добавление канала
        ↓
        [Тензор] (1, 64, ~188) - готов для CNN!
        """
        audio_path = self.audio_files[idx]
        label = self.labels[idx]

        # Загрузка аудио
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)

        # Обрезка или паддинг до нужной длины
        target_length = int(self.duration * self.sample_rate)
        if len(audio) > target_length:
            audio = audio[:target_length]
        else:
            audio = np.pad(audio, (0, target_length - len(audio)), mode="constant")

        # Преобразование в мел-спектрограмму
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )

        # Логарифмическое масштабирование
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Нормализация
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)

        # Преобразование в тензор и добавление канала
        mel_spec_tensor = torch.FloatTensor(mel_spec_db).unsqueeze(0)

        return mel_spec_tensor, label


def split_dataset(
    dataset: EmotionDataset, train_split: float = 0.7, val_split: float = 0.15, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Разделение dataset на train/val/test.

    Args:
        dataset: Dataset для разделения
        train_split: Доля train
        val_split: Доля val
        seed: Random seed

    Returns:
        Кортеж (train_indices, val_indices, test_indices)
    """
    np.random.seed(seed)
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)

    n_total = len(indices)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)

    train_indices = indices[:n_train]
    val_indices = indices[n_train : n_train + n_val]
    test_indices = indices[n_train + n_val :]

    return train_indices, val_indices, test_indices
