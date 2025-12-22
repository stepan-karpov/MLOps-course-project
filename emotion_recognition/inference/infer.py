"""Модуль для инференса модели."""

from pathlib import Path
from typing import Dict, Tuple

import dvc.api
import librosa
import numpy as np
import torch
from omegaconf import DictConfig

from emotion_recognition.models.cnn import EmotionCNN

# Маппинг эмоций
EMOTION_LABELS = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised",
]


def preprocess_audio(
    audio_path: str,
    sample_rate: int = 16000,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    duration: float = 3.0,
) -> torch.Tensor:
    """
    Препроцессинг аудио файла.

    Args:
        audio_path: Путь к аудио файлу
        sample_rate: Частота дискретизации
        n_mels: Количество мел-фильтров
        n_fft: Размер окна FFT
        hop_length: Шаг окна
        duration: Длительность аудио в секундах

    Returns:
        Препроцессированная спектрограмма
    """
    # Загрузка аудио
    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)

    # Обрезка или паддинг до нужной длины
    target_length = int(duration * sample_rate)
    if len(audio) > target_length:
        audio = audio[:target_length]
    else:
        audio = np.pad(audio, (0, target_length - len(audio)), mode="constant")

    # Преобразование в мел-спектрограмму
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    # Логарифмическое масштабирование
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Нормализация
    mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)

    # Преобразование в тензор и добавление канала
    mel_spec_tensor = torch.FloatTensor(mel_spec_db).unsqueeze(0).unsqueeze(0)

    return mel_spec_tensor


def load_model(model_path: str, device: str = "cpu", cfg: DictConfig = None) -> EmotionCNN:
    """
    Загрузка обученной модели.

    Args:
        model_path: Путь к чекпоинту модели
        device: Устройство для инференса
        cfg: Конфигурация (опционально, для определения размеров входных данных)

    Returns:
        Загруженная модель
    """
    # Загружаем checkpoint для получения hyperparameters и размеров
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    hparams = checkpoint.get("hyper_parameters", {})
    state_dict = checkpoint["state_dict"]

    # Получаем размер входного слоя fc1 из checkpoint
    fc1_weight = state_dict.get("fc1.weight")
    if fc1_weight is None:
        raise ValueError("Checkpoint не содержит fc1.weight. Модель не была полностью обучена.")

    expected_fc1_input_size = fc1_weight.shape[1]

    # Получаем параметры модели из checkpoint
    conv_layers = hparams.get("conv_layers", cfg.model.conv_layers if cfg else None)
    if conv_layers is None:
        raise ValueError("Не удалось определить параметры модели из checkpoint или конфига")

    # Создаем модель с параметрами из checkpoint
    model = EmotionCNN(
        num_classes=hparams.get("num_classes", cfg.model.num_classes if cfg else 8),
        input_channels=hparams.get("input_channels", cfg.model.input_channels if cfg else 1),
        conv_layers=conv_layers,
        dropout=hparams.get("dropout", cfg.model.dropout if cfg else 0.3),
        fc_units=hparams.get("fc_units", cfg.model.fc_units if cfg else 256),
    )

    # Подбираем правильные размеры входных данных экспериментально
    possible_n_mels = [64, 128, 256]
    possible_time_frames = [100, 150, 200, 250, 300, 350, 400, 450, 500]

    n_mels = None
    time_frames = None

    for test_n_mels in possible_n_mels:
        for test_time_frames in possible_time_frames:
            # Создаем временную модель для теста
            test_model = EmotionCNN(
                num_classes=hparams.get("num_classes", cfg.model.num_classes if cfg else 8),
                input_channels=hparams.get(
                    "input_channels", cfg.model.input_channels if cfg else 1
                ),
                conv_layers=conv_layers,
                dropout=hparams.get("dropout", cfg.model.dropout if cfg else 0.3),
                fc_units=hparams.get("fc_units", cfg.model.fc_units if cfg else 256),
            )
            test_input = torch.randn(1, 1, test_n_mels, test_time_frames)
            with torch.no_grad():
                _ = test_model(test_input)

            # Проверяем размер fc1
            if (
                test_model.fc1 is not None
                and test_model.fc1.weight.shape[1] == expected_fc1_input_size
            ):
                n_mels = test_n_mels
                time_frames = test_time_frames
                break

        if n_mels is not None:
            break

    # Если не нашли, используем значения из конфига и пробуем вычислить
    if n_mels is None and cfg is not None:
        n_mels = cfg.data.n_mels
        # Пробуем разные time_frames
        for test_time_frames in range(50, 1000, 10):
            test_model = EmotionCNN(
                num_classes=hparams.get("num_classes", cfg.model.num_classes),
                input_channels=hparams.get("input_channels", cfg.model.input_channels),
                conv_layers=conv_layers,
                dropout=hparams.get("dropout", cfg.model.dropout),
                fc_units=hparams.get("fc_units", cfg.model.fc_units),
            )
            test_input = torch.randn(1, 1, n_mels, test_time_frames)
            with torch.no_grad():
                _ = test_model(test_input)

            if (
                test_model.fc1 is not None
                and test_model.fc1.weight.shape[1] == expected_fc1_input_size
            ):
                time_frames = test_time_frames
                break

    if n_mels is None or time_frames is None:
        raise ValueError(
            f"Не удалось найти правильные размеры входных данных. "
            f"Ожидаемый размер fc1: {expected_fc1_input_size}"
        )

    # Выполняем dummy forward pass для инициализации fc1 с правильными размерами
    dummy_input = torch.randn(1, 1, n_mels, time_frames)
    with torch.no_grad():
        _ = model(dummy_input)

    # Загружаем веса из checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    return model


def predict(
    model: EmotionCNN,
    audio_path: str,
    cfg: DictConfig,
    device: str = "cpu",
) -> Tuple[str, Dict[str, float]]:
    """
    Предсказание эмоции для аудио файла.

    Args:
        model: Обученная модель
        audio_path: Путь к аудио файлу
        cfg: Конфигурация
        device: Устройство для инференса

    Returns:
        Кортеж (предсказанная эмоция, распределение вероятностей)
    """
    # Препроцессинг
    mel_spec = preprocess_audio(
        audio_path,
        sample_rate=cfg.data.sample_rate,
        n_mels=cfg.data.n_mels,
        n_fft=cfg.data.n_fft,
        hop_length=cfg.data.hop_length,
        duration=cfg.data.duration,
    )

    # Инференс
    model.eval()
    with torch.no_grad():
        mel_spec = mel_spec.to(device)
        logits = model(mel_spec)
        probabilities = torch.softmax(logits, dim=1)
        predicted_idx = torch.argmax(probabilities, dim=1).item()

    # Формирование результата
    predicted_emotion = EMOTION_LABELS[predicted_idx]
    emotion_probs = {
        EMOTION_LABELS[i]: probabilities[0][i].item() for i in range(len(EMOTION_LABELS))
    }

    return predicted_emotion, emotion_probs


def ensure_data_available(data_dir: str) -> None:
    """Проверка наличия данных и их загрузка через DVC."""
    data_path = Path(data_dir)
    if not data_path.exists() or not list(data_path.rglob("*.wav")):
        print("Данные не найдены, попытка загрузки через DVC...")
        try:
            dvc.api.get_url("data/raw", repo=".")
            from emotion_recognition.data.download_data import download_data

            download_data(data_dir)
        except Exception as e:
            print(f"Ошибка при загрузке данных: {e}")


def infer_cli(audio_path: str) -> None:
    """
    CLI функция для инференса.

    Args:
        audio_path: Путь к аудио файлу для предсказания
    """
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    # Получаем корень проекта (абсолютный путь)
    project_root = Path(__file__).resolve().parent.parent.parent
    config_dir = project_root / "configs"

    # Инициализация Hydra с абсолютным путем к конфигам
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="config")

        # Проверка данных (может потребоваться для нормализации)
        ensure_data_available(cfg.paths.data_dir)

        # Загрузка модели
        model_path = project_root / cfg.inference.model_path
        if not model_path.exists():
            raise FileNotFoundError(f"Модель не найдена: {model_path}")

        device = cfg.inference.device
        model = load_model(str(model_path), device=device, cfg=cfg)

        # Предсказание
        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"Аудио файл не найден: {audio_path}")

        predicted_emotion, emotion_probs = predict(model, str(audio_file), cfg, device=device)

        # Вывод результата
        print(f"\nПредсказанная эмоция: {predicted_emotion}")
        print("\nРаспределение вероятностей:")
        for emotion, prob in sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True):
            print(f"  {emotion}: {prob:.4f}")


if __name__ == "__main__":
    import fire

    fire.Fire(infer_cli)
