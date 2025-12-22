"""Экспорт модели в ONNX."""

from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

from emotion_recognition.models.cnn import EmotionCNN


def export_onnx() -> None:
    """Экспорт модели в ONNX формат."""
    project_root = Path(__file__).resolve().parent.parent.parent
    config_dir = project_root / "configs"

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="config")

        model_path = project_root / cfg.inference.model_path
        if not model_path.exists():
            raise FileNotFoundError(f"Модель не найдена: {model_path}")

        # Загружаем checkpoint для получения hyperparameters и размеров
        checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
        hparams = checkpoint.get("hyper_parameters", {})
        state_dict = checkpoint["state_dict"]

        fc1_weight = state_dict.get("fc1.weight")
        if fc1_weight is None:
            raise ValueError("Checkpoint не содержит fc1.weight. Модель не была полностью обучена.")

        expected_fc1_input_size = fc1_weight.shape[1]

        # Получаем параметры модели из checkpoint или конфига
        conv_layers = hparams.get("conv_layers", cfg.model.conv_layers)

        # Создаем модель с параметрами из checkpoint или конфига
        model = EmotionCNN(
            num_classes=hparams.get("num_classes", cfg.model.num_classes),
            input_channels=hparams.get("input_channels", cfg.model.input_channels),
            conv_layers=conv_layers,
            dropout=hparams.get("dropout", cfg.model.dropout),
            fc_units=hparams.get("fc_units", cfg.model.fc_units),
        )

        possible_n_mels = [64, 128, 256]
        possible_time_frames = [100, 150, 200, 250, 300, 350, 400, 450, 500]

        n_mels = None
        time_frames = None

        for test_n_mels in possible_n_mels:
            for test_time_frames in possible_time_frames:
                # Создаем временную модель для теста
                test_model = EmotionCNN(
                    num_classes=hparams.get("num_classes", cfg.model.num_classes),
                    input_channels=hparams.get("input_channels", cfg.model.input_channels),
                    conv_layers=conv_layers,
                    dropout=hparams.get("dropout", cfg.model.dropout),
                    fc_units=hparams.get("fc_units", cfg.model.fc_units),
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
        if n_mels is None:
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

        example_input = torch.randn(1, 1, n_mels, time_frames)

        # Экспорт в ONNX
        onnx_path = project_root / cfg.inference.onnx_path
        onnx_path.parent.mkdir(parents=True, exist_ok=True)

        torch.onnx.export(
            model,
            example_input,
            str(onnx_path),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["mel_spectrogram"],
            output_names=["emotion_logits"],
            dynamic_axes={
                "mel_spectrogram": {0: "batch_size"},
                "emotion_logits": {0: "batch_size"},
            },
        )

        print(f"Модель экспортирована в ONNX: {onnx_path}")


if __name__ == "__main__":
    export_onnx()
