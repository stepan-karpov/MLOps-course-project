"""Главная точка входа для команд проекта."""

import sys
from pathlib import Path

# Добавляем корень проекта в путь
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import fire  # noqa: E402

from emotion_recognition.inference.infer import infer_cli  # noqa: E402
from emotion_recognition.training.export_onnx import export_onnx  # noqa: E402
from emotion_recognition.training.train import train  # noqa: E402


def main():
    """Главная функция для CLI команд."""
    fire.Fire(
        {
            "train": train,
            "infer": infer_cli,
            "export_onnx": export_onnx,
        }
    )


if __name__ == "__main__":
    main()
