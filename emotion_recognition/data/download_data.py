"""Функции для загрузки данных RAVDESS."""

import urllib.request
import zipfile
from pathlib import Path


def download_data(data_dir: str = "data/raw") -> None:
    """
    Скачивает датасет RAVDESS из открытого источника.

    Args:
        data_dir: Директория для сохранения данных
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    # RAVDESS можно скачать с Kaggle
    url = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"

    zip_path = data_path / "RAVDESS.zip"

    if not zip_path.exists():
        print(f"Скачивание RAVDESS в {zip_path}...")
        urllib.request.urlretrieve(url, zip_path)
        print("Скачивание завершено.")

    extract_path = data_path / "RAVDESS"
    if not extract_path.exists():
        print(f"Распаковка в {extract_path}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(data_path)
        print("Распаковка завершена.")

    print(f"Данные готовы в {extract_path}")


if __name__ == "__main__":
    download_data()
