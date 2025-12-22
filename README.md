## Распознавание эмоций по аудиофайлам

**Автор:** Карпов Степан Константинович

## Постановка задачи

Определение эмоции говорящего по коротким записям речи (WAV). Применение: анализ звонков, чат-боты, исследования эмоций, образование, медицина.

## Формат данных

**Вход**

- WAV до 10 секунд, моно, 16 kHz
- Для обучения: пары `(аудио, метка эмоции)`

**Выход**

- Класс эмоции из фиксированного набора (например: happy, sad, angry, neutral, …)
- Вероятностное распределение по классам

## Метрики

- Accuracy (основная)
- F1-score (макро)
- Confusion matrix (для анализа ошибок)

Ожидание: baseline accuracy ~0.2, целевая >0.5. Accuracy интуитивна, F1 отражает баланс при несбалансированных классах.

## Валидация и тест

- Разбиение train/val/test: ~70/15/15 или 80/10/10
- Фиксированный random seed (np.random.seed) для воспроизводимости
- Исключить утечки: один файл только в одной выборке

## Датасеты

- Основной: RAVDESS — 1440 файлов, 24 актёра, 8 эмоций (~60 МБ, WAV)
  Особенности: качественная разметка, баланс классов, стандартная длина записей.
  Риски: разные голоса/пол, шум, вариативность исполнения эмоций.
- Опционально: CREMA-D или Emotion Speech Dataset (Kaggle)

## Моделирование

**Бейзлайн**

- Предсказание наиболее частой эмоции → accuracy ~0.1–0.2
- Альтернатива: MLP на MFCC (13–40 коэф.), 2-слойная сеть

**Основная модель**

- CNN на мел-спектрограммах
- Альтернатива: 1D-CNN или предобученные энкодеры (Wav2Vec, AST)
- Фреймворк: PyTorch Lightning
- Обучение: cross-entropy, Adam, ранняя остановка по валидации

## Внедрение

- Скрипт для inference: ввод WAV, выход — метка эмоции
- Экспорт в ONNX для продакшена
- MLflow Serving для развертывания модели
- CLI интерфейс для удобного использования

---

## Setup

### Требования

- Python 3.9 или выше
- Poetry (для управления зависимостями)

### Установка Poetry

Если Poetry еще не установлен:

```bash
pip3 install poetry
```

### Настройка проекта

1. Клонируйте репозиторий:

```bash
git clone <repository-url>
cd emotion-recognition
```

2. Установите зависимости:

```bash
poetry install
```

3. Установите pre-commit хуки:

```bash
poetry run pre-commit install
```

4. Проверьте качество кода:

```bash
poetry run pre-commit run -a
```

5. Инициализируйте DVC (для управления данными):

```bash
poetry run dvc init
```

### Настройка MLflow

Перед обучением убедитесь, что MLflow сервер запущен:

```bash
mlflow ui --host 127.0.0.1 --port 8085
```

Или используйте встроенный сервер MLflow (он запускается автоматически при первом логировании).

---

## Train

### Подготовка данных

Данные будут автоматически загружены при первом запуске обучения через DVC или функцию `download_data()`.

Если нужно загрузить данные вручную:

```bash
poetry run python -m emotion_recognition.data.download_data
```

Для добавления в dvc:

```bash
poetry run dvc add data/raw
```

### Запуск обучения

Обучение модели запускается командой:

```bash
poetry run python commands.py train
```

Команда выполняет следующие шаги:

1. Разделяет данные на train/val/test с фиксированным seed
2. Создает DataLoaders для каждого split
3. Инициализирует модель CNN
4. Запускает обучение с PyTorch Lightning
5. Логирует метрики в MLflow (loss, accuracy, F1-score)
6. Сохраняет лучшую модель в `models/`
7. Экспортирует модель в MLflow для дальнейшего использования

### Параметры обучения

Все гиперпараметры настраиваются через конфиги Hydra в директории `configs/`:

- `configs/data/preprocessing.yaml` - параметры препроцессинга
- `configs/model/cnn.yaml` - архитектура модели
- `configs/training/train.yaml` - параметры обучения

Для переопределения параметров можно использовать:

```bash
poetry run python commands.py train training.batch_size=64 training.learning_rate=0.0005
```

### Результаты обучения

После обучения:

- Модель сохраняется в `models/best_model-*.ckpt`
- Метрики логируются в MLflow (доступны по адресу http://127.0.0.1:8085)
- Графики метрик можно создать командой:

```bash
poetry run python -m emotion_recognition.training.plot_metrics
```

Графики сохраняются в директорию `plots/`.

---

## Production preparation

### Экспорт в ONNX

После обучения модель можно экспортировать в ONNX формат для оптимизированного инференса:

```bash
poetry run python commands.py export_onnx
```

Экспортированная модель сохраняется в `models/model.onnx`.

### Комплектация поставки

Для запуска inference в продакшене необходимы следующие артефакты:

1. **Модель**: `models/best_model-*.ckpt` или `models/model.onnx`
2. **Конфигурация**: файлы из `configs/` (особенно `configs/data/preprocessing.yaml` для параметров препроцессинга)
3. **Зависимости**: минимальный набор из `pyproject.toml`:
   - torch
   - librosa
   - numpy
   - onnxruntime (если используется ONNX модель)

### MLflow Model Registry

Модель автоматически регистрируется в MLflow Model Registry при обучении. Для использования зарегистрированной модели:

```python
import mlflow.pyfunc

model = mlflow.pyfunc.load_model("models:/emotion_cnn/Production")
```

---

## Infer

### Запуск инференса

Для предсказания эмоции на аудио файле:

```bash
poetry run python commands.py infer <path_to_audio.wav>
```

Например:

```bash
poetry run python commands.py infer data/raw/Actor_01/03-01-01-01-01-01-01.wav
```

Команда выводит:

- Предсказанную эмоцию
- Распределение вероятностей по всем классам эмоций

### Формат входных данных

- Формат: WAV файл
- Частота дискретизации: 16 kHz (автоматически ресемплируется)
- Каналы: моно
- Длительность: до 10 секунд (автоматически обрезается/дополняется)

### Пример данных

Примеры аудио файлов можно найти в датасете RAVDESS после загрузки данных в `data/raw/RAVDESS/`.

---

## Структура проекта

```
emotion-recognition/
├── emotion_recognition/          # Основной пакет
│   ├── data/                      # Работа с данными
│   │   ├── dataset.py            # Dataset класс
│   │   └── download_data.py      # Загрузка данных
│   ├── models/                    # Модели
│   │   └── cnn.py                # CNN модель
│   ├── training/                  # Обучение
│   │   ├── train.py              # Скрипт обучения
│   │   ├── export_onnx.py        # Экспорт в ONNX
│   │   └── plot_metrics.py       # Создание графиков
│   └── inference/                 # Инференс
│       └── infer.py               # Скрипт инференса
├── configs/                       # Конфиги Hydra
│   ├── config.yaml               # Главный конфиг
│   ├── data/
│   ├── model/
│   ├── training/
│   └── inference/
├── plots/                         # Графики метрик
├── models/                        # Сохраненные модели
├── data/                          # Данные (не в git)
├── commands.py                    # CLI точка входа
├── pyproject.toml                 # Зависимости Poetry
├── .pre-commit-config.yaml       # Конфиг pre-commit
└── README.md                      # Этот файл
```
