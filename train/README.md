# Обучение моделей

В каталоге `train` собраны упрощённые пайплайны для обучения детектора номеров (YOLO) и OCR‑распознавания. Оба тренера настраиваются через YAML‑файлы в `train/train_config` и имеют CLI‑обёртки в `train/scripts`.

## Детектор YOLO

Тренер YOLO использует API Ultralytics и ожидает YAML с описанием разбиения датасета на train/val/test. Готовый файл `train/train_config/detection_dataset.yaml` предполагает расположение изображений в `data/detection` с подпапками `images/train`, `images/val` и `images/test`.

Запуск обучения:

```bash
python -m train.scripts.train_yolo --config train/train_config/yolo_train_config.yaml
```

Основные параметры задаются в конфиге (чекпоинт модели, batch size, размер изображений, patience и т.д.). При необходимости можно указать другой YAML датасета или переопределить значения через CLI‑аргументы, поддерживаемые тренером.

## OCR‑распознавание

OCR‑пайплайн обучает компактную модель CRNN с функцией потерь CTC. Ожидается файл `labels.csv` с строками формата `image_path,label` (пути относительно `dataset_root` в конфиге) и одноканальные изображения нужной высоты/ширины.

Запуск OCR‑обучения:

```bash
python -m train.scripts.train_ocr --config train/train_config/ocr_config.yaml
```

В конфиге OCR задаются алфавит, размер изображений, batch size, learning rate и чекпоинтинг. Чекпоинты сохраняются в каталог `checkpoint_dir`, а возобновить обучение можно, указав в `resume_from` путь к сохранённому `.pt` файлу.

## Доступные конфигурации

- `train/train_config/yolo_train_config.yaml` — основные настройки обучения YOLO.
- `train/train_config/detection_dataset.yaml` — описание датасета для детектора.
- `train/train_config/ocr_config.yaml` — настройки обучения OCR.

При необходимости адаптируйте эти файлы под свои датасеты, оборудование и имена экспериментов.
