# Распознавание автомобильных номеров

Полностью обновлённый пайплайн для детекции и OCR автомобильных номеров на основе YOLO и CRNN с настраиваемым использованием GPU/CPU и локальной SQLite-базой для хранения событий и списков.

## Основные изменения
- **Новая конфигурация** (`config.yaml`) с секциями для модели, OCR, обработки, базы данных и путей к видео. Переключение GPU/CPU для OCR вынесено в параметр `ocr.gpu`.
- **OCR на CRNN**: использование обученной CRNN-модели с алфавитом и размером входных изображений, настроенными через конфиг.
- **Локальная база** `SQLite` (`data/plates.db`) для хранения распознанных номеров и пользовательских списков (watch-листы).
- **Упрощённая GUI**: загрузка настроек из конфига, отображение истории событий и списков из базы, управление путями к каналам.

## Быстрый старт
1. Установите зависимости проекта (PyQt5, ultralytics, opencv-python, torch и др.) вместе с весами CRNN.
2. Отредактируйте `config.yaml` под свои камеры/файлы и параметры OCR/YOLO. Укажите путь к весам `ocr.crnn_weights` и при необходимости измените алфавит и размер входных изображений.
3. Запустите GUI:
   ```bash
   python run.py
   ```
## Конфигурация
Пример `config.yaml`:
```yaml
app:
  plate_patterns_path: configs/plate_patterns.yaml
  video_paths:
    - sample_videos/demo.mp4

processing:
  plate_image_send_interval: 20
  tracking_history: 32
  draw_tracks: true

model:
  detector_weights: models/best.pt
  confidence_threshold: 0.35
  iou_threshold: 0.45

ocr:
  gpu: false
  min_confidence: 0.35
  crnn_weights: models/crnn.pth
  alphabet: "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
  crnn_img_height: 32
  crnn_img_width: 128

database:
  path: data/plates.db
  vacuum_on_start: false
  retention_days: 30
```

- **CRNN OCR**: используйте обученные веса в `ocr.crnn_weights`, при необходимости скорректируйте `ocr.alphabet` и размеры входа.
- **Переключение GPU**: измените `ocr.gpu` на `true`, чтобы использовать GPU в CRNN (при наличии).
- **Обновление путей к видео**: укажите файлы/потоки в `app.video_paths` или через вкладку «Настройки» в GUI.

## База данных
База создаётся автоматически в `data/plates.db` и содержит:
- `plates` — распознанные события (номер, регион, confidence, источник, время).
- `lists` и `list_items` — пользовательские списки номеров.

## Отладка OCR
Для улучшения качества:
- используйте более качественные кадры детектора (YOLO) и убедитесь, что номерная область хорошо читаема,
- корректируйте регулярные выражения в `configs/plate_patterns.yaml` под нужные регионы и формат номеров.

## Используйте датасеты для обучения моделей
- для OCR
- `https://www.kaggle.com/datasets/evgrafovmaxim/nomeroff-russian-license-plates?resource=download-directory`
- `https://drive.google.com/file/d/1X2cBcaZVw2gamlGW3jVqR4I_pPd7rsFu/view?usp=drive_link`
- для yolo
- `https://huggingface.co/datasets/AY000554/Car_plate_detecting_dataset`
- `https://drive.google.com/file/d/1wEdXcFBfgdGPZg2HbS-ugSwbOXEZPtoW/view?usp=drive_link`
