# Распознавание автомобильных номеров

Полностью обновлённый пайплайн для детекции и OCR автомобильных номеров на основе YOLO и EasyOCR с настраиваемым использованием GPU/CPU и локальной SQLite-базой для хранения событий и списков.

## Основные изменения
- **Новая конфигурация** (`config.yaml`) с секциями для модели, OCR, обработки, базы данных и путей к видео. Переключение GPU/CPU для OCR вынесено в параметр `ocr.gpu`.
- **Улучшенный OCR**: агрессивная предобработка (CLAHE, шарпинг, билатеральный фильтр, адаптивный порог), фильтрация по паттернам и приоритизация уверенных кандидатов.
- **Локальная база** `SQLite` (`data/plates.db`) для хранения распознанных номеров и пользовательских списков (watch-листы).
- **Упрощённая GUI**: загрузка настроек из конфига, отображение истории событий и списков из базы, управление путями к каналам.

## Быстрый старт
1. Установите зависимости проекта (PyQt5, ultralytics, easyocr, opencv-python и др.).
2. Отредактируйте `config.yaml` под свои камеры/файлы и параметры OCR/YOLO.
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
  languages: ["en"]
  gpu: false
  min_confidence: 0.35
  resize_factor: 2.5
  contrast_alpha: 1.6
  contrast_beta: 0
  denoise_diameter: 7
  threshold_block_size: 25
  threshold_c: 7
  max_candidates: 5

database:
  path: data/plates.db
  vacuum_on_start: false
  retention_days: 30
```

- **Переключение GPU**: измените `ocr.gpu` на `true`, чтобы использовать GPU в EasyOCR.
- **Обновление путей к видео**: укажите файлы/потоки в `app.video_paths` или через вкладку «Настройки» в GUI.

## База данных
База создаётся автоматически в `data/plates.db` и содержит:
- `plates` — распознанные события (номер, регион, confidence, источник, время).
- `lists` и `list_items` — пользовательские списки номеров.

## Отладка OCR
Для улучшения качества:
- увеличьте `ocr.resize_factor` или `contrast_alpha` для тёмных кадров,
- уменьшите `ocr.denoise_diameter`, если буквы «размываются»,
- скорректируйте регулярные выражения в `configs/plate_patterns.yaml` под нужные регионы.
