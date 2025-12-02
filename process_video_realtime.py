"""/process_video_realtime.py"""
import torch
import cv2
import os
from ultralytics import YOLO
from PyQt5.QtGui import QImage
from recognition_plate import PlateRecognizer
import time
import logging
from collections import defaultdict
import numpy as np
import yaml

logging.basicConfig(level=logging.DEBUG)

# Загрузка конфигурации из файла
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Путь к дообученной модели для детекции типа транспортного средства и номеров
pretrained_model_path = 'models/best.pt'

# Загрузка дообученной модели
model = YOLO(pretrained_model_path)

# Инициализация PlateRecognizer
plate_recognizer = PlateRecognizer()

# Словарь с цветами для каждого класса
class_colors = {
    'licence': (255, 255, 255),    # Зеленый
}

# Словарь для хранения истории треков объектов
track_history = defaultdict(lambda: [])

# Словарь для хранения последней позиции номера
last_plate_position = {}

# Интервал для отправки изображения номера (каждый 10-й кадр)
plate_image_send_interval = config['plate_image_send_interval']

# Последний распознанный номер
last_recognized_plate = ""

# Функция для обработки видео в реальном времени
def process_video_realtime(video_path, frame_callback, text_callback):
    global last_recognized_plate
    try:
        # Открытие видеофайла
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Error: Could not open video file {video_path}")
            return

        # Получение FPS видеофайла
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_delay = 1.0 / fps

        while True:
            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            # Преобразование кадра в формат, подходящий для модели
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model.track(frame_rgb, persist=True)  # Используем трекинг

            # Обработка результатов
            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].astype(int)
                    class_id = int(box.cls[0])
                    confidence = box.conf[0]
                    track_id = int(box.id[0]) if box.id is not None else None  # Получаем идентификатор трека

                    # Получение имени класса
                    class_name = model.names[class_id]

                    # Выбор цвета для рамки по классу
                    color = class_colors.get(class_name, (0, 255, 0))  # Зеленый по умолчанию

                    # Рисование рамки и текста на кадре
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Если обнаружен номерной знак, распознаем текст на нем
                    if class_name == 'licence':
                        plate_image = frame[y1:y2, x1:x2]
                        preprocessed_image = plate_recognizer.preprocess_image(plate_image)
                        recognized_text = plate_recognizer.recognize_plate(preprocessed_image)

                        # Проверка на изменение распознанного номера
                        if recognized_text and recognized_text != last_recognized_plate:
                            text_callback(recognized_text)
                            last_recognized_plate = recognized_text

                        cv2.putText(frame, recognized_text, (x1, y1 - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                        # Добавляем трекинг для номеров
                        if track_id is not None:
                            track = track_history[track_id]
                            current_position = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                            track.append(current_position)  # добавление координат центра объекта в историю
                            if len(track) > 30:  # ограничение длины истории до 30 кадров
                                track.pop(0)

                            # Рисование линий трека
                            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                            cv2.polylines(frame, [points], isClosed=False, color=color, thickness=2)

                            # Проверка на изменение позиции номера
                            if track_id not in last_plate_position or last_plate_position[track_id] != current_position:
                                last_plate_position[track_id] = current_position

            # Преобразование кадра в формат QImage
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)

            # Вызов callback-функции для обновления изображения в GUI
            frame_callback(q_img)

            # Синхронизация с FPS
            elapsed_time = time.time() - start_time
            if elapsed_time < frame_delay:
                time.sleep(frame_delay - elapsed_time)

        # Освобождение ресурсов
        cap.release()
    except Exception as e:
        logging.error(f"Error during video processing: {e}")