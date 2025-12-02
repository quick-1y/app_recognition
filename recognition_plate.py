"""/recognition.py"""
import cv2
import easyocr
import yaml
import re
import logging

logging.basicConfig(level=logging.DEBUG)

class PlateRecognizer:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=True)
        self.load_patterns()

    def load_patterns(self):
        with open('configs/plate_patterns.yaml', 'r') as file:
            config = yaml.safe_load(file)
            self.patterns = config['plate_patterns']

    def recognize_plate(self, img_gray):
        """Распознавание номера на уже обработанном изображении."""
        try:
            # Распознавание текста
            recognized_text = self.recognize_text(img_gray)
            logging.debug(f"Recognized text: {recognized_text}")

            # Фильтрация по шаблону
            filtered_text = self.filter_by_pattern(recognized_text)

            return filtered_text
        except Exception as e:
            logging.error(f"Error during plate recognition: {e}")
            return ""

    def filter_by_pattern(self, text):
        # Преобразуем текст в верхний регистр
        text = text.upper()

        for pattern in self.patterns:
            regex = re.compile(pattern['pattern'])
            if regex.match(text):
                logging.debug(f"Matched pattern: {text}")
                # Добавляем регион к распознанному номеру
                return f"{text} {pattern['region']}"
        logging.debug(f"No pattern matched: {text}")
        return ""

    def preprocess_image(self, img):
        cv2.imwrite("debug_original_image.png", img)

        # Увеличение размера изображения
        img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

        # Преобразование изображения в оттенки серого
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Улучшение контраста
        img_gray = cv2.convertScaleAbs(img_gray, alpha=1.5, beta=0)

        # Удаление шума
        img_gray = cv2.bilateralFilter(img_gray, 9, 75, 75)

        # Отладка: сохранение промежуточного изображения
        cv2.imwrite("debug_preprocessed_image.png", img_gray)

        return img_gray

    def recognize_text(self, img_gray):
        result = self.reader.readtext(img_gray)
        recognized_text = ''.join([text for (_, text, _) in result])
        logging.debug(f"Recognized text: {recognized_text}")
        return recognized_text