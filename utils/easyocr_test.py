import easyocr
import cv2

# Инициализация Reader с использованием предобученных моделей для английского и русского языков
reader = easyocr.Reader(['en', 'ru'], gpu=True)

# Загрузка изображения
image_path = '../debug_preprocessed_image.png'
image = cv2.imread(image_path)

# Преобразование изображения в оттенки серого
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Распознавание текста
result = reader.readtext(image_gray)

# Вывод результатов
for (bbox, text, prob) in result:
    print(f"Recognized text: {text}, Probability: {prob}")
    # Рисование рамки вокруг распознанного текста
    (top_left, top_right, bottom_right, bottom_left) = bbox
    top_left = tuple(map(int, top_left))
    bottom_right = tuple(map(int, bottom_right))
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
    cv2.putText(image, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# Отображение изображения с распознанным текстом
cv2.imshow("Image with recognized text", image)
cv2.waitKey(0)
cv2.destroyAllWindows()