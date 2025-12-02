import os
import cv2
from tqdm import tqdm

def check_and_fix_annotations(label_dir, img_dir):
    filenames = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    total_files = len(filenames)

    for filename in tqdm(filenames, desc="Processing annotations", total=total_files):
        with open(os.path.join(label_dir, filename), 'r') as file:
            lines = file.readlines()

        # Получаем размеры изображения
        img_filename = filename.replace('.txt', '.jpg')  # Предполагаем, что изображения имеют формат .jpg
        img_path = os.path.join(img_dir, img_filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Image not found: {img_path}")
            continue
        height, width, _ = img.shape

        with open(os.path.join(label_dir, filename), 'w') as file:
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    print(f"Invalid annotation in {filename}: {line}")
                    continue

                try:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    box_width = float(parts[3])
                    box_height = float(parts[4])

                    # Проверка и исправление значений
                    if x_center < 0:
                        x_center = 0
                    elif x_center > 1:
                        x_center = 1

                    if y_center < 0:
                        y_center = 0
                    elif y_center > 1:
                        y_center = 1

                    if box_width < 0:
                        box_width = 0
                    elif box_width > 1:
                        box_width = 1

                    if box_height < 0:
                        box_height = 0
                    elif box_height > 1:
                        box_height = 1

                    # Записываем исправленные аннотации в файл
                    file.write(f"{class_id} {x_center} {y_center} {box_width} {box_height}\n")
                except ValueError:
                    print(f"Invalid annotation format in {filename}: {line}")

# Путь к директории с аннотациями
label_dir = 'D:\\plate_dataset\\valid\\labels'
# Путь к директории с изображениями
img_dir = 'D:\\plate_dataset\\valid\\images'

# Проверка и исправление аннотаций
check_and_fix_annotations(label_dir, img_dir)

# Повторяем для тренировочной директории
label_dir = 'D:\\plate_dataset\\train\\labels'
img_dir = 'D:\\plate_dataset\\train\\images'

check_and_fix_annotations(label_dir, img_dir)