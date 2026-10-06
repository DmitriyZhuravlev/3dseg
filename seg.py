import cv2
import numpy as np
import matplotlib.pyplot as plt

def fill_convex_gaps(image_path):
    # Загрузка изображения в градациях серого
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Бинаризация изображения
    _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    
    # Поиск контуров
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Создание пустого холста для рисования
    mask = np.zeros_like(binary)
    color_result = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    for contour in contours:
        # Сглаживание контура
        #contour = cv2.approxPolyDP(contour, 3, True)
        
        # Построение выпуклой оболочки
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)
        
        if defects is not None and len(defects) > 1:
            for i in range(0, len(defects) - 1, 2):  # Обрабатываем только крупные дефекты
                s1, e1, f1, d1 = defects[i, 0]
                s2, e2, f2, d2 = defects[i + 1, 0]
                
                start1 = tuple(contour[s1][0])
                end1 = tuple(contour[e1][0])
                far1 = tuple(contour[f1][0])
                
                start2 = tuple(contour[s2][0])
                end2 = tuple(contour[e2][0])
                far2 = tuple(contour[f2][0])
                
                # Центр эллипса
                center = ((far1[0] + far2[0]) // 2, (far1[1] + far2[1]) // 2)
                
                # Полуоси эллипса
                axes = (abs(start2[0] - far1[0]) // 2, abs(far2[1] - end1[1]) // 2)
                
                # Рисуем эллипс
                cv2.ellipse(binary, center, axes, 0, 0, 360, 255, -1)
                cv2.ellipse(color_result, center, axes, 0, 0, 360, (0, 0, 255), 2)
        
        # Заполняем контур выпуклой оболочки
        cv2.drawContours(mask, [cv2.convexHull(contour)], -1, 255, -1)
    
    return binary, color_result

# Путь к изображению
image_path = "/home/dzhura/ComputerVision/data/img/test/out/object_mask_pig.JPG"

# Применение функции
filled_image, color_image = fill_convex_gaps(image_path)

# Отображение результата
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.imshow(filled_image, cmap='gray')
plt.axis("off")
plt.title("Сегментация с заполнением выпуклых частей эллипсами")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Красные эллипсы на изображении")

plt.show()

