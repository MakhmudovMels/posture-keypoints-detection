import streamlit as st
import cv2
from ultralytics import YOLO

# Указываем какие пары точек будем соединять между собой
POINT_PAIRS = [
    (0, 1),
    (1, 2),
    (2, 3)
]
# Пороговое значение оценки осанки
THRECHOLD = 1 / 6


def main():
    st.set_page_config(
        page_title="Трекер осанки",
        )
    st.title("Трекер осанки")

    # Загружаем предобученную модель
    model = YOLO('best.pt')

    on = st.toggle("Запуск")
    # Создаем пустой элемент для отображения кадров
    FRAME_WINDOW = st.image([])
    # Создаем пустой элемент для отображения сообщений
    placeholder = st.empty()
    # Подключаем камеру
    camera = cv2.VideoCapture(0)

    while on:
        # Считываем изображение с камеры
        _, frame = camera.read()
        # Инференс модели
        results = model.predict(frame, iou=0.4, conf=0.5)

        for i in range(len(results[0].keypoints.xy)):
            # Массив ключевых точек
            keypoints = results[0].keypoints.xy[i]
            is_correct_posture = True
            
            # Проверяем ровная ли осанка по ключевым точкам
            if len(keypoints) == 4: # Если нашлись все точки
                length_spine = abs(keypoints[3][1] - keypoints[0][1]) # Высота позвоночника
                for i in range(len(keypoints) - 1):
                    for j in range(i + 1, len(keypoints)):
                        x1 = keypoints[i][0]
                        x2 = keypoints[j][0]
                        if abs(x2 - x1) > THRECHOLD * length_spine:
                            is_correct_posture = False
                            break
            
                if is_correct_posture:
                    # Устанавливаем зелёный цвет
                    color = (0, 255, 0)
                    placeholder.success('Отличная осанка')
                else:
                    # Устанавливаем красный цвет
                    color = (0, 0, 255)
                    placeholder.error('Выпрями спину')

            # Отображаем точки
            for j, point in enumerate(keypoints):
                x, y = point
                if x >= 0 and y >= 0:
                    cv2.circle(frame, (int(x), int(y)), 5, color, -1)

            # Соединяем точки линиями
            if len(keypoints) > 0:
                for pair in POINT_PAIRS:
                    start, end = pair
                    if keypoints[start][0] > 0 and keypoints[start][1] > 0 and keypoints[end][0] > 0 and keypoints[end][1] > 0:  # Проверяем, что обе пары координат больше нуля
                        x1, y1 = int(keypoints[start][0]), int(keypoints[start][1])
                        x2, y2 = int(keypoints[end][0]), int(keypoints[end][1])
                        cv2.line(frame, (x1, y1), (x2, y2), color, 3)
        
        # Преобразование из BGR в RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Отображаем кадр
        FRAME_WINDOW.image(frame)


if __name__ == '__main__':
    main()