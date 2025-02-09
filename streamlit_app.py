import streamlit as st
import cv2
import requests
import numpy as np

# Конфигурация API
API_URL = "http://localhost:8000/process-image"

st.set_page_config(page_title="Трекер осанки")
st.title("Трекер осанки")

def process_frame(frame):
    # Конвертация кадра в bytes
    _, img_encoded = cv2.imencode('.jpg', frame)
    files = {'file': ('image.jpg', img_encoded.tobytes(), 'image/jpeg')}
    
    try:
        response = requests.post(API_URL, files=files)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Ошибка подключения к API: {str(e)}")
        return None

def main():
    on = st.toggle("Запуск")
    FRAME_WINDOW = st.image([])
    placeholder = st.empty()
    camera = cv2.VideoCapture(0)

    while on:
        _, frame = camera.read()
        
        if frame is None:
            continue

        # Отправка кадра на обработку
        result = process_frame(frame)
        
        if result:
            # Отрисовка результатов
            color = tuple(result['color'])
            keypoints = result['keypoints']
            
            # Рисуем точки
            for point in keypoints:
                x, y = point
                cv2.circle(frame, (int(x), int(y)), 5, color, -1)
            
            # Рисуем линии
            for (start, end) in [(0,1), (1,2), (2,3)]:
                if start < len(keypoints) and end < len(keypoints):
                    x1, y1 = int(keypoints[start][0]), int(keypoints[start][1])
                    x2, y2 = int(keypoints[end][0]), int(keypoints[end][1])
                    cv2.line(frame, (x1, y1), (x2, y2), color, 3)
            
            # Отображаем сообщение
            if result['is_correct']:
                placeholder.success(result['message'])
            else:
                placeholder.error(result['message'])

        # Конвертация и отображение кадра
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)

    camera.release()

if __name__ == '__main__':
    main()