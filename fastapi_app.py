import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from pydantic import BaseModel

app = FastAPI()

# Конфигурация модели
model = YOLO('models/best.pt')
POINT_PAIRS = [(0, 1), (1, 2), (2, 3)]
THRESHOLD = 1/6

class ProcessResult(BaseModel):
    keypoints: list
    is_correct: bool
    color: tuple
    message: str

@app.post("/process-image", response_model=ProcessResult)
async def process_image(file: UploadFile = File(...)):
    try:
        # Чтение и декодирование изображения
        image_data = await file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Обработка кадра моделью
        results = model.predict(frame, iou=0.4, conf=0.5)
        response_data = None
        
        for i in range(len(results[0].keypoints.xy)):
            keypoints = results[0].keypoints.xy[i].tolist()
            is_correct_posture = True
        
            if len(keypoints) == 4:
                length_spine = abs(keypoints[3][1] - keypoints[0][1])
                for i in range(len(keypoints)-1):
                    for j in range(i+1, len(keypoints)):
                        if abs(keypoints[j][0] - keypoints[i][0]) > THRESHOLD * length_spine:
                            is_correct_posture = False
                            break
            
            color = (0, 255, 0) if is_correct_posture else (0, 0, 255)
            message = 'Отличная осанка' if is_correct_posture else 'Выпрями спину'
            
            response_data = {
                "keypoints": keypoints,
                "is_correct": is_correct_posture,
                "color": color,
                "message": message
            }
        
        return JSONResponse(content=response_data)
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error processing image: {str(e)}"}
        )