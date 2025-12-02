import cv2
import json
import numpy as np

# --- НАСТРОЙКИ ---
VIDEO_PATH = 'input_video.mp4'  # Путь к вашему видео
JSON_PATH = 'keypoints.json'    # Путь к вашему JSON
OUTPUT_PATH = 'output_skeleton.mp4' # Куда сохранить результат

# Пары индексов точек, которые нужно соединить линиями (пример для COCO формата)
# Вам нужно узнать карту соединений (skeleton map) для вашей модели.
# Это просто пример: (0-нос, 1-глаз и т.д.)
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4), # Голова
    (5, 7), (7, 9), (6, 8), (8, 10), # Руки
    (11, 13), (13, 15), (12, 14), (14, 16), # Ноги
    (5, 6), (5, 11), (6, 12), (11, 12) # Тело
]

def draw_pose(frame, keypoints):
    """
    Рисует скелет на кадре.
    keypoints: список формата [[x, y], [x, y], ...] или [x, y, conf, x, y, conf...]
    """
    # Если данные плоские [x, y, conf, x, y, conf...], превращаем в список точек
    # (Раскомментируйте следующие две строки, если у вас такой формат)
    # kps = np.array(keypoints)
    # points = kps.reshape(-1, 3)[:, :2] # Берем только x и y, отбрасываем confidence
    
    # Если данные уже [[x,y], [x,y]], то просто:
    points = keypoints 

    # 1. Рисуем точки
    for i, point in enumerate(points):
        x, y = int(point[0]), int(point[1])
        if x == 0 and y == 0: continue # Пропускаем ненайденные точки
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1) # Красные точки

    # 2. Рисуем линии (кости)
    for pair in SKELETON_CONNECTIONS:
        idx1, idx2 = pair
        if idx1 >= len(points) or idx2 >= len(points): continue
        
        p1 = points[idx1]
        p2 = points[idx2]
        
        # Проверяем, что точки валидны (не 0,0)
        if (p1[0] == 0 and p1[1] == 0) or (p2[0] == 0 and p2[1] == 0):
            continue

        pt1 = (int(p1[0]), int(p1[1]))
        pt2 = (int(p2[0]), int(p2[1]))
        cv2.line(frame, pt1, pt2, (0, 255, 0), 2) # Зеленые линии

    return frame

def main():
    # Загрузка JSON
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)
    
    # Загрузка видео
    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Настройка сохранения видео
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    current_frame = 0
    
    print(f"Начинаем обработку {frame_count} кадров...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # --- ЛОГИКА ИЗВЛЕЧЕНИЯ ДАННЫХ ДЛЯ ТЕКУЩЕГО КАДРА ---
        # Здесь нужно адаптировать под структуру вашего JSON.
        # Пример: Если JSON это список объектов, где каждый объект - это кадр:
        if current_frame < len(data):
            # Пример структуры: data[i]['keypoints']
            # Вам нужно поменять ключи ниже на те, что в вашем файле
            try:
                # ВАРИАНТ А: Если в JSON массив кадров
                frame_data = data[current_frame] 
                kps = frame_data.get('keypoints', []) # Получаем список точек
                
                # ВАРИАНТ Б: Если массив людей внутри массива кадров
                # for person in frame_data:
                #    draw_pose(frame, person['keypoints'])
                
                draw_pose(frame, kps)

            except Exception as e:
                print(f"Ошибка парсинга кадра {current_frame}: {e}")

        # Сохраняем кадр
        out.write(frame)
        
        # Отображаем процесс (опционально, можно убрать для скорости)
        cv2.imshow('Pose Estimation', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        current_frame += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Готово! Видео сохранено в {OUTPUT_PATH}")

if __name__ == "__main__":
    main()