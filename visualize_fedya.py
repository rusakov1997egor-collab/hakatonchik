import cv2
import json
import numpy as np
import os
import sys

# --- НАСТРОЙКИ ---
VIDEO_PATH = 'input_video.mov'     # Ваше видео
JSON_PATH = 'fedya_pose.json'      # Ваш файл с координатами
OUTPUT_PATH = 'output_fedya.mp4'   # Результат

# Стандартные связи костей (формат COCO - 17 точек). 
# Если у вас MediaPipe (33 точки) или YOLO, скелет может выглядеть странно, 
# но точки все равно отрисуются.
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
]

def draw_skeleton(frame, keypoints):
    """Рисует точки и линии на кадре"""
    # Преобразуем список в numpy массив для удобства
    # Если формат [x, y, conf, x, y, conf...], то берем каждый 3-й элемент
    if len(np.array(keypoints).flatten()) > 50: # Скорее всего формат плоский [x,y,c, x,y,c...]
        points = np.array(keypoints).reshape(-1, 3)[:, :2]
    elif len(np.array(keypoints).shape) == 1: # Плоский [x,y, x,y...]
        points = np.array(keypoints).reshape(-1, 2)
    else:
        points = np.array(keypoints) # Уже формат [[x,y], [x,y]]

    # Рисуем точки
    for i, point in enumerate(points):
        x, y = int(point[0]), int(point[1])
        if x <= 0 and y <= 0: continue 
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

    # Рисуем связи
    for pair in SKELETON_CONNECTIONS:
        idx1, idx2 = pair
        if idx1 >= len(points) or idx2 >= len(points): continue
        
        p1 = points[idx1]
        p2 = points[idx2]
        
        if (p1[0] <= 1 and p1[1] <= 1) or (p2[0] <= 1 and p2[1] <= 1): continue

        pt1 = (int(p1[0]), int(p1[1]))
        pt2 = (int(p2[0]), int(p2[1]))
        cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

def main():
    # 1. ПРОВЕРКА ФАЙЛОВ
    if not os.path.exists(VIDEO_PATH):
        print(f"ОШИБКА: Файл видео не найден: {VIDEO_PATH}")
        return
    if not os.path.exists(JSON_PATH):
        print(f"ОШИБКА: Файл JSON не найден: {JSON_PATH}")
        return

    print("Загрузка JSON...")
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)

    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if width == 0:
        print("ОШИБКА: Не удалось открыть видео. Проверьте кодеки или путь.")
        return

    # Подготовка записи видео
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    print(f"Видео открыто: {width}x{height}, {fps} FPS, {total_frames} кадров.")
    
    # --- ДИАГНОСТИКА СТРУКТУРЫ JSON (чтобы понять, как читать) ---
    sample = data[0] if isinstance(data, list) and len(data) > 0 else data
    print("\n--- ДИАГНОСТИКА: ПРИМЕР ДАННЫХ В JSON ---")
    print(str(sample)[:500] + "...") 
    print("-------------------------------------------\n")

    current_frame = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- ПОПЫТКА ИЗВЛЕЧЬ ДАННЫЕ (самая сложная часть) ---
        try:
            persons_to_draw = []
            
            # ВАРИАНТ 1: JSON это список кадров, внутри список людей (AlphaPose, MMPose)
            if isinstance(data, list) and current_frame < len(data):
                frame_data = data[current_frame]
                # Проверяем, есть ли ключ 'keypoints' напрямую или список людей
                if isinstance(frame_data, dict) and 'keypoints' in frame_data:
                     persons_to_draw = [frame_data['keypoints']]
                elif isinstance(frame_data, list): # Список словарей
                     for p in frame_data:
                         if 'keypoints' in p: persons_to_draw.append(p['keypoints'])
                elif isinstance(frame_data, dict):
                    # Ищем списки внутри (например "candidates" или "people")
                    for key in frame_data:
                        if isinstance(frame_data[key], list):
                             # Эвристика: пробуем найти keypoints внутри
                             for item in frame_data[key]:
                                 if isinstance(item, dict) and 'keypoints' in item:
                                     persons_to_draw.append(item['keypoints'])
                                 elif isinstance(item, list): # иногда сами точки это списки
                                     persons_to_draw.append(item)

            # Отрисовка всех найденных скелетов
            for kps in persons_to_draw:
                draw_skeleton(frame, kps)

        except Exception as e:
            # Не спамим ошибками, выводим только первую
            if current_frame == 0:
                print(f"ПРЕДУПРЕЖДЕНИЕ: Не удалось отрисовать кадр 0. Ошибка: {e}")

        out.write(frame)
        
        # Вывод прогресса каждые 50 кадров
        if current_frame % 50 == 0:
            print(f"Обработано кадров: {current_frame}/{total_frames}", end='\r')
            
        current_frame += 1

    cap.release()
    out.release()
    print(f"\nГотово! Результат сохранен в файл: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()