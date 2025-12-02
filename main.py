import cv2
import pandas as pd
import json
import os
import time
from logic import analyze_worker_activity

# --- НАСТРОЙКИ ---
VIDEO_PATH = "input_video.mov"
JSON_PATH = "final_report.json"
OUTPUT_CSV = "hackathon_result.csv"
OUTPUT_VIDEO = "result_video.avi"

# 🔥 НАСТРОЙКА СИНХРОНИЗАЦИИ 🔥
# 1. СКОРОСТЬ ДАННЫХ (Файл Саши ускорен)
DATA_SPEED_FACTOR = 0.33333  # <-- ТВОЙ КОЭФФИЦИЕНТ (1/3)
# 2. СДВИГ (Тот самый, из калибровки)
SYNC_OFFSET = 0  

# --- ЦВЕТА ---
ROLE_COLORS = {
    "worker": (0, 255, 0),       # Зеленый
    "janitor": (255, 255, 0),    # Желтый
    "manager": (0, 0, 255),      # Красный
    "darkmechanic": (128, 0, 128),
    "lightmechanic": (255, 192, 203),
    "signalman": (0, 165, 255),
    "train": (128, 128, 128)
}

KNOWN_TRAINS = { 82: "No.4521", 95: "No.77-B" }

print(f"🚀 ЗАПУСК ФИНАЛЬНОГО РЕНДЕРА! (Скорость x{DATA_SPEED_FACTOR}, Сдвиг {SYNC_OFFSET})")

# --- 1. ЗАГРУЗКА ---
if not os.path.exists(JSON_PATH):
    print("❌ Файл JSON не найден!")
    exit()

with open(JSON_PATH, 'r') as f:
    sasha_data = json.load(f)

detections_by_frame = {}
for item in sasha_data:
    f_num = int(item['frame'])
    if f_num not in detections_by_frame: detections_by_frame[f_num] = []
    detections_by_frame[f_num].append(item)

start_data_frame = min(detections_by_frame.keys())
end_data_frame = max(detections_by_frame.keys())
print(f"📊 Данные: {start_data_frame}-{end_data_frame}")

# --- 2. ВИДЕО ---
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"❌ Ошибка видео: {VIDEO_PATH}")
    exit()

video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30
total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (video_w, video_h))

# --- 3. СИНХРОНИЗАЦИЯ ---
OFFSET_MODE = False
frame_counter = 0

expected_end_frame = int(total_video_frames * DATA_SPEED_FACTOR)
if expected_end_frame < start_data_frame:
    print("⚠️ Видео слишком короткое для этих данных. Режим наложения.")
    OFFSET_MODE = True
else:
    # Перематываем видео к началу данных
    start_video_frame = int(start_data_frame / DATA_SPEED_FACTOR)
    print(f"⏩ Перемотка видео на кадр {start_video_frame}...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_video_frame)
    frame_counter = start_video_frame - 1

# --- 4. ЦИКЛ ---
history_storage = {}
all_events = []

print("⏳ Обработка началась...")

while True:
    ret, frame_img = cap.read()
    if not ret: break

    if OFFSET_MODE:
        current_video_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        # Здесь нет умножения на DATA_SPEED_FACTOR, так как start_data_frame уже учитывает его
        target_frame = start_data_frame + current_video_pos + SYNC_OFFSET
    else:
        frame_counter += 1
        # ГЛАВНАЯ ФОРМУЛА: Умножаем текущий счетчик видео на фактор
        target_frame = int(frame_counter * DATA_SPEED_FACTOR) + SYNC_OFFSET

    if target_frame > end_data_frame:
        print("\n🏁 Данные закончились.")
        break

    if frame_counter % 50 == 0:
        print(f"   Видео: {frame_counter} -> Данные: {target_frame}", end="\r")

    if target_frame in detections_by_frame:
        people = detections_by_frame[target_frame]
        
        # 1. ПОЕЗД
        current_train_bbox = None
        for obj in people:
            if obj.get('role') == 'train':
                r = obj['bbox']
                cx, cy, w, h = r[0]*video_w, r[1]*video_h, r[2]*video_w, r[3]*video_h
                x1, y1 = int(cx - w/2), int(cy - h/2)
                x2, y2 = int(cx + w/2), int(cy + h/2)
                
                current_train_bbox = [x1, y1, x2, y2]
                
                p_id = obj.get('id', 0)
                name = KNOWN_TRAINS.get(p_id, f"TRAIN {p_id}")
                
                cv2.rectangle(frame_img, (x1, y1), (x2, y2), (100, 100, 100), 2)
                cv2.putText(frame_img, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                all_events.append({"time_sec": round(target_frame/fps, 2), "frame": target_frame, "id": name, "role": "train", "status": "REPAIR_OBJECT", "is_danger": 0})

        # 2. ЛЮДИ
        for person in people:
            base_role = person.get('role', 'worker')
            if base_role == 'train': continue

            worker_type = person.get('class', person.get('class_name', base_role))
            p_id = person.get('id', 0)
            
            r = person['bbox']
            cx, cy, w, h = r[0]*video_w, r[1]*video_h, r[2]*video_w, r[3]*video_h
            x1, y1 = int(cx - w/2), int(cy - h/2)
            x2, y2 = int(cx + w/2), int(cy + h/2)
            
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(video_w, x2)
            y2 = min(video_h, y2)

            try:
                status = analyze_worker_activity(p_id, [x1, y1, x2, y2], history_storage)
            except:
                status = "WORKING"
            
            box_color = ROLE_COLORS.get(worker_type, (0, 255, 0))
            is_danger = 0
            
            if status == "FALL_DETECTED":
                box_color = (0, 0, 255)
                is_danger = 1
            elif status == "IDLE":
                box_color = (0, 255, 255)

            cv2.rectangle(frame_img, (x1, y1), (x2, y2), box_color, 2)
            label = f"{p_id} | {worker_type.upper()} | {status}"
            cv2.rectangle(frame_img, (x1, y1-20), (x1+250, y1), box_color, -1)
            cv2.putText(frame_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

            all_events.append({"time_sec": round(target_frame/fps, 2), "frame": target_frame, "id": p_id, "role": worker_type, "status": status, "is_danger": is_danger})

    out.write(frame_img)

cap.release()
out.release()

if all_events:
    try:
        df = pd.DataFrame(all_events)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n✅ Таблица готова!")
    except:
        df.to_csv(f"result_{int(time.time())}.csv", index=False)

print(f"🎥 Видео готово: {OUTPUT_VIDEO}")