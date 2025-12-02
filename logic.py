import numpy as np

import math

# (Примечание: вам нужно добавить 'import math' в main.py для функции match_skeleton)



def calculate_center(bbox):

    x1, y1, x2, y2 = bbox

    center_x = int((x1 + x2) / 2)

    center_y = int((y1 + y2) / 2)

    return (center_x, center_y)



def is_hands_up(skeleton, height, width):

    """ Проверяет, подняты ли руки (например, для сигнала SOS). """

    if not skeleton:

        return False

       

    kps = skeleton['keypoints']

   

    # 0: Нос, 5: Левое Плечо, 6: Правое Плечо, 9: Левое Запястье, 10: Правое Запястье

   

    # Предполагаем, что точки нормализованы (0-1) и переводим в пиксели

    try:

        y_nose = kps[0][1] * height

        y_l_wrist = kps[9][1] * height

        y_r_wrist = kps[10][1] * height

        y_shoulders = (kps[5][1] + kps[6][1]) / 2 * height



        # Если оба запястья находятся ВЫШЕ плеч (меньше Y-координата = выше)

        if y_l_wrist < y_shoulders and y_r_wrist < y_shoulders and y_l_wrist < y_nose:

            return True

        return False

    except IndexError:

        return False



def analyze_worker_activity(worker_id, bbox, history_data, train_bbox=None, skeleton_data=None):

    """

    Главная логика: Падение, Простой, Руки Вверх.

    """

    x1, y1, x2, y2 = bbox

    width = x2 - x1

    height = y2 - y1

   

    # --- 1. АНАЛИЗ ДЕЙСТВИЙ (Требует ФЕДИ) ---

    if skeleton_data:

        # ПРИМЕР: Проверка на SOS/Помощь

        # (Нам нужны размеры видео, чтобы перевести нормализованные точки в пиксели)

        # В данном случае, передадим их из main.py

        if is_hands_up(skeleton_data, 720, 1280): # Здесь нужны реальные размеры видео!

            return "SOS_ALERT"



    # --- 2. Проверка на падение (Базовая) ---

    if width > height * 1.4:

        return "FALL_DETECTED"



    # --- 3. Проверка на простой (IDLE) ---

    # ... (старый код проверки скорости) ...



    return "WORKING"