
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 15:16:22 2018
Updated to detect cells with HoughCircles, save as txt, fix left bar height, and prioritize saved files on April 20, 2025
@author: avsthiago
"""

import os
import cv2
import numpy as np
import datetime
import imghdr
from tqdm import tqdm

# RESOLUTIONS [0] 1920x1080 [1] 1280x720
RESOLUTION = 0
CONFIG_FILE = 'config.txt'

IMAGES_PATH = '../original_images/'
DET_PATH = '../annotations/detections/'
OUT_ANNOT = '../annotations/predictions_corrected/'
OUT_IMGS = '../output/labeled_images/'
LEFT_BAR_SIZE = None
SIZE_WINDOW_COMB = None
X_POS_WINDOW_COMB = None
OPTION = 32
RECTANGLE = ((0, 0), (1920, 1080))
COLORS = [
    (255, 0, 0),    # Capped_brood
    (0, 255, 255),  # Unlabeled
    (0, 0, 128),    # Honey
    (255, 0, 255),  # Brood
    (0, 255, 0),    # Capped_honey
    (255, 255, 100),# Other
    (0, 0, 255),    # Pollen
]
LABELS = ["Capped_brood", "Unlabeled", "Honey", "Brood", "Capped_honey", "Other", "Pollen"]
CLASSES_KEY = [49, 50, 51, 52, 53, 54, 55]  # Keys 1-7
REFRESH = False
REFRESH_RECTANGLE = False
RECTANGLE_MODE = False
MOUSE_PRESSED = False
LABELING_MODE = False
ADDING_MODE = False
REMOVING_MODE = False
MOVING_MODE = True
VIEW_MODE = False
AVG_CELL_SIZE = 10
IX, IY = 0, 0
MIN_DIST = 0
LAST_SAVING = ""
MIN_CONFIDENCE = 99950
MOVING_DIRECTION = 'F'

OLD_GAMMA = 0
OLD_BRIGHTNESS = 0
OLD_CONTRAST = 0
GAMMA = 2
BRIGHTNESS = 0
CONTRAST = 0
IM_NAME = ""
POINTS = np.array([])

def load_configs():
    global RESOLUTION
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as file:
            lines = file.readlines()
            try:
                mx_sam = [i for i in lines if 'RESOLUTION' in i][0]
                RESOLUTION = int(mx_sam.split(':')[-1].strip())
            except:
                print('RESOLUTION not found in the config.txt file')

def set_window_size():
    global LEFT_BAR_SIZE, X_POS_WINDOW_COMB, SIZE_WINDOW_COMB
    if RESOLUTION == 0:
        LEFT_BAR_SIZE = (1000, 400)
        SIZE_WINDOW_COMB = (1920, 1080)
        X_POS_WINDOW_COMB = 600
    elif RESOLUTION == 1:
        LEFT_BAR_SIZE = (850, 410)
        SIZE_WINDOW_COMB = (1280, 720)
        X_POS_WINDOW_COMB = 300

def detect_cells(im_name, img):
    try:
        if RESOLUTION == 0:
            img = cv2.resize(img, (1920, 1080))
        else:
            img = cv2.resize(img, (1280, 720))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=4.7,
            minDist=12,
            param1=80,
            param2=10,
            minRadius=6,
            maxRadius=12
        )
        circle_data = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (cx, cy, r) in circles:
                if cx - r > 0 and cy - r > 0 and cx + r < img.shape[1] and cy + r < img.shape[0]:
                    # [x, y, r, class, class_copy, is_modified, is_visible, confidence]
                    circle_data.append([cx, cy, r, 1, 1, 0, 1, 99950])
            if circle_data:
                circle_data = np.array(circle_data, dtype=np.int32)
                txt_filename = os.path.splitext(os.path.basename(im_name))[0] + '.txt'
                output_path = os.path.join(DET_PATH, txt_filename)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w') as f:
                    for circle in circle_data:
                        f.write(f"{circle[0]},{circle[1]},{circle[2]}\n")
                print(f"บันทึก {len(circle_data)} วงกลมไปที่ {output_path}")
            else:
                print(f"ไม่พบวงกลมใน {im_name}")
            return circle_data
        else:
            print(f"HoughCircles ไม่พบวงกลมใน {im_name}")
            return np.array([])
    except Exception as e:
        print(f"การตรวจจับเซลล์ล้มเหลวในภาพ {im_name}: {str(e)}")
        return np.array([])

def draw_circles(img):
    img2 = np.copy(img)
    for c in POINTS:
        if c[6]:
            cv2.circle(img2, (c[0], c[1]), c[2], COLORS[c[4]], 1)
            if not c[5]:
                cv2.circle(
                    img2,
                    (c[0], c[1]),
                    int(max(0.1 * AVG_CELL_SIZE + 1, 1)),
                    (0, 0, 255),
                    -1,
                )
    return img2

def draw_rectangle(img):
    return cv2.rectangle(np.copy(img), *RECTANGLE, (0, 255, 0), 10)

def count_cells_by_class(cl):
    cond = np.where(np.logical_and(POINTS[:, 4] == cl, POINTS[:, 6] == 1))
    return str(len(POINTS[cond]))

def generate_left_bar_image(saving=False):
    left_panel = np.zeros((LEFT_BAR_SIZE[0], LEFT_BAR_SIZE[1], 3), dtype=np.uint8)
    for i, color_lb in enumerate(zip(COLORS, LABELS)):
        color, label = color_lb
        chosen = "[X]" if OPTION == i + 49 else "[ ]" if not saving else ""
        cv2.putText(
            left_panel,
            " ".join([chosen, str(i + 1), ".", label, count_cells_by_class(i)]),
            (15, 40 * (i + 1)),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            color,
            2,
        )
    cv2.putText(
        left_panel,
        "Saved:",
        (15, 40 * 20),
        cv2.FONT_HERSHEY_DUPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        left_panel,
        LAST_SAVING,
        (15, 40 * 21),
        cv2.FONT_HERSHEY_DUPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    if not saving:
        cv2.putText(
            left_panel,
            "[S]. Save",
            (15, 40 * 14),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            left_panel,
            "[N]. Next",
            (15, 40 * 15),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            left_panel,
            "[P]. Previous",
            (15, 40 * 16),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            left_panel,
            "[BS]. Reset",
            (15, 40 * 17),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            left_panel,
            "[ESC]. Quit",
            (15, 40 * 18),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.namedWindow("Options", 0)
        cv2.resizeWindow("Options", LEFT_BAR_SIZE[1], LEFT_BAR_SIZE[0])
        cv2.moveWindow("Options", 0, 0)
        cv2.imshow("Options", left_panel)
    return left_panel

def find_image_names():
    l_images = []
    for path, subdirs, files in os.walk(IMAGES_PATH):
        for name in files:
            full_path = os.path.join(path, name)
            if imghdr.what(full_path) is not None:
                l_images.append(full_path.replace(IMAGES_PATH, ""))
    return l_images

def get_images_detections(folder_im, folder_txt, folder_corrected):
    images = find_image_names()
    im_to_txt = lambda x: os.path.join(
        "/".join(x.split("/")[:-1]), ".".join(x.split("/")[-1].split(".")[:1]) + ".txt"
    )
    files_path = []
    for i in images:
        img_path = os.path.join(folder_im, i)
        det_txt = os.path.join(folder_txt, im_to_txt(i))
        corr_txt = os.path.join(folder_corrected, im_to_txt(i))
        if os.path.isfile(img_path):
            files_path.append([i, img_path, det_txt, corr_txt])
    return sorted(files_path)

def next_image(direction, len_images, index):
    if direction == "F":
        if index + 1 >= len_images:
            return 0
        return index + 1
    else:
        if index - 1 <= -1:
            return len_images - 1
        return index - 1

def change_mode(mode):
    global MOVING_MODE, RECTANGLE_MODE, LABELING_MODE, MOUSE_PRESSED, ADDING_MODE, REMOVING_MODE, VIEW_MODE
    if mode == "VIEW_MODE":
        VIEW_MODE = True
        RECTANGLE_MODE = False
        MOVING_MODE = True
        ADDING_MODE = False
        REMOVING_MODE = False
        LABELING_MODE = False
    else:
        VIEW_MODE = False
        if mode == "RECTANGLE_MODE":
            RECTANGLE_MODE = True
            MOVING_MODE = False
            ADDING_MODE = False
            REMOVING_MODE = False
        else:
            if RECTANGLE_MODE:
                MOUSE_PRESSED = False
                finish_roi_selection()
            RECTANGLE_MODE = False
            if mode == "MOVING_MODE":
                LABELING_MODE = False
                MOVING_MODE = True
                ADDING_MODE = False
                REMOVING_MODE = False
            elif mode == "LABELING_MODE":
                LABELING_MODE = True
                MOVING_MODE = False
                ADDING_MODE = False
                REMOVING_MODE = False
            elif mode == "ADDING_MODE":
                LABELING_MODE = False
                MOVING_MODE = False
                ADDING_MODE = True
                REMOVING_MODE = False
            elif mode == "REMOVING_MODE":
                LABELING_MODE = False
                MOVING_MODE = False
                ADDING_MODE = False
                REMOVING_MODE = True
    generate_left_bar_image()

def load_data(list_files):
    global RECTANGLE, LAST_SAVING, POINTS
    corr_txt = list_files[3]  # predictions_corrected
    det_txt = list_files[2]   # detections
    try:
        points = []
        # Priority 1: Load from predictions_corrected if exists
        if os.path.isfile(corr_txt):
            with open(corr_txt, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    x, y, r, cls = map(int, line.strip().split(','))
                    # [x, y, r, class, class_copy, is_modified, is_visible, confidence]
                    points.append([x, y, r, cls, cls, 1, 1, 99950])
            print(f"โหลดข้อมูลแก้ไขแล้วจาก {corr_txt}")
        # Priority 2: Load from detections if no corrected file
        elif os.path.isfile(det_txt):
            with open(det_txt, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    x, y, r = map(int, line.strip().split(','))
                    points.append([x, y, r, 1, 1, 0, 1, 99950])
            print(f"โหลดข้อมูลตรวจจับจาก {det_txt}")
        points = np.array(points, dtype=np.int32) if points else np.array([], dtype=np.int32)
        RECTANGLE = ((0, 0), (1920, 1080) if RESOLUTION == 0 else (1280, 720))
        LAST_SAVING = datetime.datetime.fromtimestamp(os.path.getmtime(corr_txt)).strftime("%d/%m/%y %H:%M:%S") if os.path.isfile(corr_txt) else ""
        return RECTANGLE, LAST_SAVING, points
    except Exception as e:
        print(f"ข้อผิดพลาดในการโหลดข้อมูลจาก {corr_txt} หรือ {det_txt}: {str(e)}")
        return RECTANGLE, LAST_SAVING, np.array([])

def process_entries(img_det):
    global OPTION, POINTS, REFRESH, MOVING_MODE, MOVING_DIRECTION, RECTANGLE_MODE, MIN_DIST, LAST_SAVING, RECTANGLE, REFRESH_RECTANGLE, AVG_CELL_SIZE, VIEW_MODE
    img = cv2.imread(img_det[1])
    if img is None:
        print(f"ไม่สามารถโหลดภาพ {img_det[1]}")
        return True
    img_bk = np.copy(img)
    RECTANGLE, LAST_SAVING, POINTS = load_data(img_det)
    if POINTS.size == 0:
        # If no data, detect cells
        POINTS = detect_cells(img_det[0], img)
        if POINTS.size == 0:
            print(f"ไม่มีข้อมูลเซลล์สำหรับ {img_det[0]}")
            return True
    MIN_DIST = np.average(POINTS[:, 2]) if POINTS.size > 0 else 10
    AVG_CELL_SIZE = int(np.bincount(POINTS[:, 2]).argmax()) if POINTS.size > 0 else 10
    generate_left_bar_image()
    img = process_trackbar(img)
    img = draw_circles(img)
    cv2.imshow(img_det[0], draw_rectangle(img))
    last_key = 1
    while True:
        k = cv2.waitKey(1)
        if k != last_key:
            if k == 27:  # Esc close stop
                return False
            if k in CLASSES_KEY:  # Classes
                OPTION = k
                change_mode("LABELING_MODE")
            elif k == 32:  # Space Moving image
                OPTION = 32
                change_mode("MOVING_MODE")
            elif k == 97:  # A add cell
                change_mode("ADDING_MODE")
            elif k == 100:  # D delete cell
                change_mode("REMOVING_MODE")
            elif k == 114:  # R key select ROI
                OPTION = 114
                change_mode("RECTANGLE_MODE")
            elif k == 118:  # V key view cells without label
                if VIEW_MODE:
                    OPTION = 32
                    change_mode("MOVING_MODE")
                else:
                    OPTION = 118
                    change_mode("VIEW_MODE")
                REFRESH = True
            elif k == 13:  # Enter key Finish ROI selection
                if RECTANGLE_MODE:
                    OPTION = 32
                    change_mode("MOVING_MODE")
                    finish_roi_selection()
                    generate_left_bar_image()
            elif k == 110:  # N key to next image
                MOVING_DIRECTION = "F"
                change_mode("MOVING_MODE")
                return True
            elif k == 112:  # P key to previous image
                MOVING_DIRECTION = "B"
                change_mode("MOVING_MODE")
                return True
            elif k == 115:  # S key to save
                save(draw_rectangle(img), img_det[0], img_det[3])
            elif k == 8:  # BS key to reset
                if os.path.isfile(img_det[3]):
                    os.remove(img_det[3])
                process_entries(img_det)
            last_key = k
        if REFRESH:
            img = process_trackbar(np.copy(img_bk))
            REFRESH = False
            if not VIEW_MODE:
                img = draw_circles(img)
                cv2.imshow(img_det[0], draw_rectangle(img))
            else:
                cv2.imshow(img_det[0], img)
        if REFRESH_RECTANGLE:
            img = np.copy(img_bk)
            img = process_trackbar(img)
            img = draw_circles(img)
            cv2.imshow(img_det[0], draw_rectangle(img))
            REFRESH_RECTANGLE = False

def finish_roi_selection():
    global REFRESH_RECTANGLE, POINTS
    if POINTS.size == 0:
        return
    (xmin, ymin), (xmax, ymax) = RECTANGLE
    POINTS[:, 6] = 0
    cond = np.where(
        np.logical_and(POINTS[:, 0] >= xmin, POINTS[:, 0] <= xmax)
        & np.logical_and(POINTS[:, 1] >= ymin, POINTS[:, 1] <= ymax)
    )
    POINTS[cond, 6] = 1
    REFRESH_RECTANGLE = True

def get_selected_class():
    return OPTION - 49

def hover_cell(x, y):
    global MIN_DIST, POINTS, REFRESH
    if POINTS.size == 0:
        return
    point = np.array([x, y])
    min_dist_index = np.sum(np.square(np.abs(point - POINTS[:, :2])), 1).argmin()
    min_dist = np.sqrt(np.sum(np.square(np.abs(point - POINTS[min_dist_index, :2])), 0))
    if min_dist <= MIN_DIST and POINTS[min_dist_index, 6]:
        if POINTS[min_dist_index, 4] != get_selected_class():
            POINTS[min_dist_index, 4] = get_selected_class()
            POINTS[min_dist_index, 5] = 1
        else:
            POINTS[min_dist_index, 5] = 0 if POINTS[min_dist_index, 5] == 1 else 1
        REFRESH = True

def callback_rectangle(lbt_down, mouse_move, lbt_up, x, y):
    global IX, IY, RECTANGLE, MOUSE_PRESSED, REFRESH
    if lbt_down:
        MOUSE_PRESSED = True
        IX, IY = x, y
        RECTANGLE = ((IX, IY), (x, y))
    elif mouse_move:
        if MOUSE_PRESSED:
            RECTANGLE = ((IX, IY), (x, y))
            REFRESH = True
    elif lbt_up:
        MOUSE_PRESSED = False
        RECTANGLE = ((IX, IY), (x, y))
        REFRESH = True

def callback_labeling(lbt_down, mouse_move, lbt_up, x, y):
    global MOUSE_PRESSED, REFRESH
    if lbt_down or (mouse_move and MOUSE_PRESSED):
        MOUSE_PRESSED = True
        hover_cell(x, y)
    elif lbt_up:
        MOUSE_PRESSED = False
        generate_left_bar_image()

def callback_adding(lbt_down, mouse_move, lbt_up, x, y):
    global POINTS, REFRESH
    if lbt_down:
        (xmin, ymin), (xmax, ymax) = RECTANGLE
        if xmin <= x <= xmax and ymin <= y <= ymax:
            cl = get_selected_class() if OPTION in CLASSES_KEY else 1
            new_cell = np.array([x, y, AVG_CELL_SIZE, cl, cl, 1, 1, 99950], dtype=np.int32)
            if POINTS.size == 0:
                POINTS = new_cell.reshape(1, -1)
            else:
                POINTS = np.vstack((POINTS, new_cell))
    elif lbt_up:
        generate_left_bar_image()
        REFRESH = True

def callback_remove(lbt_down, mouse_move, lbt_up, x, y):
    global POINTS, REFRESH
    if lbt_down and POINTS.size > 0:
        (xmin, ymin), (xmax, ymax) = RECTANGLE
        if xmin <= x <= xmax and ymin <= y <= ymax:
            point = np.array([x, y])
            min_dist_index = np.sum(np.square(np.abs(point - POINTS[:, :2])), 1).argmin()
            POINTS = np.delete(POINTS, min_dist_index, 0)
    elif lbt_up:
        generate_left_bar_image()
        REFRESH = True

def annotate_cells(event, x, y, flags, param):
    global MOUSE_PRESSED, RECTANGLE_MODE, MOVING_MODE, LABELING_MODE, ADDING_MODE, REMOVING_MODE, VIEW_MODE
    if RECTANGLE_MODE:
        MOVING_MODE = False
        LABELING_MODE = False
        ADDING_MODE = False
        REMOVING_MODE = False
        callback_rectangle(
            event == cv2.EVENT_LBUTTONDOWN,
            event == cv2.EVENT_MOUSEMOVE,
            event == cv2.EVENT_LBUTTONUP,
            x,
            y,
        )
    elif LABELING_MODE:
        MOVING_MODE = False
        RECTANGLE_MODE = False
        ADDING_MODE = False
        REMOVING_MODE = False
        callback_labeling(
            event == cv2.EVENT_LBUTTONDOWN,
            event == cv2.EVENT_MOUSEMOVE,
            event == cv2.EVENT_LBUTTONUP,
            x,
            y,
        )
    elif ADDING_MODE:
        MOVING_MODE = False
        RECTANGLE_MODE = False
        LABELING_MODE = False
        REMOVING_MODE = False
        callback_adding(
            event == cv2.EVENT_LBUTTONDOWN,
            event == cv2.EVENT_MOUSEMOVE,
            event == cv2.EVENT_LBUTTONUP,
            x,
            y,
        )
    elif REMOVING_MODE:
        MOVING_MODE = False
        RECTANGLE_MODE = False
        LABELING_MODE = False
        ADDING_MODE = False
        callback_remove(
            event == cv2.EVENT_LBUTTONDOWN,
            event == cv2.EVENT_MOUSEMOVE,
            event == cv2.EVENT_LBUTTONUP,
            x,
            y,
        )
    elif MOVING_MODE or VIEW_MODE:
        pass

def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    return cv2.LUT(image, table)

def process_trackbar(img):
    global REFRESH, IM_NAME, OLD_GAMMA, OLD_BRIGHTNESS, OLD_CONTRAST, REFRESH_RECTANGLE
    if GAMMA != OLD_GAMMA or REFRESH or REFRESH_RECTANGLE:
        gamma = GAMMA / 2
        img = adjust_gamma(img, 0.1 if gamma == 0 else gamma)
    OLD_GAMMA = GAMMA
    return img

def trackbar(x):
    global REFRESH, GAMMA, BRIGHTNESS, CONTRAST, IM_NAME, OLD_GAMMA, OLD_BRIGHTNESS, OLD_CONTRAST
    GAMMA = cv2.getTrackbarPos("Gamma", IM_NAME)
    REFRESH = True

def save(img, im_name, out_txt):
    global LEFT_BAR_SIZE, LAST_SAVING, POINTS
    LAST_SAVING = datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S")
    out_path = os.path.join(OUT_IMGS, im_name)
    create_folder(out_path)
    old_size = LEFT_BAR_SIZE
    LEFT_BAR_SIZE = (img.shape[0], LEFT_BAR_SIZE[1])
    left_bar = generate_left_bar_image(True)
    cv2.imwrite(out_path, np.hstack((left_bar, img)))
    LEFT_BAR_SIZE = old_size
    create_folder(out_txt)
    with open(out_txt, 'w') as f:
        for point in POINTS:
            if point[6] == 1:
                x, y, r, cell_class = point[0], point[1], point[2], point[4]
                f.write(f"{x},{y},{r},{cell_class}\n")
    generate_left_bar_image()

def create_folder(path):
    path = "/".join(path.split("/")[:-1])
    if not os.path.exists(path):
        os.makedirs(path)

def create_detections():
    images = find_image_names()
    with tqdm(total=len(images)) as j:
        for i in images:
            det_txt = os.path.join(DET_PATH, os.path.splitext(os.path.basename(i))[0] + '.txt')
            corr_txt = os.path.join(OUT_ANNOT, os.path.splitext(os.path.basename(i))[0] + '.txt')
            if os.path.isfile(det_txt) or os.path.isfile(corr_txt):
                print(f"ข้ามการตรวจจับสำหรับ {i} เพราะมีไฟล์ .txt แล้ว")
                j.update(1)
                continue
            img = cv2.imread(os.path.join(IMAGES_PATH, i))
            if img is None:
                print(f"ไม่สามารถโหลดภาพ {i}")
                j.update(1)
                continue
            detect_cells(i, img)
            j.update(1)

def main():
    global MOVING_DIRECTION, OPTION, POINTS, IM_NAME
    load_configs()
    set_window_size()
    print("\nกำลังตรวจจับเซลล์สำหรับภาพที่ยังไม่มีไฟล์ .txt...")
    create_detections()
    print("\nกำลังเปิด UI สำหรับแก้ไข...")
    list_im_det = get_images_detections(IMAGES_PATH, DET_PATH, OUT_ANNOT)
    if not list_im_det:
        print("ไม่พบภาพที่สอดคล้อง")
        return
    index = 0
    while True:
        i = list_im_det[index]
        OPTION = 32
        POINTS = np.array([])
        IM_NAME = i[0]
        cv2.namedWindow(i[0], 0)
        cv2.createTrackbar("Gamma", i[0], 2, 10, trackbar)
        cv2.setMouseCallback(i[0], annotate_cells)
        cv2.resizeWindow(i[0], *(SIZE_WINDOW_COMB))
        cv2.moveWindow(i[0], X_POS_WINDOW_COMB, 0)
        if not process_entries(i):
            break
        index = next_image(MOVING_DIRECTION, len(list_im_det), index)
        cv2.destroyAllWindows()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
