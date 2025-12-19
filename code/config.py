# config.py
import cv2

# ================== 使用者設定區 ==================
VIDEO_SOURCE = 0
CSV_FILE = r"yourpath\sedentary_log.csv"

# ================== Email 通知設定 ==================
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
SENDER_EMAIL = "youremail@gmail.com"
SENDER_PASSWORD = ""
RECEIVER_EMAIL = "youremail@gmail.com"

# ================== 音效檔案 ==================
ALARM_SOUND_FILE = "up.mp3"
SUCCESS_SOUND_FILE = "success.mp3"
START_SOUND_FILE = "start.mp3"
EMERGENCY_SOUND_FILE = "emergency.mp3"

# ================== YOLO & AI 參數 ==================
YOLO_MODEL = "yolov8n.pt"
YOLO_CONF = 0.60
YOLO_IOU = 0.45
YOLO_IMGSZ = 640
ROI_EXPAND = 0.20
MIN_ROI_SIZE = 150
HUMAN_CONFIDENCE_THRESHOLD = 0.60

# ================== 姿勢與安全參數 ==================
ANGLE_THRESHOLD_KNEE_STAND = 140
ANGLE_THRESHOLD_HIP_SIT = 120
THIGH_SLOPE_VERTICAL = 1.0
BOX_RATIO_THRESHOLD = 1.15
HEIGHT_DROP_THRESHOLD = 0.25

MOTION_THRESHOLD = 0.015
MOTION_EMA_ALPHA = 0.05

UNCONSCIOUS_TIME_THRESHOLD = 60.0
WAIT_FOR_STAND_TIMEOUT = 30.0
TRACKING_DISTANCE_THRESHOLD = 200

REHAB_TARGET_COUNT = 5

ALERT_INTERVAL = 10.0
MIN_SIT_DURATION = 1.0
WALKING_BUFFER_TIME = 1.0

# ================== 系統常數 ==================
BUFFER_SIZE = 10
FONT = cv2.FONT_HERSHEY_SIMPLEX
WINDOW_NAME = "Smart Care: Modularized V81"
