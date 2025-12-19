# filename: sedentary_final_v81_simple_audio.py
import cv2
import time
import mediapipe as mp
import pygame
import numpy as np
import csv
import os
import random 
from datetime import datetime, timedelta
import warnings
import math
import threading
from threading import Thread
from flask import Flask, jsonify, render_template_string, send_from_directory, Response, request, send_file
import traceback

# === Email 相關 ===
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# --- YOLO import ---
try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None
    print("ultralytics not available. Install with `pip install ultralytics`.")

# 過濾警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')

# ================== 使用者設定區 ==================
VIDEO_SOURCE = 0 # 電腦鏡頭
CSV_FILE = r"yourpath\sedentary_log.csv"

# ================== 📧 Email 通知設定 ==================
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
SENDER_EMAIL = "youremail@gmail.com"
SENDER_PASSWORD = ""
RECEIVER_EMAIL = "youremail@gmail.com"

# 音效檔案設定
ALARM_SOUND_FILE = "up.mp3"         
SUCCESS_SOUND_FILE = "success.mp3"  
START_SOUND_FILE = "start.mp3"      
EMERGENCY_SOUND_FILE = "emergency.mp3" 

YOLO_MODEL = "yolov8n.pt"
YOLO_CONF = 0.60 
YOLO_IOU = 0.45
YOLO_IMGSZ = 640

ROI_EXPAND = 0.20 
MIN_ROI_SIZE = 150 

# === 姿勢判斷參數 ===
ANGLE_THRESHOLD_KNEE_STAND = 140
ANGLE_THRESHOLD_HIP_SIT = 120     
THIGH_SLOPE_VERTICAL = 1.0        
BOX_RATIO_THRESHOLD = 1.15        
HEIGHT_DROP_THRESHOLD = 0.25 

# === 動作偵測參數 ===
MOTION_THRESHOLD = 0.02  
MOTION_EMA_ALPHA = 0.05    

# === 安全參數 ===
UNCONSCIOUS_TIME_THRESHOLD = 60.0 
WAIT_FOR_STAND_TIMEOUT = 30.0     
TRACKING_DISTANCE_THRESHOLD = 200 

# === 真人檢測參數 ===
HUMAN_CONFIDENCE_THRESHOLD = 0.60 

# === 復健遊戲參數 ===
REHAB_TARGET_COUNT = 5  

# === 隱私保護設定 ===
ENABLE_PRIVACY_MODE = True  

ALERT_INTERVAL = 10.0
MIN_SIT_DURATION = 1.0
WALKING_BUFFER_TIME = 1.0

BUFFER_SIZE = 10
FONT = cv2.FONT_HERSHEY_SIMPLEX
WINDOW_NAME = "Smart Care: V81 Simple Audio"

# ================== 全域系統設定 ==================
APP_CONFIG = {
    "privacy_mode": True,      
    "sound_alert": True,       
    "alert_interval": 10.0,     
    "force_reset": False,
    "force_recalibrate": False,
    "hard_reset": False,
    "sleep_mode": False,
    "emergency_reset": False 
}

# 全域狀態
GLOBAL_STATUS = {
    "is_rehab_mode": False,
    "balance_score": "--",
    "balance_grade": "Waiting...",
    "balance_color": "#b2b2b2",
    "balance_history": [],
    "is_emergency": False 
}

outputFrame = None
lock = threading.Lock()

# ================== 📧 Email 發送函式 ==================
def send_emergency_email():
    try:
        # print("📧 正在連線 SMTP 伺服器...")
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECEIVER_EMAIL
        msg['Subject'] = "⚠️ [緊急警報] 偵測到長輩長時間無反應！"

        body = f"""
        <h1>緊急通知</h1>
        <p>系統偵測到長輩處於緊急狀態。</p>
        <p><b>發生時間：</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p style="color:red; font-weight:bold;">請立即確認長輩狀況！</p>
        <hr>
        <p><i>此為 Smart Care 系統自動發送</i></p>
        """
        msg.attach(MIMEText(body, 'html'))

        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
        
        print(f"✅ Email Sent to {RECEIVER_EMAIL}")
    except Exception as e:
        print(f"❌ Email Failed: {e}")

# ================== 視覺特效粒子系統 ==================
class FireworkParticle:
    def __init__(self, x, y, color, explode=False):
        self.x = x
        self.y = y
        self.color = color
        if explode:
            self.vx = random.uniform(-10, 10)
            self.vy = random.uniform(-10, 10)
            self.life = random.uniform(0.8, 1.2)
            self.gravity = 0.5
        else:
            self.vx = random.uniform(-5, 5)
            self.vy = random.uniform(-5, 5)
            self.life = 1.0
            self.gravity = 0.1
        self.radius = random.randint(3, 8)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += self.gravity
        self.life -= 0.04
        if self.radius > 0.1:
            self.radius -= 0.1

    def draw(self, img):
        if self.life > 0:
            cv2.circle(img, (int(self.x), int(self.y)), int(self.radius), self.color, -1)

particles = []

def spawn_hit_effect(x, y):
    for _ in range(10):
        c = (random.randint(200,255), random.randint(200,255), 255)
        particles.append(FireworkParticle(x, y, c, explode=False))

def spawn_firework(x, y):
    color = (random.randint(50,255), random.randint(50,255), random.randint(50,255))
    for _ in range(30):
        particles.append(FireworkParticle(x, y, color, explode=True))

# ================== 繪圖樣式 ==================
landmark_style = mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 255), thickness=4, circle_radius=5)
connection_style = mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 0), thickness=3)

# ================== 骨架平滑器 ==================
class PoseSmoother:
    def __init__(self, alpha=0.6):
        self.alpha = alpha
        self.prev_landmarks = None

    def smooth(self, current_landmarks):
        if not current_landmarks: return None
        if self.prev_landmarks is None:
            self.prev_landmarks = current_landmarks
            return current_landmarks
        for i, lm in enumerate(current_landmarks.landmark):
            prev = self.prev_landmarks.landmark[i]
            lm.x = self.alpha * lm.x + (1 - self.alpha) * prev.x
            lm.y = self.alpha * lm.y + (1 - self.alpha) * prev.y
            lm.z = self.alpha * lm.z + (1 - self.alpha) * prev.z
        self.prev_landmarks = current_landmarks
        return current_landmarks

    def reset(self):
        self.prev_landmarks = None

# ================== ActivityLogger ==================
class ActivityLogger:
    def __init__(self, filename):
        self.filename = filename
        self.events_file = os.path.splitext(self.filename)[0] + "_events.csv"
        self.header = ["Date", "First_Seen_Time", "Stand_Count", "Total_Sit_Time(s)", "Max_Continuous_Sit(s)"]
        folder = os.path.dirname(self.filename)
        if folder and not os.path.exists(folder):
            try: os.makedirs(folder)
            except: pass
        self.current_sit_start_ts = None 
        self.data = { "First_Seen_Time": "", "Stand_Count": 0, "Total_Sit_Seconds": 0.0, "Max_Sit_Seconds": 0.0 }
        self._check_files()
        self.load_today_data()

    def get_today_str(self):
        return datetime.now().strftime("%Y-%m-%d")

    def _check_files(self):
        if not os.path.exists(self.events_file):
            try:
                with open(self.events_file, 'w', newline='', encoding='utf-8-sig') as f:
                    csv.writer(f).writerow(["Date", "Sit_Start", "Sit_End", "Duration_sec"])
            except: pass
        if not os.path.exists(self.filename):
            with open(self.filename, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(self.header)

    def load_today_data(self):
        current_date = self.get_today_str()
        found_today = False
        if os.path.exists(self.filename):
            try:
                rows = []
                with open(self.filename, 'r', encoding='utf-8-sig') as f:
                    reader = csv.reader(f)
                    next(reader, None)
                    for r in reader: rows.append(r)
                for r in rows:
                    if len(r) >= 5 and r[0] == current_date:
                        self.data["First_Seen_Time"] = r[1]
                        self.data["Stand_Count"] = int(r[2])
                        self.data["Total_Sit_Seconds"] = float(r[3])
                        self.data["Max_Sit_Seconds"] = float(r[4])
                        found_today = True
                        break
                if not found_today:
                    self.data = { "First_Seen_Time": datetime.now().strftime("%H:%M:%S"), "Stand_Count": 0, "Total_Sit_Seconds": 0.0, "Max_Sit_Seconds": 0.0 }
                    self.append_today_row()
            except: pass

    def append_today_row(self):
        with open(self.filename, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow([self.get_today_str(), self.data["First_Seen_Time"], self.data["Stand_Count"], 
                             self.data["Total_Sit_Seconds"], self.data["Max_Sit_Seconds"]])
    
    def hard_reset_today(self):
        print("Executing Hard Reset...")
        self.data = {
            "First_Seen_Time": datetime.now().strftime("%H:%M:%S"),
            "Stand_Count": 0,
            "Total_Sit_Seconds": 0.0,
            "Max_Sit_Seconds": 0.0
        }
        self.current_sit_start_ts = None
        self.save() 

    def log_first_seen(self):
        self.load_today_data()

    def set_current_sitting(self, start_ts):
        self.current_sit_start_ts = start_ts

    def log_stand_up(self, sit_start_ts, sit_end_ts, sit_duration):
        self.current_sit_start_ts = None
        self.data["Stand_Count"] += 1
        self.data["Total_Sit_Seconds"] += sit_duration
        if sit_duration > self.data["Max_Sit_Seconds"]:
            self.data["Max_Sit_Seconds"] = sit_duration
        self.save()
        try:
            with open(self.events_file, 'a', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.get_today_str(),
                    time.strftime("%H:%M:%S", time.localtime(sit_start_ts)), 
                    time.strftime("%H:%M:%S", time.localtime(sit_end_ts)),
                    round(sit_duration, 1)
                ])
        except Exception as e: print("Append event failed:", e)

    def save(self):
        current_date = self.get_today_str()
        try:
            lines = []
            with open(self.filename, 'r', encoding='utf-8-sig') as f:
                reader = csv.reader(f)
                lines = list(reader)
            updated = False
            for i, row in enumerate(lines):
                if i > 0 and len(row) > 0 and row[0] == current_date:
                    lines[i] = [current_date, self.data["First_Seen_Time"], self.data["Stand_Count"],
                                round(self.data["Total_Sit_Seconds"], 1), round(self.data["Max_Sit_Seconds"], 1)]
                    updated = True
                    break
            if not updated:
                lines.append([current_date, self.data["First_Seen_Time"], self.data["Stand_Count"],
                              round(self.data["Total_Sit_Seconds"], 1), round(self.data["Max_Sit_Seconds"], 1)])
            with open(self.filename, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerows(lines)
        except: pass

    def read_events(self, limit=200):
        rows = []
        if os.path.exists(self.events_file):
            try:
                with open(self.events_file, 'r', encoding='utf-8-sig') as f:
                    reader = csv.reader(f)
                    next(reader, None)
                    for r in reader: 
                        if r: rows.append(r)
            except: pass
        return rows[-limit:]

    def read_hourly_stats(self):
        current_date = self.get_today_str()
        hourly_data = [0.0] * 24
        if os.path.exists(self.events_file):
            try:
                with open(self.events_file, 'r', encoding='utf-8-sig') as f:
                    reader = csv.reader(f)
                    next(reader, None)
                    for r in reader:
                        if len(r) >= 4 and r[0] == current_date:
                            try:
                                h = int(r[1].split(':')[0])
                                hourly_data[h] += (float(r[3]) / 60.0)
                            except: pass
            except: pass
        if self.current_sit_start_ts is not None:
            if not GLOBAL_STATUS["is_rehab_mode"]:
                now_ts = time.time()
                start_dt = datetime.fromtimestamp(self.current_sit_start_ts)
                if start_dt.strftime("%Y-%m-%d") == current_date:
                    current_hour = datetime.now().hour
                    duration_min = (now_ts - self.current_sit_start_ts) / 60.0
                    hourly_data[current_hour] += duration_min
        return [int(x) for x in hourly_data]

    def read_daily_history(self):
        dates, minutes = [], []
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r', encoding='utf-8-sig') as f:
                    reader = csv.reader(f)
                    next(reader, None)
                    for r in reader:
                        if len(r) >= 4:
                            try:
                                d_obj = datetime.strptime(r[0], "%Y-%m-%d")
                                dates.append(d_obj.strftime("%m/%d"))
                                minutes.append(int(float(r[3]) / 60.0))
                            except: pass
            except: pass
        if self.current_sit_start_ts is not None and len(minutes) > 0:
             if not GLOBAL_STATUS["is_rehab_mode"]:
                 duration_min = (time.time() - self.current_sit_start_ts) / 60.0
                 minutes[-1] += int(duration_min)
        return dates[-7:], minutes[-7:]

# ================== 幾何計算 ==================
def calculate_angle_3d(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom == 0: return 0.0
    cosine_angle = np.dot(ba, bc) / denom
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))

def expand_box(x1, y1, x2, y2, W, H, expand_ratio=0.2):
    bw = x2 - x1
    bh = y2 - y1
    x1e = max(0, int(x1 - bw * expand_ratio))
    y1e = max(0, int(y1 - bh * expand_ratio))
    x2e = min(W - 1, int(x2 + bw * expand_ratio))
    y2e = min(H - 1, int(y2 + bh * expand_ratio))
    return x1e, y1e, x2e, y2e

def select_primary_person(boxes):
    if not boxes: return None
    return max(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))

# ================== analyze_pose ==================
def analyze_pose(world_landmarks, img_landmarks, img_w, img_h, box_ratio, current_y, ref_stand_y, box_h):
    debug_vals = {"Knee": 0, "Hip": 0, "Ratio": 0.0}
    try:
        VIS_THRESH = 0.5
        l_hip_lm = img_landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip_lm = img_landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        l_knee_lm = img_landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        r_knee_lm = img_landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        l_ankle_lm = img_landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_ankle_lm = img_landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        l_sh = img_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_sh = img_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

        l_wrist = img_landmarks[15]
        r_wrist = img_landmarks[16]

        is_knees_visible = (l_knee_lm.visibility > VIS_THRESH or r_knee_lm.visibility > VIS_THRESH)
        is_ankles_visible = (l_ankle_lm.visibility > VIS_THRESH or r_ankle_lm.visibility > VIS_THRESH)

        if (l_wrist.visibility > 0.5 and l_wrist.y < l_sh.y) or \
           (r_wrist.visibility > 0.5 and r_wrist.y < r_sh.y):
            return "standing", "Hands Up!", True, debug_vals

        l_sh_3d = [world_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, world_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y, world_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
        l_hip_3d = [world_landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, world_landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y, world_landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z]
        l_knee_3d = [world_landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, world_landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y, world_landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z]
        l_ankle_3d = [world_landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, world_landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y, world_landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z]
        r_sh_3d = [world_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, world_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y, world_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z]
        r_hip_3d = [world_landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, world_landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y, world_landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z]
        r_knee_3d = [world_landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, world_landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y, world_landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z]
        r_ankle_3d = [world_landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, world_landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y, world_landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z]

        knee_angle = 0
        hip_angle = 0
        if is_ankles_visible and is_knees_visible:
            knee_angle = max(calculate_angle_3d(l_hip_3d, l_knee_3d, l_ankle_3d), calculate_angle_3d(r_hip_3d, r_knee_3d, r_ankle_3d))
            hip_angle = max(calculate_angle_3d(l_sh_3d, l_hip_3d, l_knee_3d), calculate_angle_3d(r_sh_3d, r_hip_3d, r_knee_3d))
            debug_vals["Knee"] = int(knee_angle)
            debug_vals["Hip"] = int(hip_angle)

    except Exception: return "unknown", "Error", False, debug_vals

    if ref_stand_y is not None and box_h > 0:
        drop_ratio = (current_y * img_h - ref_stand_y * img_h) / box_h
        if drop_ratio > HEIGHT_DROP_THRESHOLD:
            return "sitting", f"Drop {drop_ratio:.2f}", False, debug_vals

    ratio = 0
    if is_knees_visible:
        l_thigh_h = l_knee_lm.y - l_hip_lm.y 
        r_thigh_h = r_knee_lm.y - r_hip_lm.y
        max_thigh_h = max(l_thigh_h, r_thigh_h)
        torso_h = max(abs(l_hip_lm.y - l_sh.y), abs(r_hip_lm.y - r_sh.y))
        
        if torso_h > 0:
            ratio = max_thigh_h / torso_h
            debug_vals["Ratio"] = round(ratio, 2)
            
            if ref_stand_y is None:
                if hip_angle > 0 and hip_angle < 135 and ratio < 0.4:
                    return "sitting", f"HipBend:{int(hip_angle)}", False, debug_vals
                if ratio > 0.45 or knee_angle > 140 or box_ratio > 1.15:
                    return "standing", "Init:Stand", True, debug_vals
                else:
                    return "sitting", "Init:Check", False, debug_vals
            
            if ratio > 0.5: return "standing", f"R:{ratio:.2f}", True, debug_vals
            elif ratio < 0.35: return "sitting", f"R:{ratio:.2f}", False, debug_vals

    if knee_angle > ANGLE_THRESHOLD_KNEE_STAND and hip_angle > ANGLE_THRESHOLD_HIP_SIT:
        return "standing", "Angles OK", True, debug_vals

    debug_str = f"Box R:{box_ratio:.1f}"
    box_thresh = 1.15 if ref_stand_y is None else BOX_RATIO_THRESHOLD 
    if box_ratio > box_thresh: return "standing", debug_str, True, debug_vals
    else: return "sitting", debug_str, False, debug_vals

# ================== 初始化 ==================
try:
    pygame.mixer.init()
    if os.path.exists(ALARM_SOUND_FILE): 
        alert_sound = pygame.mixer.Sound(ALARM_SOUND_FILE)
    else: 
        alert_sound = None
    
    if os.path.exists(SUCCESS_SOUND_FILE):
        success_sound = pygame.mixer.Sound(SUCCESS_SOUND_FILE)
    else:
        success_sound = None

    if os.path.exists(START_SOUND_FILE):
        start_sound = pygame.mixer.Sound(START_SOUND_FILE)
    else:
        start_sound = None

    if os.path.exists(EMERGENCY_SOUND_FILE):
        emergency_sound = pygame.mixer.Sound(EMERGENCY_SOUND_FILE)
    else:
        emergency_sound = None

except: 
    alert_sound = None
    success_sound = None
    start_sound = None
    emergency_sound = None

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7, model_complexity=1, smooth_landmarks=True)

yolo = None
if YOLO and YOLO_MODEL:
    try: yolo = YOLO(YOLO_MODEL)
    except: yolo = None

logger = ActivityLogger(CSV_FILE)

# ================== Flask Web App ==================
def create_flask_app(logger_obj):
    app = Flask(__name__)
    TEMPLATE = """
    <!doctype html>
    <html lang="zh-Hant">
      <head>
        <meta charset="utf-8"/>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta http-equiv="refresh" content="10">
        <title>銀髮族久坐照護系統</title>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
          :root { --bg-color: #1a1a2e; --card-bg: #16213e; --accent-color: #0f3460; --highlight: #e94560; --text-main: #eaeaea; --text-sub: #b2b2b2; --success: #4cc9f0; }
          body { font-family: 'Segoe UI', Roboto, sans-serif; background: var(--bg-color); color: var(--text-main); margin: 0; padding: 20px; }
          .container { max-width: 1000px; margin: 0 auto; }
          header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; border-bottom: 2px solid var(--accent-color); padding-bottom: 10px; }
          .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px; }
          .card { background: var(--card-bg); border-radius: 12px; padding: 15px; text-align: center; border: 1px solid #2a2a40; }
          .card-val { font-size: 2rem; font-weight: bold; }
          .charts-row { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }
          @media (max-width: 768px) { .charts-row { grid-template-columns: 1fr; } }
          .chart-box { background: var(--card-bg); padding: 15px; border-radius: 12px; border: 1px solid #2a2a40; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
          .video-box { background: #000; border-radius: 12px; overflow: hidden; margin-bottom: 20px; border: 2px solid var(--success); text-align: center; }
          .control-panel { background: var(--card-bg); padding: 15px; border-radius: 12px; border: 1px solid var(--highlight); margin-bottom: 20px; display: flex; flex-wrap: wrap; gap: 20px; align-items: center; justify-content: space-around; }
          .ctrl-item { display: flex; align-items: center; gap: 10px; }
          .btn { background: var(--accent-color); color: #fff; border: none; padding: 8px 16px; border-radius: 5px; cursor: pointer; font-size: 0.9rem; transition: 0.3s; }
          .btn:hover { background: var(--success); color: #000; }
          .btn-yellow { background: #d4a017; }
          .btn-yellow:hover { background: #ffd700; color: #000; }
          .btn-red { background: #8B0000; }
          .btn-red:hover { background: #FF0000; }
          .switch { position: relative; display: inline-block; width: 40px; height: 24px; }
          .switch input { opacity: 0; width: 0; height: 0; }
          .slider { position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0; background-color: #ccc; transition: .4s; border-radius: 24px; }
          .slider:before { position: absolute; content: ""; height: 16px; width: 16px; left: 4px; bottom: 4px; background-color: white; transition: .4s; border-radius: 50%; }
          input:checked + .slider { background-color: var(--success); }
          input:checked + .slider:before { transform: translateX(16px); }
          table { width: 100%; border-collapse: collapse; background: var(--card-bg); border-radius: 12px; overflow: hidden; }
          th, td { padding: 10px; text-align: left; border-bottom: 1px solid var(--accent-color); }
          th { background: #0f3460; color: #4cc9f0; }
          
          .audio-toggle-btn { background: #6c757d; }
          .audio-toggle-btn.on { background: #28a745; }
        </style>
      </head>
      <body>
        <div class="container">
          <header>
            <h2><i class="fa-solid fa-heart-pulse"></i> 銀髮族照護監控</h2>
            <div id="serverTime">{{server_time}}</div>
          </header>
          
          <div class="control-panel">
            <div class="ctrl-item">
                <button class="btn btn-red" onclick="resetEmergency()"><i class="fa-solid fa-bell-slash"></i> 解除緊急警報</button>
            </div>
            <div class="ctrl-item">
                <span>睡眠模式</span>
                <label class="switch">
                  <input type="checkbox" id="sleepToggle" onchange="toggleSleep()" {{ 'checked' if config.sleep_mode else '' }}>
                  <span class="slider"></span>
                </label>
            </div>
            <div class="ctrl-item">
                <span>隱私模式</span>
                <label class="switch">
                  <input type="checkbox" id="privacyToggle" onchange="togglePrivacy()" {{ 'checked' if config.privacy_mode else '' }}>
                  <span class="slider"></span>
                </label>
            </div>
            <div class="ctrl-item">
                <button id="webAudioBtn" class="btn audio-toggle-btn" onclick="toggleWebAudio()">
                    <i class="fa-solid fa-volume-xmark"></i> 網頁音效: 關閉
                </button>
            </div>
            <div class="ctrl-item">
                <span>警報(秒):</span>
                <input type="number" id="alertInput" value="{{ config.alert_interval }}" style="width:60px; padding:5px; border-radius:4px; border:none;">
                <button class="btn" onclick="updateAlertTime()">設定</button>
            </div>
            <div class="ctrl-item">
                <button class="btn btn-yellow" onclick="recalibrate()"><i class="fa-solid fa-ruler-vertical"></i> 重新校正身高</button>
            </div>
            <div class="ctrl-item">
                <button class="btn btn-red" onclick="hardReset()"><i class="fa-solid fa-bomb"></i> 系統重置</button>
            </div>
          </div>

          <div class="stats-grid">
             <div class="card"><div style="color:#e94560">今日起立 (次)</div><div class="card-val" id="val_stand">{{stand_count}}</div></div>
             <div class="card"><div style="color:#4cc9f0">今日累積 (秒)</div><div class="card-val" id="val_sit">{{total_sit}}</div></div>
             <div class="card"><div style="color:#e94560">最長久坐 (秒)</div><div class="card-val" id="val_max">{{max_sit}}</div></div>
             <div class="card">
               <div style="color:#20c997">今日平均平衡 (Avg Score)</div>
               <div class="card-val" id="val_balance" style="color: {{ balance_color }}">{{ balance_score }} <br><span style="font-size:0.5em">{{ balance_grade }}</span></div>
             </div>
          </div>

          <div class="video-box">
             <div style="background:#222; padding:5px; color:#fff;">即時監控畫面 (Live Stream)</div>
             <img src="{{ url_for('video_feed') }}" width="100%" style="display:block;">
          </div>

          <div class="charts-row">
             <div class="chart-box"><h3 style="margin-top:0; color:#4cc9f0; font-size:1rem;">今日24小時分佈</h3><canvas id="hourlyChart"></canvas></div>
             <div class="chart-box"><h3 style="margin-top:0; color:#e94560; font-size:1rem;">歷史每日總久坐</h3><canvas id="dailyChart"></canvas></div>
          </div>
          
          <h3>近期久坐紀錄</h3>
          <table>
             <tr><th>日期</th><th>開始時間</th><th>結束時間</th><th>持續時間</th></tr>
             {% for e in events %}
             <tr>
                 <td>{{e[0]}}</td>
                 <td>{{e[1]}}</td>
                 <td>{{e[2]}}</td>
                 <td>{{e[4]}}</td>
             </tr>
             {% endfor %}
          </table>
        </div>
        
        <audio id="alarmAudio" src="" loop></audio>

        <script>
           const ctx1 = document.getElementById('hourlyChart').getContext('2d');
           const chart1 = new Chart(ctx1, { type: 'bar', data: { labels: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23], datasets: [{ label: '分鐘', data: {{ hourly_data | safe }}, backgroundColor: 'rgba(76, 201, 240, 0.6)', borderColor: '#4cc9f0', borderWidth: 1 }] }, options: { scales: { y: { beginAtZero: true, grid: {color: '#2a2a40'} }, x: { grid: {display: false} } }, plugins: { legend: {display: false} } } });
           
           const ctx2 = document.getElementById('dailyChart').getContext('2d');
           const chart2 = new Chart(ctx2, { type: 'line', data: { labels: {{ daily_labels | safe }}, datasets: [{ label: '分鐘', data: {{ daily_data | safe }}, backgroundColor: 'rgba(233, 69, 96, 0.2)', borderColor: '#e94560', borderWidth: 2, fill: true, tension: 0.3 }] }, options: { scales: { y: { beginAtZero: true, grid: {color: '#2a2a40'} }, x: { grid: {display: false} } }, plugins: { legend: {display: false} } } });

           let webAudioEnabled = false;
           const audioPlayer = document.getElementById('alarmAudio');
           const audioBtn = document.getElementById('webAudioBtn');
           let currentSound = "";

           function toggleWebAudio() {
               if (!webAudioEnabled) {
                   webAudioEnabled = true;
                   audioBtn.innerHTML = '<i class="fa-solid fa-volume-high"></i> 網頁音效: 開啟';
                   audioBtn.classList.add('on');
               } else {
                   webAudioEnabled = false;
                   audioPlayer.pause();
                   audioBtn.innerHTML = '<i class="fa-solid fa-volume-xmark"></i> 網頁音效: 關閉';
                   audioBtn.classList.remove('on');
               }
           }

           setInterval(() => {
               fetch('/api/status').then(r => r.json()).then(data => {
                   document.getElementById('val_stand').innerText = data.stand_count;
                   document.getElementById('val_sit').innerText = parseInt(data.total_sit);
                   document.getElementById('val_max').innerText = parseInt(data.max_sit);
                   
                   const balElem = document.getElementById('val_balance');
                   balElem.innerHTML = data.balance_score + '<br><span style="font-size:0.5em">' + data.balance_grade + '</span>';
                   balElem.style.color = data.balance_color;

                   if (webAudioEnabled) {
                       let targetSound = "";
                       if (data.is_emergency) targetSound = "/stream_emergency";
                       else if (data.trigger_sound) targetSound = "/stream_alarm";
                       
                       if (targetSound) {
                           if (currentSound !== targetSound) {
                               currentSound = targetSound;
                               audioPlayer.src = targetSound;
                               audioPlayer.play().catch(e => console.log(e));
                           } else if (audioPlayer.paused) {
                               audioPlayer.play().catch(e => console.log(e));
                           }
                       } else {
                           audioPlayer.pause();
                           currentSound = "";
                       }
                   } else {
                       if (!audioPlayer.paused) audioPlayer.pause();
                   }
               });
           }, 1000);
           
           setTimeout(function(){ location.reload(); }, 10000);

           function togglePrivacy() {
               let state = document.getElementById('privacyToggle').checked;
               fetch('/api/control/privacy', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({state: state}) });
           }
           function toggleSleep() {
               let state = document.getElementById('sleepToggle').checked;
               fetch('/api/control/sleep', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({state: state}) })
                    .then(r => alert('睡眠模式已' + (state ? '開啟' : '關閉')));
           }
           function resetEmergency() {
               fetch('/api/control/emergency_reset', { method: 'POST' }).then(r => alert('警報已解除'));
           }
           function updateAlertTime() {
               let val = document.getElementById('alertInput').value;
               fetch('/api/control/alert_time', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({val: parseFloat(val)}) })
                   .then(r => alert('設定已更新'));
           }
           function recalibrate() {
               if(confirm('確定要重新校正身高嗎？(需要重新站立)')) {
                   fetch('/api/control/recalibrate', { method: 'POST' }).then(r => alert('請移動到鏡頭前站立'));
               }
           }
           function hardReset() {
               if(confirm('⚠️ 警告：這將會清除今天所有的久坐數據！確定嗎？')) {
                   fetch('/api/control/hard_reset', { method: 'POST' }).then(r => alert('系統已重置歸零'));
               }
           }
        </script>
      </body>
    </html>
    """
    
    @app.after_request
    def add_header(r):
        r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        r.headers["Pragma"] = "no-cache"
        r.headers["Expires"] = "0"
        return r

    def generate_frames():
        while True:
            with lock:
                if outputFrame is None: continue
                (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
                if not flag: continue
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

    @app.route('/video_feed')
    def video_feed():
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

    
    @app.route('/stream_alarm')
    def stream_alarm():
        if os.path.exists(ALARM_SOUND_FILE):
            return send_file(ALARM_SOUND_FILE, mimetype="audio/mpeg")
        return "No Audio", 404

    @app.route('/stream_emergency')
    def stream_emergency():
        if os.path.exists(EMERGENCY_SOUND_FILE):
            return send_file(EMERGENCY_SOUND_FILE, mimetype="audio/mpeg")
        return "No Audio", 404

    @app.route('/api/status')
    def api_status():
        with lock:
            total_time = logger.data["Total_Sit_Seconds"]
            if logger.current_sit_start_ts and not GLOBAL_STATUS.get("is_rehab_mode", False):
                total_time += (time.time() - logger.current_sit_start_ts)
            
            is_sitting_in_rehab = (GLOBAL_STATUS["is_rehab_mode"] and GLOBAL_STATUS["balance_grade"] == "Sitting")
            is_emerg = GLOBAL_STATUS.get("is_emergency", False)

            return jsonify({
                "balance_score": GLOBAL_STATUS["balance_score"],
                "balance_grade": GLOBAL_STATUS["balance_grade"],
                "balance_color": GLOBAL_STATUS["balance_color"],
                "stand_count": logger.data["Stand_Count"],
                "total_sit": int(total_time),
                "max_sit": int(logger.data["Max_Sit_Seconds"]),
                "trigger_sound": is_sitting_in_rehab,
                "is_emergency": is_emerg
            })

    # API Endpoints
    @app.route('/api/control/privacy', methods=['POST'])
    def api_privacy():
        APP_CONFIG['privacy_mode'] = request.json['state']
        return jsonify({"status": "ok", "mode": APP_CONFIG['privacy_mode']})

    @app.route('/api/control/sound', methods=['POST'])
    def api_sound():
        APP_CONFIG['sound_alert'] = request.json['state']
        return jsonify({"status": "ok", "mode": APP_CONFIG['sound_alert']})

    @app.route('/api/control/sleep', methods=['POST'])
    def api_sleep():
        APP_CONFIG['sleep_mode'] = request.json['state']
        return jsonify({"status": "ok", "mode": APP_CONFIG['sleep_mode']})

    @app.route('/api/control/emergency_reset', methods=['POST'])
    def api_emergency_reset():
        APP_CONFIG['emergency_reset'] = True
        return jsonify({"status": "ok"})

    @app.route('/api/control/alert_time', methods=['POST'])
    def api_alert_time():
        APP_CONFIG['alert_interval'] = float(request.json['val'])
        return jsonify({"status": "ok", "val": APP_CONFIG['alert_interval']})

    @app.route('/api/control/recalibrate', methods=['POST'])
    def api_recalibrate():
        APP_CONFIG['force_recalibrate'] = True
        return jsonify({"status": "ok"})

    @app.route('/api/control/hard_reset', methods=['POST'])
    def api_hard_reset():
        APP_CONFIG['hard_reset'] = True
        return jsonify({"status": "ok"})

    @app.route("/")
    def index():
        try:
            data = logger.data
            events_raw = logger.read_events(10)
            events = []
            if events_raw:
                for r in reversed(events_raw):
                    if len(r) >= 4:
                        dur_sec = int(float(r[3]))
                        m, s = divmod(dur_sec, 60)
                        dur_str = f"{m}分 {s}秒"
                        r_new = (r[0], r[1], r[2], r[3], dur_str)
                        events.append(r_new)
            
            realtime_total_sit = data.get("Total_Sit_Seconds", 0.0)
            if logger.current_sit_start_ts is not None:
                realtime_total_sit += (time.time() - logger.current_sit_start_ts)

            current_max = data.get("Max_Sit_Seconds", 0.0)
            if logger.current_sit_start_ts is not None:
                current_duration = time.time() - logger.current_sit_start_ts
                current_max = max(current_max, current_duration)

            with lock:
                history = GLOBAL_STATUS["balance_history"]
                if len(history) > 0:
                    avg = int(sum(history) / len(history))
                    web_balance_score = f"{avg}"
                    web_balance_grade = f"Based on {len(history)} checks"
                    if avg >= 90: web_balance_color = "#20c997"
                    elif avg >= 70: web_balance_color = "#ffc107"
                    else: web_balance_color = "#dc3545"
                else:
                    web_balance_score = "--"
                    web_balance_grade = "Waiting data..."
                    web_balance_color = "#b2b2b2"

            return render_template_string(TEMPLATE, 
                                          config=APP_CONFIG,
                                          stand_count=data.get("Stand_Count", 0), 
                                          total_sit=int(realtime_total_sit), 
                                          max_sit=int(current_max), 
                                          events=events,
                                          balance_score=web_balance_score,
                                          balance_grade=web_balance_grade,
                                          balance_color=web_balance_color,
                                          hourly_data=logger.read_hourly_stats(), 
                                          daily_labels=logger.read_daily_history()[0], 
                                          daily_data=logger.read_daily_history()[1], 
                                          server_time=datetime.now().strftime("%H:%M:%S"))
        except Exception as e: return str(e)

    @app.route("/events.csv")
    def dl():
        if os.path.exists(logger.events_file): return send_from_directory(os.path.dirname(logger.events_file) or ".", os.path.basename(logger.events_file))
        return "No file"
    return app

flask_app = create_flask_app(logger)
def run_flask():
    try: flask_app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
    except: pass
Thread(target=run_flask, daemon=True).start()

# ================== Main ==================
def main():
    global outputFrame
    debug_vals = {"Knee": 0, "Hip": 0, "Ratio": 0.0}
    balance_color_text = (200, 200, 200)
    final_status = "unknown" 

    print(f"Connecting to {VIDEO_SOURCE}...")
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    
    if not cap.isOpened():
        print(f"Error: Cannot connect to {VIDEO_SOURCE}")
        return

    logger.log_first_seen()
    
    status_buffer = []
    last_stand_time = time.time()
    inactivity_start_time = None
    last_beep_time = 0.0
    ref_stand_y = None
    stand_y_buffer = []
    
    prev_kpts = None
    ema_motion_score = 0.0
    motion_status = "STILL"
    MOTION_JOINTS_IDX = [13, 14, 15, 16, 25, 26]
    
    last_active_time = time.time() 

    is_rehab_mode = False
    has_started_playing = False 
    rehab_wait_start_time = 0
    rehab_count = 0
    game_target = None 
    last_rehab_sound_time = 0

    is_celebrating = False
    celebration_end_time = 0
    
    is_emergency = False
    
    # Email 發送旗標
    email_sent = False

    shoulder_mid_x_buffer = []
    last_balance_calc_time = time.time()

    last_seen_time = time.time()
    
    smoother = PoseSmoother(alpha=0.6) 

    BODY_CONNECTIONS = [(start, end) for start, end in mp_pose.POSE_CONNECTIONS if start > 10 and end > 10]

    print(f"Start. Web: http://localhost:5000")

    try:
        while True:
            # 檢查睡眠模式
            if APP_CONFIG['sleep_mode']:
                if alert_sound: alert_sound.stop()
                if emergency_sound: emergency_sound.stop()
                if start_sound: start_sound.stop()
                if success_sound: success_sound.stop()

                last_active_time = time.time() 
                inactivity_start_time = None 
                logger.set_current_sitting(None) 
                is_emergency = False 
                email_sent = False   

                ret, frame = cap.read()
                if not ret:
                     cap.release()
                     time.sleep(2)
                     cap = cv2.VideoCapture(VIDEO_SOURCE)
                     continue
                
                frame = cv2.flip(frame, 1)
                H, W = frame.shape[:2]
                
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (W, H), (20, 20, 50), -1) 
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                cv2.putText(frame, "SLEEPING MODE (Zzz...)", (50, H//2), FONT, 1.5, (255, 255, 255), 3)

                with lock: outputFrame = frame.copy()
                cv2.imshow(WINDOW_NAME, frame)
                if cv2.waitKey(1) == ord('q'): break
                continue 

            # === 正常邏輯 ===
            
            if APP_CONFIG['emergency_reset']:
                print("Emergency Reset by User")
                is_emergency = False
                email_sent = False 
                is_rehab_mode = False
                has_started_playing = False
                rehab_count = 0
                game_target = None
                inactivity_start_time = time.time()
                logger.set_current_sitting(inactivity_start_time)
                if emergency_sound: emergency_sound.stop()
                APP_CONFIG['emergency_reset'] = False 
            
            with lock:
                GLOBAL_STATUS["is_emergency"] = is_emergency

            if len(particles) > 200: particles[:] = particles[-200:]

            ret, frame = cap.read()
            if not ret: 
                print("Stream lost. Reconnecting...")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(VIDEO_SOURCE)
                continue
            
            frame = cv2.flip(frame, 1)

            if APP_CONFIG['force_recalibrate']:
                print("Remote Recalibration Command Received.")
                ref_stand_y = None
                stand_y_buffer = []
                smoother.reset()
                APP_CONFIG['force_recalibrate'] = False
                is_rehab_mode = False 
                is_emergency = False
                email_sent = False
                shoulder_mid_x_buffer = []
                with lock:
                    GLOBAL_STATUS["balance_history"] = []

            if APP_CONFIG['hard_reset']:
                print("Remote HARD RESET Command Received.")
                ref_stand_y = None
                stand_y_buffer = []
                status_buffer = []
                inactivity_start_time = None
                is_rehab_mode = False
                is_emergency = False
                email_sent = False
                rehab_count = 0
                game_target = None
                smoother.reset()
                logger.hard_reset_today()
                shoulder_mid_x_buffer = []
                with lock:
                    GLOBAL_STATUS["balance_history"] = []
                APP_CONFIG['hard_reset'] = False

            H, W = frame.shape[:2]

            # --- 緊急狀態邏輯 ---
            if is_emergency:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (W, H), (0, 0, 255), -1) 
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                
                cv2.putText(frame, "EMERGENCY: POTENTIAL FAINTING!", (20, H//2), FONT, 1.2, (255, 255, 255), 3)
                cv2.putText(frame, "Check on the User Immediately!", (50, H//2 + 50), FONT, 0.8, (255, 255, 255), 2)
                
                if not pygame.mixer.get_busy() and emergency_sound: 
                    emergency_sound.play()
                
                if not email_sent:
                    print("Status: Emergency! Sending Email...")
                    Thread(target=send_emergency_email).start()
                    email_sent = True 

                with lock: outputFrame = frame.copy()
                cv2.imshow(WINDOW_NAME, frame)
                if cv2.waitKey(1) == ord('q'): break
                continue 

            # --- 慶祝模式邏輯 ---
            if is_celebrating:
                if time.time() > celebration_end_time:
                    is_celebrating = False
                    is_rehab_mode = False
                    has_started_playing = False 
                    rehab_count = 0
                    game_target = None
                    particles.clear()
                    inactivity_start_time = time.time()
                    logger.set_current_sitting(inactivity_start_time)
                    is_emergency = False
                    email_sent = False
                else:
                    if random.random() < 0.1: 
                        ex = random.randint(50, W-50)
                        ey = random.randint(50, H-100)
                        spawn_firework(ex, ey)
                    
                    for p in particles:
                        p.update()
                        p.draw(frame)
                    particles[:] = [p for p in particles if p.life > 0]
                    
                    cv2.putText(frame, "CONGRATULATIONS!", (W//2 - 250, H//2), FONT, 2, (0, 215, 255), 4)

                with lock: outputFrame = frame.copy()
                cv2.imshow(WINDOW_NAME, frame)
                if cv2.waitKey(1) == ord('q'): break
                continue

            # --- 正常邏輯 (選最大框) ---
            boxes = []
            if yolo:
                res = yolo.predict(frame, conf=YOLO_CONF, verbose=False)[0]
                if res.boxes:
                    for b in res.boxes:
                        x1,y1,x2,y2 = map(int, b.xyxy[0])
                        if (x2-x1)>MIN_ROI_SIZE: boxes.append((x1,y1,x2,y2, b.conf.item()))
            
            main_box = select_primary_person(boxes)
            
            status = "unknown"
            debug = ""
            is_confident_stand = False

            if main_box is None:
                if time.time() - last_seen_time > 3.0:
                    if ref_stand_y is not None:
                        print("User left. Resetting calibration.")
                        ref_stand_y = None 
                        stand_y_buffer = []
                        status_buffer = [] 
                        smoother.reset()
                        is_rehab_mode = False 
                        shoulder_mid_x_buffer = []
                        with lock:
                            GLOBAL_STATUS["balance_score"] = "--"
                            GLOBAL_STATUS["balance_grade"] = "No Person"
                            GLOBAL_STATUS["balance_color"] = "#b2b2b2"
            else:
                last_seen_time = time.time()

            if main_box:
                bx1, by1, bx2, by2, _ = main_box
                box_h, box_w = by2-by1, bx2-bx1
                box_ratio = box_h / box_w if box_w>0 else 0
                bx1e, by1e, bx2e, by2e = expand_box(bx1, by1, bx2, by2, W, H, ROI_EXPAND)
                roi = frame[by1e:by2e, bx1e:bx2e]
                
                if roi.size > 0:
                    res = pose.process(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                    
                    if res.pose_landmarks:
                        res.pose_landmarks = smoother.smooth(res.pose_landmarks)
                        
                        nose_vis = res.pose_landmarks.landmark[0].visibility
                        l_sh_vis = res.pose_landmarks.landmark[11].visibility
                        r_sh_vis = res.pose_landmarks.landmark[12].visibility
                        avg_vis = (nose_vis + l_sh_vis + r_sh_vis) / 3.0
                        
                        if avg_vis > HUMAN_CONFIDENCE_THRESHOLD:
                            # 畫骨架
                            for start_idx, end_idx in BODY_CONNECTIONS:
                                start_pt = res.pose_landmarks.landmark[start_idx]
                                end_pt = res.pose_landmarks.landmark[end_idx]
                                if start_pt.visibility > 0.5 and end_pt.visibility > 0.5:
                                    sx, sy = int(start_pt.x * (bx2e-bx1e)), int(start_pt.y * (by2e-by1e))
                                    ex, ey = int(end_pt.x * (bx2e-bx1e)), int(end_pt.y * (by2e-by1e))
                                    cv2.line(roi, (sx, sy), (ex, ey), (255, 255, 0), 3)

                            for idx, lm in enumerate(res.pose_landmarks.landmark):
                                if idx > 10 and lm.visibility > 0.5:
                                    cx, cy = int(lm.x * (bx2e-bx1e)), int(lm.y * (by2e-by1e))
                                    cv2.circle(roi, (cx, cy), 5, (0, 255, 255), -1)
                            
                            frame[by1e:by2e, bx1e:bx2e] = roi
                            
                            lm_list = []
                            current_kpts = [] 
                            
                            for i, p in enumerate(res.pose_landmarks.landmark):
                                full_x = (p.x*(bx2e-bx1e)+bx1e)/W
                                full_y = (p.y*(by2e-by1e)+by1e)/H
                                lm_list.append(type('LM', (object,), {'x':full_x, 'y':full_y, 'z':p.z, 'visibility':p.visibility}))
                                
                                if i in MOTION_JOINTS_IDX:
                                    current_kpts.append((full_x, full_y))

                            # === AR 體感遊戲邏輯 ===
                            if is_rehab_mode:
                                l_sh_y, r_sh_y = lm_list[11].y, lm_list[12].y
                                current_shoulder_y = (l_sh_y + r_sh_y) / 2
                                current_status, _, _, debug_vals = analyze_pose(res.pose_world_landmarks.landmark, lm_list, W, H, box_ratio, current_shoulder_y, ref_stand_y, box_h)
                                
                                if not has_started_playing:
                                    if current_status == "standing":
                                        has_started_playing = True 
                                        if alert_sound: alert_sound.stop()
                                        if start_sound and not pygame.mixer.get_busy(): start_sound.play()
                                    else:
                                        # 30秒未站立 -> 緊急狀態
                                        if time.time() - rehab_wait_start_time > WAIT_FOR_STAND_TIMEOUT:
                                            print("EMERGENCY TRIGGERED: User failed to stand up.")
                                            is_emergency = True
                                            if alert_sound: alert_sound.stop()
                                            continue 
                                        else:
                                            if time.time() - last_rehab_sound_time > 3.0:
                                                if alert_sound and not pygame.mixer.get_busy():
                                                    alert_sound.play()
                                                last_rehab_sound_time = time.time()

                                            overlay = frame.copy()
                                            cv2.rectangle(overlay, (0, 0), (W, H), (0, 0, 100), -1) 
                                            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
                                            remaining_time = int(WAIT_FOR_STAND_TIMEOUT - (time.time() - rehab_wait_start_time))
                                            cv2.putText(frame, f"PLEASE STAND UP! ({remaining_time}s)", (30, 200), FONT, 1.5, (0, 0, 255), 3)
                                            with lock: GLOBAL_STATUS["balance_grade"] = "Sitting"

                                if has_started_playing:
                                    with lock: GLOBAL_STATUS["balance_grade"] = "Playing"

                                    for p in particles:
                                        p.update()
                                        p.draw(frame)
                                    particles[:] = [p for p in particles if p.life > 0]

                                    if game_target is None:
                                        margin_x = int(box_w * 0.6) 
                                        min_x = max(0, bx1 - margin_x)
                                        max_x = min(W, bx2 + margin_x)
                                        min_y = max(0, int(by1 - box_h * 0.1)) 
                                        max_y = min(H, int(by1 + box_h * 0.6))

                                        tgt_x = random.randint(min_x, max_x)
                                        tgt_y = random.randint(min_y, max_y)
                                        game_target = (tgt_x, tgt_y)

                                    cv2.circle(frame, game_target, 30, (0, 255, 255), -1)
                                    cv2.circle(frame, game_target, 30, (0, 0, 255), 2)
                                    cv2.putText(frame, "TOUCH!", (game_target[0]-40, game_target[1]-40), FONT, 0.8, (0, 255, 255), 2)

                                    l_wrist = (int(lm_list[15].x * W), int(lm_list[15].y * H))
                                    r_wrist = (int(lm_list[16].x * W), int(lm_list[16].y * H))
                                    
                                    hit = False
                                    if math.hypot(l_wrist[0]-game_target[0], l_wrist[1]-game_target[1]) < 50: hit = True
                                    if math.hypot(r_wrist[0]-game_target[0], r_wrist[1]-game_target[1]) < 50: hit = True
                                    
                                    if hit:
                                        rehab_count += 1
                                        spawn_hit_effect(game_target[0], game_target[1])
                                        game_target = None
                                        if success_sound and not pygame.mixer.get_busy():
                                            success_sound.play()
                                    
                                    cv2.putText(frame, f"Score: {rehab_count} / {REHAB_TARGET_COUNT}", (20, 100), FONT, 1.5, (0, 255, 0), 3)

                                if rehab_count >= REHAB_TARGET_COUNT:
                                    print("Rehab Complete! Celebration!")
                                    is_celebrating = True
                                    celebration_end_time = time.time() + 5

                            # === 一般監測邏輯 ===
                            else:
                                if prev_kpts is not None and len(prev_kpts) == len(current_kpts):
                                    dist_sum = 0
                                    for (cx, cy), (px, py) in zip(current_kpts, prev_kpts):
                                        dist_sum += math.hypot(cx-px, cy-py)
                                    avg_dist = dist_sum / len(current_kpts) if len(current_kpts) > 0 else 0
                                    ema_motion_score = MOTION_EMA_ALPHA * avg_dist + (1 - MOTION_EMA_ALPHA) * ema_motion_score
                                    if ema_motion_score > MOTION_THRESHOLD: 
                                        motion_status = "MOVING"
                                        last_active_time = time.time() # 活動中，重置
                                    else: 
                                        motion_status = "STILL"
                                prev_kpts = current_kpts

                                l_sh_y = lm_list[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
                                r_sh_y = lm_list[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
                                current_shoulder_y = (l_sh_y + r_sh_y) / 2

                                status, debug, is_confident_stand, debug_vals = analyze_pose(res.pose_world_landmarks.landmark, lm_list, W, H, box_ratio, current_shoulder_y, ref_stand_y, box_h)

                                if final_status == "standing" and is_confident_stand:
                                    stand_y_buffer.append(current_shoulder_y)
                                    if len(stand_y_buffer) > 30: 
                                        stand_y_buffer.pop(0)
                                        ref_stand_y = sum(stand_y_buffer) / len(stand_y_buffer)

                            if APP_CONFIG['privacy_mode'] and len(lm_list) > 32:
                                try:
                                    head_indices = [0, 3, 6, 7, 8, 9, 10]
                                    head_x_coords = []
                                    head_y_coords = []
                                    for idx in head_indices:
                                        if idx < len(lm_list):
                                            px = int(lm_list[idx].x * W)
                                            py = int(lm_list[idx].y * H)
                                            head_x_coords.append(px)
                                            head_y_coords.append(py)
                                    if head_x_coords and head_y_coords:
                                        min_x, max_x = min(head_x_coords), max(head_x_coords)
                                        min_y, max_y = min(head_y_coords), max(head_y_coords)
                                        curr_head_w = max_x - min_x
                                        curr_head_h = max_y - min_y
                                        pad_w = int(curr_head_w * 0.4)  
                                        pad_h_top = int(curr_head_h * 0.8) 
                                        pad_h_bottom = int(curr_head_h * 0.3)
                                        fx1 = max(0, min_x - pad_w)
                                        fy1 = max(0, min_y - pad_h_top)
                                        fx2 = min(W, max_x + pad_w)
                                        fy2 = min(H, max_y + pad_h_bottom)
                                        if fx2 > fx1 and fy2 > fy1:
                                            face_roi = frame[fy1:fy2, fx1:fx2]
                                            face_roi = cv2.GaussianBlur(face_roi, (151, 151), 50)
                                            frame[fy1:fy2, fx1:fx2] = face_roi
                                except: pass
                        else:
                            debug_str = f"Box R:{box_ratio:.1f}"
                            if box_ratio > BOX_RATIO_THRESHOLD: 
                                status = "standing"
                            else: 
                                status = "sitting"
                            debug_vals = {"Knee": 0, "Hip": 0, "Ratio": 0.0}

            if status != "unknown":
                status_buffer.append(status)
                if len(status_buffer) > BUFFER_SIZE: status_buffer.pop(0)
                final_status = max(set(status_buffer), key=status_buffer.count)
            elif main_box is None and time.time() - last_seen_time > 2.0:
                final_status = "no_person"
            else:
                if 'final_status' not in locals(): final_status = "unknown"

            if ref_stand_y is not None:
                cv2.putText(frame, "System Ready", (W-180, 30), FONT, 0.7, (0, 255, 0), 2)
            else:
                loading_dots = "." * (int(time.time() * 2) % 4)
                cv2.putText(frame, f"Please STAND to Calibrate{loading_dots}", (W-310, 30), FONT, 0.7, (0, 0, 255), 2)
                cv2.rectangle(frame, (W-300, 40), (W-50, 50), (100, 100, 100), 1)
                if len(stand_y_buffer) > 0:
                    prog_w = int(250 * (len(stand_y_buffer) / 30))
                    cv2.rectangle(frame, (W-300, 40), (W-300+prog_w, 50), (0, 255, 255), -1)

            if is_rehab_mode and main_box is None:
                 overlay = frame.copy()
                 cv2.rectangle(overlay, (0, 0), (W, H), (0, 0, 0), -1) 
                 cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                 cv2.putText(frame, "NO PERSON DETECTED", (50, H//2), FONT, 1.5, (0, 0, 255), 3)

            elif not is_rehab_mode:
                if final_status == "standing":
                    last_stand_time = time.time()
                    if inactivity_start_time:
                        dur = time.time() - inactivity_start_time
                        if dur >= MIN_SIT_DURATION: logger.log_stand_up(inactivity_start_time, time.time(), dur)
                        inactivity_start_time = None
                    
                    if motion_status == "STILL" and main_box:
                        sh_mid_x = (lm_list[11].x + lm_list[12].x) / 2.0
                        shoulder_mid_x_buffer.append(sh_mid_x)
                        
                        if time.time() - last_balance_calc_time > 3.0:
                            if len(shoulder_mid_x_buffer) > 20: 
                                sway_std = np.std(shoulder_mid_x_buffer)
                                if sway_std < 0.005:
                                    score = 95 + random.randint(0, 5)
                                    balance_grade = "Excellent"
                                    balance_color_text = (0, 255, 0)
                                    web_color = "#20c997"
                                elif sway_std < 0.015:
                                    score = 80 + random.randint(0, 10)
                                    balance_grade = "Fair"
                                    balance_color_text = (0, 255, 255)
                                    web_color = "#ffc107"
                                else:
                                    score = max(0, 60 - int((sway_std - 0.015) * 2000))
                                    balance_grade = "Poor (Risk)"
                                    balance_color_text = (0, 0, 255)
                                    web_color = "#dc3545"
                                
                                with lock:
                                    GLOBAL_STATUS["balance_score"] = str(score)
                                    GLOBAL_STATUS["balance_grade"] = balance_grade
                                    GLOBAL_STATUS["balance_color"] = web_color
                                    GLOBAL_STATUS["balance_history"].append(score)
                            
                            shoulder_mid_x_buffer = []
                            last_balance_calc_time = time.time()
                    else:
                        shoulder_mid_x_buffer = []
                        with lock:
                            GLOBAL_STATUS["balance_score"] = "--"
                            GLOBAL_STATUS["balance_grade"] = "Moving..."
                            GLOBAL_STATUS["balance_color"] = "#b2b2b2"

                elif final_status == "sitting":
                    shoulder_mid_x_buffer = []
                    with lock:
                        GLOBAL_STATUS["balance_score"] = "--"
                        GLOBAL_STATUS["balance_grade"] = "Sitting"
                        GLOBAL_STATUS["balance_color"] = "#b2b2b2"

                    if (time.time() - last_stand_time) > WALKING_BUFFER_TIME:
                        if not inactivity_start_time: 
                            inactivity_start_time = time.time()
                            logger.set_current_sitting(inactivity_start_time)

                        elapsed = time.time() - inactivity_start_time
                        
                        if elapsed >= APP_CONFIG['alert_interval']:
                            if not is_rehab_mode and ref_stand_y is not None:
                                print("Alert! Entering Rehab Mode.")
                                if alert_sound and not pygame.mixer.get_busy():
                                    alert_sound.play()
                                is_rehab_mode = True
                                has_started_playing = False
                                rehab_wait_start_time = time.time() # 開始計時
                                rehab_count = 0
                                game_target = None 
                    else: final_status = "standing (buffer)"
                else: 
                    inactivity_start_time = None
                    logger.set_current_sitting(None)
                    shoulder_mid_x_buffer = []
                    with lock:
                        GLOBAL_STATUS["balance_score"] = "--"
                        GLOBAL_STATUS["balance_grade"] = "--"
                        GLOBAL_STATUS["balance_color"] = "#b2b2b2"

                color = (0,255,0) if "standing" in final_status else (0,0,255)
                
                if main_box and final_status != "no_person":
                    cv2.rectangle(frame, (bx1,by1), (bx2,by2), color, 2)
                    
                    # 【新增】 一般監測模式的長時間無反應偵測
                    if time.time() - last_active_time > UNCONSCIOUS_TIME_THRESHOLD:
                         print("EMERGENCY TRIGGERED: No movement for too long.")
                         is_emergency = True
                         continue

                cv2.putText(frame, f"Status: {final_status}", (20,40), FONT, 1, color, 2)
                motion_color = (0, 255, 0) if motion_status == "MOVING" else (0, 165, 255)
                cv2.putText(frame, f"Activity: {motion_status}", (20,80), FONT, 0.8, motion_color, 2)
                
                d_str = f"K:{debug_vals['Knee']} H:{debug_vals['Hip']} R:{debug_vals['Ratio']}"
                cv2.putText(frame, d_str, (20,120), FONT, 0.6, (200,200,200), 1)

                if is_rehab_mode:
                     sit_sec = int(APP_CONFIG['alert_interval'])
                else:
                     sit_sec = int(time.time()-inactivity_start_time) if inactivity_start_time else 0
                
                cv2.putText(frame, f"Sitting: {sit_sec}s", (20,160), FONT, 0.8, (0,165,255), 2)
                
                if final_status == "standing":
                    with lock:
                        disp_score = GLOBAL_STATUS["balance_score"]
                        disp_grade = GLOBAL_STATUS["balance_grade"]
                    cv2.putText(frame, f"Balance: {disp_score} ({disp_grade})", (50, H - 50), FONT, 0.8, balance_color_text, 2)

            with lock:
                GLOBAL_STATUS["is_rehab_mode"] = is_rehab_mode

            with lock:
                outputFrame = frame.copy()

            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) == ord('q'): break
    
    except KeyboardInterrupt:
        print("\nSaving last data...")
        if inactivity_start_time:
            dur = time.time() - inactivity_start_time
            if dur >= MIN_SIT_DURATION: logger.log_stand_up(inactivity_start_time, time.time(), dur)

    cap.release()
    cv2.destroyAllWindows()
    print("Video stopped. Web server running...")
    while True: time.sleep(1)

if __name__ == "__main__":
    main()
