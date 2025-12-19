# utils.py
import numpy as np
import smtplib
import random
import cv2
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import config

# === Email 模組 ===
def send_emergency_email():
    try:
        msg = MIMEMultipart()
        msg['From'] = config.SENDER_EMAIL
        msg['To'] = config.RECEIVER_EMAIL
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

        with smtplib.SMTP_SSL(config.SMTP_SERVER, config.SMTP_PORT) as server:
            server.login(config.SENDER_EMAIL, config.SENDER_PASSWORD)
            server.send_message(msg)
        
        print(f"✅ Email Sent to {config.RECEIVER_EMAIL}")
    except Exception as e:
        print(f"❌ Email Failed: {e}")

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

# === 骨架平滑器 ===
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

# === 特效 ===
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
