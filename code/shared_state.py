# shared_state.py
import threading

# 執行緒鎖
lock = threading.Lock()

# 影像緩衝區
outputFrame = None

# 全域系統設定 (可變動)
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

# 全域狀態監測
GLOBAL_STATUS = {
    "is_rehab_mode": False,
    "balance_score": "--",
    "balance_grade": "Waiting...",
    "balance_color": "#b2b2b2",
    "balance_history": [],
    "is_emergency": False
}
