# main.py
import cv2
import time
import mediapipe as mp
import pygame
import numpy as np
import threading
import warnings
import os

import config
import utils
from logger import ActivityLogger
from pose_logic import analyze_pose
import web_server
from shared_state import lock, outputFrame, APP_CONFIG, GLOBAL_STATUS

# 忽略警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')

# 初始化 YOLO
try:
    from ultralytics import YOLO
    yolo = YOLO(config.YOLO_MODEL)
except Exception as e:
    yolo = None
    print("YOLO Load Error:", e)

# 初始化 MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7, model_complexity=1, smooth_landmarks=True)

# 初始化 Logger
logger = ActivityLogger(config.CSV_FILE)

# 初始化音效
try:
    pygame.mixer.init()
    if os.path.exists(config.ALARM_SOUND_FILE): 
        alert_sound = pygame.mixer.Sound(config.ALARM_SOUND_FILE)
    else: alert_sound = None

    if os.path.exists(config.SUCCESS_SOUND_FILE):
        success_sound = pygame.mixer.Sound(config.SUCCESS_SOUND_FILE)
    else: success_sound = None

    if os.path.exists(config.START_SOUND_FILE):
        start_sound = pygame.mixer.Sound(config.START_SOUND_FILE)
    else: start_sound = None

    if os.path.exists(config.EMERGENCY_SOUND_FILE):
        emergency_sound = pygame.mixer.Sound(config.EMERGENCY_SOUND_FILE)
    else: emergency_sound = None
except:
    print("Audio Init Failed")
    alert_sound = None
    success_sound = None
    start_sound = None
    emergency_sound = None

def main():
    global outputFrame
    
    # 啟動 Web Server 執行緒
    threading.Thread(target=web_server.run_flask, args=(logger,), daemon=True).start()
    
    debug_vals = {"Knee": 0, "Hip": 0, "Ratio": 0.0}
    balance_color_text = (200, 200, 200)
    final_status = "unknown"

    print(f"Connecting to {config.VIDEO_SOURCE}...")
    cap = cv2.VideoCapture(config.VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Error: Cannot connect to {config.VIDEO_SOURCE}")
        return

    logger.log_first_seen()

    # 狀態變數
    status_buffer = []
    last_stand_time = time.time()
    inactivity_start_time = None
    ref_stand_y = None
    stand_y_buffer = []
    prev_kpts = None
    ema_motion_score = 0.0
    motion_status = "STILL"
    MOTION_JOINTS_IDX = [13, 14, 15, 16, 25, 26]
    last_active_time = time.time()

    # 復健與緊急狀態
    is_rehab_mode = False
    has_started_playing = False
    rehab_wait_start_time = 0
    rehab_count = 0
    game_target = None
    last_rehab_sound_time = 0
    is_celebrating = False
    celebration_end_time = 0
    
    is_emergency = False
    email_sent = False
    
    shoulder_mid_x_buffer = []
    last_balance_calc_time = time.time()
    last_seen_time = time.time()
    
    smoother = utils.PoseSmoother(alpha=0.6)
    BODY_CONNECTIONS = [(start, end) for start, end in mp_pose.POSE_CONNECTIONS if start > 10 and end > 10]

    print(f"Start. Web: http://localhost:5000")

    try:
        while True:
            # 1. 檢查睡眠模式
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
                     cap.release(); time.sleep(2); cap = cv2.VideoCapture(config.VIDEO_SOURCE); continue
                
                frame = cv2.flip(frame, 1)
                H, W = frame.shape[:2]
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (W, H), (20, 20, 50), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                cv2.putText(frame, "SLEEPING MODE (Zzz...)", (50, H//2), config.FONT, 1.5, (255, 255, 255), 3)

                with lock: outputFrame = frame.copy()
                cv2.imshow(config.WINDOW_NAME, frame)
                if cv2.waitKey(1) == ord('q'): break
                continue

            # 2. 檢查緊急重置
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
            
            with lock: GLOBAL_STATUS["is_emergency"] = is_emergency

            # 3. 讀取影像
            if len(utils.particles) > 200: utils.particles[:] = utils.particles[-200:]
            ret, frame = cap.read()
            if not ret:
                print("Stream lost. Reconnecting..."); cap.release(); time.sleep(2); cap = cv2.VideoCapture(config.VIDEO_SOURCE); continue
            
            frame = cv2.flip(frame, 1)
            H, W = frame.shape[:2]

            # 4. 處理校正與重置
            if APP_CONFIG['force_recalibrate']:
                print("Recalibrating..."); ref_stand_y = None; stand_y_buffer = []; smoother.reset(); APP_CONFIG['force_recalibrate'] = False
                is_rehab_mode = False; is_emergency = False; email_sent = False
            
            if APP_CONFIG['hard_reset']:
                print("Hard Resetting..."); ref_stand_y = None; stand_y_buffer = []; status_buffer = []; inactivity_start_time = None
                is_rehab_mode = False; is_emergency = False; email_sent = False; rehab_count = 0; smoother.reset(); logger.hard_reset_today()
                APP_CONFIG['hard_reset'] = False

            # 5. 緊急狀態邏輯 (最高優先級)
            if is_emergency:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (W, H), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                cv2.putText(frame, "EMERGENCY: POTENTIAL FAINTING!", (40, H//2), config.FONT, 1.2, (255, 255, 255), 3)
                cv2.putText(frame, "Check on the User Immediately!", (50, H//2 + 50), config.FONT, 0.8, (255, 255, 255), 2)
                
                if not pygame.mixer.get_busy() and emergency_sound: emergency_sound.play()
                
                if not email_sent:
                    print("Status: Emergency! Sending Email...")
                    threading.Thread(target=utils.send_emergency_email).start()
                    email_sent = True

                with lock: outputFrame = frame.copy()
                cv2.imshow(config.WINDOW_NAME, frame)
                if cv2.waitKey(1) == ord('q'): break
                continue

            # 6. 慶祝邏輯
            if is_celebrating:
                if time.time() > celebration_end_time:
                    is_celebrating = False; is_rehab_mode = False; has_started_playing = False; rehab_count = 0; game_target = None; utils.particles.clear()
                    inactivity_start_time = time.time(); logger.set_current_sitting(inactivity_start_time); is_emergency = False; email_sent = False
                else:
                    if random.random() < 0.1: 
                        ex = random.randint(50, W-50); ey = random.randint(50, H-100); utils.spawn_firework(ex, ey)
                    for p in utils.particles: p.update(); p.draw(frame)
                    utils.particles[:] = [p for p in utils.particles if p.life > 0]
                    cv2.putText(frame, "CONGRATULATIONS!", (W//2 - 250, H//2), config.FONT, 2, (0, 215, 255), 4)
                
                with lock: outputFrame = frame.copy()
                cv2.imshow(config.WINDOW_NAME, frame)
                if cv2.waitKey(1) == ord('q'): break
                continue

            # 7. YOLO 偵測
            boxes = []
            if yolo:
                res = yolo.predict(frame, conf=config.YOLO_CONF, verbose=False)[0]
                if res.boxes:
                    for b in res.boxes:
                        x1,y1,x2,y2 = map(int, b.xyxy[0])
                        if (x2-x1) > config.MIN_ROI_SIZE: boxes.append((x1,y1,x2,y2, b.conf.item()))
            
            main_box = utils.select_primary_person(boxes)
            status = "unknown"
            is_confident_stand = False

            if main_box is None:
                if time.time() - last_seen_time > 3.0:
                    if ref_stand_y is not None:
                        print("User left."); ref_stand_y = None; is_rehab_mode = False; shoulder_mid_x_buffer = []
                        with lock: GLOBAL_STATUS["balance_score"] = "--"; GLOBAL_STATUS["balance_grade"] = "No Person"
            else:
                last_seen_time = time.time()

            if main_box:
                bx1, by1, bx2, by2, _ = main_box
                box_h, box_w = by2-by1, bx2-bx1
                box_ratio = box_h / box_w if box_w>0 else 0
                bx1e, by1e, bx2e, by2e = utils.expand_box(bx1, by1, bx2, by2, W, H, config.ROI_EXPAND)
                roi = frame[by1e:by2e, bx1e:bx2e]
                
                if roi.size > 0:
                    res = pose.process(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                    if res.pose_landmarks:
                        res.pose_landmarks = smoother.smooth(res.pose_landmarks)
                        
                        nose_vis = res.pose_landmarks.landmark[0].visibility
                        l_sh_vis = res.pose_landmarks.landmark[11].visibility
                        r_sh_vis = res.pose_landmarks.landmark[12].visibility
                        avg_vis = (nose_vis + l_sh_vis + r_sh_vis) / 3.0
                        
                        if avg_vis > config.HUMAN_CONFIDENCE_THRESHOLD:
                            # 畫骨架
                            for start_idx, end_idx in BODY_CONNECTIONS:
                                start_pt = res.pose_landmarks.landmark[start_idx]; end_pt = res.pose_landmarks.landmark[end_idx]
                                if start_pt.visibility > 0.5 and end_pt.visibility > 0.5:
                                    sx, sy = int(start_pt.x * (bx2e-bx1e)), int(start_pt.y * (by2e-by1e))
                                    ex, ey = int(end_pt.x * (bx2e-bx1e)), int(end_pt.y * (by2e-by1e))
                                    cv2.line(roi, (sx, sy), (ex, ey), (255, 255, 0), 3)
                            
                            lm_list = []
                            current_kpts = []
                            for i, p in enumerate(res.pose_landmarks.landmark):
                                full_x = (p.x*(bx2e-bx1e)+bx1e)/W
                                full_y = (p.y*(by2e-by1e)+by1e)/H
                                lm_list.append(type('LM', (object,), {'x':full_x, 'y':full_y, 'z':p.z, 'visibility':p.visibility}))
                                if i in MOTION_JOINTS_IDX: current_kpts.append((full_x, full_y))
                            
                            frame[by1e:by2e, bx1e:bx2e] = roi # 將畫好的 ROI 貼回

                            # --- 8. 復健遊戲邏輯 ---
                            if is_rehab_mode:
                                l_sh_y = lm_list[11].y; r_sh_y = lm_list[12].y; current_shoulder_y = (l_sh_y + r_sh_y) / 2
                                current_status, _, _, debug_vals = analyze_pose(res.pose_world_landmarks.landmark, lm_list, W, H, box_ratio, current_shoulder_y, ref_stand_y, box_h)
                                
                                if not has_started_playing:
                                    if current_status == "standing":
                                        has_started_playing = True 
                                        if alert_sound: alert_sound.stop()
                                        if start_sound and not pygame.mixer.get_busy(): start_sound.play()
                                    else:
                                        if time.time() - rehab_wait_start_time > config.WAIT_FOR_STAND_TIMEOUT:
                                            print("EMERGENCY: User failed to stand."); is_emergency = True; continue 
                                        else:
                                            if time.time() - last_rehab_sound_time > 3.0:
                                                if alert_sound and not pygame.mixer.get_busy(): alert_sound.play()
                                                last_rehab_sound_time = time.time()
                                            
                                            overlay = frame.copy()
                                            cv2.rectangle(overlay, (0, 0), (W, H), (0, 0, 100), -1)
                                            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
                                            remaining_time = int(config.WAIT_FOR_STAND_TIMEOUT - (time.time() - rehab_wait_start_time))
                                            cv2.putText(frame, f"PLEASE STAND UP! ({remaining_time}s)", (30, 200), config.FONT, 1.5, (0, 0, 255), 3)
                                            with lock: GLOBAL_STATUS["balance_grade"] = "Sitting"

                                if has_started_playing:
                                    with lock: GLOBAL_STATUS["balance_grade"] = "Playing"
                                    for p in utils.particles: p.update(); p.draw(frame)
                                    utils.particles[:] = [p for p in utils.particles if p.life > 0]
                                    
                                    if game_target is None:
                                        margin_x = int(box_w * 0.6); min_x = max(0, bx1 - margin_x); max_x = min(W, bx2 + margin_x)
                                        min_y = max(0, int(by1 - box_h * 0.1)); max_y = min(H, int(by1 + box_h * 0.6))
                                        game_target = (random.randint(min_x, max_x), random.randint(min_y, max_y))

                                    cv2.circle(frame, game_target, 30, (0, 255, 255), -1); cv2.circle(frame, game_target, 30, (0, 0, 255), 2)
                                    cv2.putText(frame, "TOUCH!", (game_target[0]-40, game_target[1]-40), config.FONT, 0.8, (0, 255, 255), 2)

                                    l_wrist = (int(lm_list[15].x * W), int(lm_list[15].y * H))
                                    r_wrist = (int(lm_list[16].x * W), int(lm_list[16].y * H))
                                    
                                    hit = False
                                    if math.hypot(l_wrist[0]-game_target[0], l_wrist[1]-game_target[1]) < 50: hit = True
                                    if math.hypot(r_wrist[0]-game_target[0], r_wrist[1]-game_target[1]) < 50: hit = True
                                    
                                    if hit:
                                        rehab_count += 1; utils.spawn_hit_effect(game_target[0], game_target[1]); game_target = None
                                        if success_sound and not pygame.mixer.get_busy(): success_sound.play()
                                    
                                    cv2.putText(frame, f"Score: {rehab_count} / {config.REHAB_TARGET_COUNT}", (20, 100), config.FONT, 1.5, (0, 255, 0), 3)

                                if rehab_count >= config.REHAB_TARGET_COUNT:
                                    print("Rehab Complete!"); is_celebrating = True; celebration_end_time = time.time() + 5

                            # --- 9. 一般監測邏輯 ---
                            else:
                                if prev_kpts is not None and len(prev_kpts) == len(current_kpts):
                                    dist_sum = 0
                                    for (cx, cy), (px, py) in zip(current_kpts, prev_kpts): dist_sum += math.hypot(cx-px, cy-py)
                                    avg_dist = dist_sum / len(current_kpts) if len(current_kpts) > 0 else 0
                                    ema_motion_score = config.MOTION_EMA_ALPHA * avg_dist + (1 - config.MOTION_EMA_ALPHA) * ema_motion_score
                                    if ema_motion_score > config.MOTION_THRESHOLD: motion_status = "MOVING"; last_active_time = time.time()
                                    else: motion_status = "STILL"
                                prev_kpts = current_kpts

                                l_sh_y = lm_list[11].y; r_sh_y = lm_list[12].y; current_shoulder_y = (l_sh_y + r_sh_y) / 2
                                status, debug, is_confident_stand, debug_vals = analyze_pose(res.pose_world_landmarks.landmark, lm_list, W, H, box_ratio, current_shoulder_y, ref_stand_y, box_h)

                                if final_status == "standing" and is_confident_stand:
                                    stand_y_buffer.append(current_shoulder_y)
                                    if len(stand_y_buffer) > 30: stand_y_buffer.pop(0); ref_stand_y = sum(stand_y_buffer) / len(stand_y_buffer)

                            if APP_CONFIG['privacy_mode'] and len(lm_list) > 10:
                                try:
                                    head_x = [int(lm_list[i].x * W) for i in [0,3,6,7,8,9,10] if i < len(lm_list)]
                                    head_y = [int(lm_list[i].y * H) for i in [0,3,6,7,8,9,10] if i < len(lm_list)]
                                    if head_x and head_y:
                                        min_x, max_x = min(head_x), max(head_x); min_y, max_y = min(head_y), max(head_y)
                                        pad_w = int((max_x-min_x)*0.4); pad_h = int((max_y-min_y)*0.8)
                                        fx1 = max(0, min_x - pad_w); fy1 = max(0, min_y - pad_h)
                                        fx2 = min(W, max_x + pad_w); fy2 = min(H, max_y + int(pad_h*0.3))
                                        face_roi = frame[fy1:fy2, fx1:fx2]
                                        frame[fy1:fy2, fx1:fx2] = cv2.GaussianBlur(face_roi, (151, 151), 50)
                                except: pass
                        else:
                            if box_ratio > config.BOX_RATIO_THRESHOLD: status = "standing"
                            else: status = "sitting"
            
            # 10. 狀態更新與顯示
            if status != "unknown":
                status_buffer.append(status)
                if len(status_buffer) > config.BUFFER_SIZE: status_buffer.pop(0)
                final_status = max(set(status_buffer), key=status_buffer.count)
            elif main_box is None and time.time() - last_seen_time > 2.0:
                final_status = "no_person"
            
            if ref_stand_y is not None: cv2.putText(frame, "System Ready", (W-180, 30), config.FONT, 0.7, (0, 255, 0), 2)
            else: cv2.putText(frame, "Please STAND to Calibrate...", (W-310, 30), config.FONT, 0.7, (0, 0, 255), 2)

            if is_rehab_mode and main_box is None:
                 overlay = frame.copy(); cv2.rectangle(overlay, (0, 0), (W, H), (0, 0, 0), -1)
                 cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame); cv2.putText(frame, "NO PERSON DETECTED", (50, H//2), config.FONT, 1.5, (0, 0, 255), 3)

            elif not is_rehab_mode:
                if final_status == "standing":
                    last_stand_time = time.time()
                    if inactivity_start_time:
                        if (time.time() - inactivity_start_time) >= config.MIN_SIT_DURATION:
                            logger.log_stand_up(inactivity_start_time, time.time(), time.time() - inactivity_start_time)
                        inactivity_start_time = None
                    
                    if motion_status == "STILL" and main_box:
                        shoulder_mid_x_buffer.append((lm_list[11].x + lm_list[12].x) / 2.0)
                        if time.time() - last_balance_calc_time > 3.0:
                            if len(shoulder_mid_x_buffer) > 20:
                                sway = np.std(shoulder_mid_x_buffer)
                                if sway < 0.005: score = 95; grade = "Excellent"; col = "#20c997"
                                elif sway < 0.015: score = 80; grade = "Fair"; col = "#ffc107"
                                else: score = 60; grade = "Poor"; col = "#dc3545"
                                with lock: GLOBAL_STATUS.update({"balance_score": str(score), "balance_grade": grade, "balance_color": col})
                                shoulder_mid_x_buffer = []
                                last_balance_calc_time = time.time()
                    else:
                        with lock: GLOBAL_STATUS.update({"balance_score": "--", "balance_grade": "Moving...", "balance_color": "#b2b2b2"})

                elif final_status == "sitting":
                    with lock: GLOBAL_STATUS.update({"balance_score": "--", "balance_grade": "Sitting", "balance_color": "#b2b2b2"})
                    if (time.time() - last_stand_time) > config.WALKING_BUFFER_TIME:
                        if not inactivity_start_time:
                            inactivity_start_time = time.time(); logger.set_current_sitting(inactivity_start_time)
                        
                        if (time.time() - inactivity_start_time) >= config.ALERT_INTERVAL:
                            if not is_rehab_mode and ref_stand_y is not None:
                                print("Alert! Entering Rehab Mode."); is_rehab_mode = True; has_started_playing = False; rehab_count = 0; rehab_wait_start_time = time.time()
                                if alert_sound: alert_sound.play()
                    else: final_status = "standing (buffer)"

                if main_box and final_status != "no_person":
                     if time.time() - last_active_time > config.UNCONSCIOUS_TIME_THRESHOLD:
                         print("EMERGENCY: No movement."); is_emergency = True; continue

                cv2.putText(frame, f"Status: {final_status}", (20,40), config.FONT, 1, (0,255,0) if "standing" in final_status else (0,0,255), 2)
                cv2.putText(frame, f"Sitting: {int(time.time()-inactivity_start_time) if inactivity_start_time else 0}s", (20,160), config.FONT, 0.8, (0,255,255), 2)

            with lock: GLOBAL_STATUS["is_rehab_mode"] = is_rehab_mode
            with lock: outputFrame = frame.copy()
            cv2.imshow(config.WINDOW_NAME, frame)
            if cv2.waitKey(1) == ord('q'): break
            
    except KeyboardInterrupt:
        print("\nSaving...")
        if inactivity_start_time: logger.log_stand_up(inactivity_start_time, time.time(), time.time() - inactivity_start_time)

    cap.release()
    cv2.destroyAllWindows()
    print("System Shutdown.")

if __name__ == "__main__":
    main()
