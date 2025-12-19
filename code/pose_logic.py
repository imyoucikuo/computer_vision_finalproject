# pose_logic.py
import mediapipe as mp
import utils
import config

mp_pose = mp.solutions.pose

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

        # 舉手
        if (l_wrist.visibility > 0.5 and l_wrist.y < l_sh.y) or \
           (r_wrist.visibility > 0.5 and r_wrist.y < r_sh.y):
            return "standing", "Hands Up!", True, debug_vals

        # 3D 座標獲取 
        l_sh_3d = [world_landmarks[11].x, world_landmarks[11].y, world_landmarks[11].z]
        l_hip_3d = [world_landmarks[23].x, world_landmarks[23].y, world_landmarks[23].z]
        l_knee_3d = [world_landmarks[25].x, world_landmarks[25].y, world_landmarks[25].z]
        l_ankle_3d = [world_landmarks[27].x, world_landmarks[27].y, world_landmarks[27].z]
        
        r_sh_3d = [world_landmarks[12].x, world_landmarks[12].y, world_landmarks[12].z]
        r_hip_3d = [world_landmarks[24].x, world_landmarks[24].y, world_landmarks[24].z]
        r_knee_3d = [world_landmarks[26].x, world_landmarks[26].y, world_landmarks[26].z]
        r_ankle_3d = [world_landmarks[28].x, world_landmarks[28].y, world_landmarks[28].z]

        knee_angle = 0
        hip_angle = 0
        if is_ankles_visible and is_knees_visible:
            knee_angle = max(utils.calculate_angle_3d(l_hip_3d, l_knee_3d, l_ankle_3d), utils.calculate_angle_3d(r_hip_3d, r_knee_3d, r_ankle_3d))
            hip_angle = max(utils.calculate_angle_3d(l_sh_3d, l_hip_3d, l_knee_3d), utils.calculate_angle_3d(r_sh_3d, r_hip_3d, r_knee_3d))
            debug_vals["Knee"] = int(knee_angle)
            debug_vals["Hip"] = int(hip_angle)

    except Exception: return "unknown", "Error", False, debug_vals

    if ref_stand_y is not None and box_h > 0:
        drop_ratio = (current_y * img_h - ref_stand_y * img_h) / box_h
        if drop_ratio > config.HEIGHT_DROP_THRESHOLD:
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

    if knee_angle > config.ANGLE_THRESHOLD_KNEE_STAND and hip_angle > config.ANGLE_THRESHOLD_HIP_SIT:
        return "standing", "Angles OK", True, debug_vals

    debug_str = f"Box R:{box_ratio:.1f}"
    box_thresh = 1.15 if ref_stand_y is None else config.BOX_RATIO_THRESHOLD 
    if box_ratio > box_thresh: return "standing", debug_str, True, debug_vals
    else: return "sitting", debug_str, False, debug_vals
