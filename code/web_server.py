# web_server.py
from flask import Flask, jsonify, render_template_string, Response, request, send_file
import time
import cv2
import os
from datetime import datetime
import config
from shared_state import lock, outputFrame, APP_CONFIG, GLOBAL_STATUS

app = Flask(__name__)

TEMPLATE = """
<!doctype html>
<html lang="zh-Hant">
  <head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
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
      
      .control-panel { 
          display: grid; 
          grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); 
          gap: 15px; 
          background: var(--card-bg); 
          padding: 20px; 
          border-radius: 12px; 
          border: 1px solid var(--highlight); 
          margin-bottom: 20px; 
          align-items: center;
      }
      .full-width-item { grid-column: 1 / -1; text-align: center; }
      .ctrl-item { display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 8px; background: #232338; padding: 10px; border-radius: 8px; }
      .btn { background: var(--accent-color); color: #fff; border: none; padding: 8px 16px; border-radius: 5px; cursor: pointer; font-size: 0.9rem; transition: 0.3s; width: 100%; }
      .btn:hover { background: var(--success); color: #000; }
      .btn-red { background: #8B0000; font-weight: bold; }
      .btn-red:hover { background: #FF0000; color: #fff; box-shadow: 0 0 10px #FF0000; }
      .btn-emergency { font-size: 1.2rem; padding: 15px; }
      .btn-yellow { background: #d4a017; }
      .btn-yellow:hover { background: #ffd700; color: #000; }
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
        <div class="full-width-item">
            <button class="btn btn-red btn-emergency" onclick="resetEmergency()">
                <i class="fa-solid fa-bell-slash"></i> 🔕 解除緊急警報 (Reset Alarm)
            </button>
        </div>
        <div class="ctrl-item">
            <span>😴 睡眠模式</span>
            <label class="switch">
              <input type="checkbox" id="sleepToggle" onchange="toggleSleep()" {{ 'checked' if config.sleep_mode else '' }}>
              <span class="slider"></span>
            </label>
        </div>
        <div class="ctrl-item">
            <span>👀 隱私模式</span>
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
            <span style="font-size:0.8em">警報間隔 (秒)</span>
            <div style="display:flex; gap:5px;">
                <input type="number" id="alertInput" value="{{ config.alert_interval }}" style="width:50px; padding:5px; border-radius:4px; border:none; text-align:center;">
                <button class="btn" style="padding:5px 10px;" onclick="updateAlertTime()">設定</button>
            </div>
        </div>
        <div class="ctrl-item">
            <button class="btn btn-yellow" onclick="recalibrate()"><i class="fa-solid fa-ruler-vertical"></i> 重新校正身高</button>
        </div>
        <div class="ctrl-item">
            <button class="btn btn-red" onclick="hardReset()"><i class="fa-solid fa-bomb"></i> 數據歸零重置</button>
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
    <audio id="alarmAudio" src=""></audio>
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
               audioPlayer.src = "/stream_alarm?type=test"; 
               audioPlayer.play().then(() => {
                   audioPlayer.pause();
                   webAudioEnabled = true;
                   audioBtn.innerHTML = '<i class="fa-solid fa-volume-high"></i> 網頁音效: 開啟';
                   audioBtn.classList.add('on');
               }).catch(e => alert("無法啟用音效，請檢查瀏覽器設定"));
           } else {
               webAudioEnabled = false;
               audioPlayer.pause();
               audioPlayer.currentTime = 0;
               currentAlarmType = "none";
               audioBtn.innerHTML = '<i class="fa-solid fa-volume-xmark"></i> 網頁音效: 關閉';
               audioBtn.classList.remove('on');
           }
       }
       audioPlayer.addEventListener('ended', function() {
            if(webAudioEnabled && currentSound !== "") {
                this.currentTime = 0;
                this.play().catch(e => console.log("Loop failed", e));
            }
       });
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

# 需要一個 Logger 實例 (稍後在 main 注入，這裡先用 None 或 Lazy Load)
# 但為了簡單，可以在 main.py 創建 app 時傳入 logger

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
    # 支援 Test 請求
    if request.args.get('type') == 'test':
         if os.path.exists(config.ALARM_SOUND_FILE):
            return send_file(config.ALARM_SOUND_FILE, mimetype="audio/mpeg")

    if os.path.exists(config.ALARM_SOUND_FILE):
        return send_file(config.ALARM_SOUND_FILE, mimetype="audio/mpeg")
    return "No Audio", 404

@app.route('/stream_emergency')
def stream_emergency():
    if os.path.exists(config.EMERGENCY_SOUND_FILE):
        return send_file(config.EMERGENCY_SOUND_FILE, mimetype="audio/mpeg")
    return "No Audio", 404

# 這裡需要 logger，我們用一個全域變數或依賴注入
logger_instance = None

@app.route('/api/status')
def api_status():
    global logger_instance
    if logger_instance:
        with lock:
            total_time = logger_instance.data["Total_Sit_Seconds"]
            if logger_instance.current_sit_start_ts and not GLOBAL_STATUS.get("is_rehab_mode", False):
                total_time += (time.time() - logger_instance.current_sit_start_ts)
            
            is_sitting_in_rehab = (GLOBAL_STATUS["is_rehab_mode"] and GLOBAL_STATUS["balance_grade"] == "Sitting")
            is_emerg = GLOBAL_STATUS.get("is_emergency", False)

            return jsonify({
                "balance_score": GLOBAL_STATUS["balance_score"],
                "balance_grade": GLOBAL_STATUS["balance_grade"],
                "balance_color": GLOBAL_STATUS["balance_color"],
                "stand_count": logger_instance.data["Stand_Count"],
                "total_sit": int(total_time),
                "max_sit": int(logger_instance.data["Max_Sit_Seconds"]),
                "trigger_sound": is_sitting_in_rehab,
                "is_emergency": is_emerg
            })
    return jsonify({})

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
    global logger_instance
    try:
        data = logger_instance.data
        events_raw = logger_instance.read_events(10)
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
        if logger_instance.current_sit_start_ts is not None:
            realtime_total_sit += (time.time() - logger_instance.current_sit_start_ts)

        current_max = data.get("Max_Sit_Seconds", 0.0)
        if logger_instance.current_sit_start_ts is not None:
            current_duration = time.time() - logger_instance.current_sit_start_ts
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
                                      hourly_data=logger_instance.read_hourly_stats(), 
                                      daily_labels=logger_instance.read_daily_history()[0], 
                                      daily_data=logger_instance.read_daily_history()[1], 
                                      server_time=datetime.now().strftime("%H:%M:%S"))
    except Exception as e: return str(e)

@app.route("/events.csv")
def dl():
    if os.path.exists(logger_instance.events_file): return send_from_directory(os.path.dirname(logger_instance.events_file) or ".", os.path.basename(logger_instance.events_file))
    return "No file"

def run_flask(logger):
    global logger_instance
    logger_instance = logger
    try: app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
    except: pass
