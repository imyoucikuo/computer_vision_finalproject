
# Smart Care: 銀髮族久坐與復健照護系統
**Smart Care: Inactivity Care Alert System for Older Adults**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-green?style=for-the-badge&logo=opencv&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web_Dashboard-red?style=for-the-badge&logo=flask&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-yellow?style=for-the-badge&logo=yolo&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose_Estimation-teal?style=for-the-badge&logo=google&logoColor=white)

> **Computer Vision Course Final Project**

---

## 📖 專案簡介 (Introduction)

**Smart Care** 是一個整合 **YOLOv8 物體偵測** 與 **MediaPipe 姿態估計** 的智能照護系統原型。

本系統專為獨居長者設計，旨在解決「跌倒無人知曉」與「肌少症預防」兩大痛點。透過 WebCam 即時監控，系統能自動判斷長者的坐/站姿態，統計久坐時間，並提供 AR 體感復健遊戲鼓勵運動。一旦偵測到異常（如復健失敗或長時間無反應），系統將自動觸發紅色警報並發送 Email 通知家屬，實現從「日常監測」到「緊急救援」的完整閉環。

---

## 🚀 功能特點 (Features)

* 🧘 **精準姿態識別**：採用 YOLOv8 過濾背景雜訊，再由 MediaPipe 進行骨架分析，精準判斷「坐姿」與「站姿」。
* ⏱️ **久坐提醒監控**：即時追蹤坐姿持續時間，超過設定閥值即發出語音警報，引導長者起身。
* 🎮 **AR 體感復健遊戲**：互動式復健模式，使用者需揮手觸碰螢幕上的虛擬目標，並伴隨粒子煙火特效作為正向回饋。
* 🚨 **緊急狀態偵測 (Emergency Detection)**：
    * **復健失敗**：若在引導後 30 秒內無法站立，視為行動異常。
    * **意識喪失**：若在一般模式下超過 60 秒完全無細微動作，視為昏厥。
    * **自動通報**：觸發上述條件時，畫面轉紅、播放警報音，並自動發送 Email 給緊急聯絡人。
* 📉 **平衡能力評估**：分析站立時肩膀的晃動程度 (Body Sway)，計算平衡分數並給予評級。
* 💻 **Web 遠端儀表板**：基於 Flask 的 RWD 網頁介面，提供即時影像、歷史數據圖表 (Chart.js)，以及遠端控制功能。
* 🛡️ **隱私與人性化設計**：
    * **隱私模式**：一鍵模糊人臉。
    * **睡眠模式**：休息時暫停偵測與警報。

---

## 🏗️ 系統架構 (Architecture)

```mermaid
graph TD
    Input[攝影機輸入] --> Preproc[YOLOv8: 人體偵測 & ROI 裁切]
    Preproc --> Pose[MediaPipe: 33點骨架估計]
    Pose --> Analysis{狀態機邏輯核心}
    
    Analysis -->|正常活動| Status1[累積久坐時間 & 姿勢判斷]
    Analysis -->|久坐超時| Status2[觸發 AR 復健遊戲]
    Analysis -->|長時間靜止 / 復健失敗| Status3[緊急狀態 EMERGENCY]
    
    Status1 --> Log[CSV 數據紀錄]
    Status2 --> Game[Pygame 音效 & 粒子特效]
    Status3 --> Alert[紅色畫面 & Email 推播]
    
    Log --> Web[Flask Web Server]
    Game --> Web
    Alert --> Web[Web 儀表板: 監控/圖表/控制]
📂 專案結構 (File Structure)
建議將程式碼模組化以利維護：

Plaintext

Smart-Care-System/
├── main.py                 # 主程式入口 (Entry Point)
├── config.py               # 系統參數設定 (Email, 閾值, 路徑)
├── utils.py                # 工具函式庫 (幾何計算, Email發送, 特效)
├── logger.py               # 數據紀錄模組 (CSV Handler)
├── pose_logic.py           # 姿勢分析核心演算法
├── web_server.py           # Flask 網頁伺服器路由
├── shared_state.py         # 跨執行緒共享變數
├── sedentary_log.csv       # (自動生成) 每日數據日誌
├── assets/                 # 音效素材資料夾
│   ├── up.mp3              # 久坐提醒音
│   ├── success.mp3         # 遊戲得分音
│   ├── start.mp3           # 開始復健音
│   └── emergency.mp3       # 緊急警報音
└── README.md               # 說明文件
⚙️ 安裝與設定 (Installation)
1. 環境需求
Python 3.8 或以上版本

Web Browser (Chrome/Safari/Edge)

網路連接 (用於發送 Email 通知)

2. 安裝依賴套件
請在終端機 (Terminal) 執行以下指令：

Bash

pip install opencv-python mediapipe pygame numpy flask ultralytics requests
3. 音效檔案準備
請確保專案根目錄下有 up.mp3, success.mp3, start.mp3, emergency.mp3 這四個檔案，否則系統會以無聲模式運行。

4. 設定 Email 通知 (⚠️ 重要)
本系統使用 Gmail SMTP 發送警報。由於 Google 安全性政策，不能使用登入密碼。

前往 Google 帳戶 > 安全性 > 兩步驟驗證 > 應用程式密碼 (App Passwords)。

申請一組新的密碼 (選擇「郵件」/「Windows 電腦」)。

打開 config.py，修改以下設定：

Python

# config.py

# 請填入你的 Gmail
SENDER_EMAIL = "your_email@gmail.com"

# 請填入剛剛申請的 16 位數應用程式密碼 (不要有空格)
SENDER_PASSWORD = "abcd efgh ijkl mnop" 

# 接收警報的信箱
RECEIVER_EMAIL = "caregiver_email@gmail.com"
⚠️ 資安警告：若要將專案上傳至 GitHub 公開倉庫，請務必將 config.py 中的真實密碼刪除，或改用環境變數 (os.getenv) 讀取，以免帳號外洩。

🎮 使用方法 (Usage)
1. 啟動系統
在專案目錄下執行：

Bash

python main.py
系統啟動後會顯示：

OpenCV 視窗：顯示即時偵測畫面。

Flask 伺服器：在背景執行，Port 5000。

2. 初始化校正
使用者請移動至鏡頭前，保持 站立姿勢 約 3-5 秒。待畫面右上方出現綠色的 "System Ready" 字樣，即代表身高比例校正完成，系統開始運作。

3. 開啟網頁儀表板
打開瀏覽器（手機或電腦皆可），輸入網址：

本機：http://localhost:5000

區網：http://<你的電腦IP>:5000 (例如 192.168.1.100:5000)

注意：若是手機瀏覽器，進入網頁後請點擊控制面板上的 「音效」按鈕 或畫面任意處，以解鎖瀏覽器的自動播放限制。

❓ 常見問題 (FAQ)
Q: 為什麼系統一直顯示 "Please STAND to Calibrate"? A: 系統需要基準身高來作為後續坐姿判斷的依據。請確保全身（至少上半身與膝蓋）完整出現在鏡頭內，並保持直立站姿，直到系統抓取到穩定的骨架數據。

Q: 手機瀏覽器聽不到警報聲？ A: 這是手機瀏覽器 (iOS Safari / Android Chrome) 的自動播放安全限制。請在進入網頁後，手動點擊一次控制面板上的「音效開關」按鈕，即可解鎖音效權限。

Q: 為什麼沒收到 Email 通知？ A: 請依序檢查：

是否已申請 Google App Password (非登入密碼)？

是否在 config.py 中填寫正確？

檢查收件信箱的「垃圾郵件」夾。

系統是否處於「睡眠模式」？(睡眠模式下不會發送警報)。

Q: 畫面上的紅色 "EMERGENCY" 怎麼消除？ A: 有兩種方式：

使用者自解：長者只需站起來並面對鏡頭揮手（觸發動態判定）。

遠端解除：照護者在網頁儀表板點擊紅色的 「🔕 解除緊急警報」 按鈕。
