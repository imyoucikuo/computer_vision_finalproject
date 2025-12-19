# logger.py
import os
import csv
import time
from datetime import datetime
import config
from shared_state import GLOBAL_STATUS

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
