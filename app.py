import cv2
import torch
import numpy as np
import math
import time
import threading
import random
from datetime import datetime
from PIL import Image

# --- Sound Library Handling ---
try:
    from playsound import playsound
    SOUND_AVAILABLE = True
except ImportError:
    SOUND_AVAILABLE = False
    print("playsound library not found. Sound alerts will be visual only.")

# ------------------- Configuration -------------------
try:
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', source='github')
except:
    model = torch.hub.load('ultralytics/yolov5', 'yolov5m')

# UI Constants
CLR_MAIN = (0, 255, 100)    # Tactical Green
CLR_ALERT = (50, 50, 255)   # Red
CLR_WHITE = (255, 255, 255) # Fixed Variable Name
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Global Variables
event_logs = ["SYSTEM INITIALIZED", "SATELLITE LINK ESTABLISHED"]
last_sound_time = 0
base_lat, base_lon = 23.8103, 90.4125
start_time = time.time()

# ------------------- Logic Functions -------------------

def play_alert():
    """Plays sound in a separate thread with a cooldown."""
    global last_sound_time
    if SOUND_AVAILABLE and (time.time() - last_sound_time > 3):
        try:
            # Note: Ensure sound.mp3 is in the same folder as this script
            threading.Thread(target=lambda: playsound('sound.mp3'), daemon=True).start()
            last_sound_time = time.time()
        except Exception as e:
            print(f"Sound Error: {e}")

def add_log(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    # Only add if it's a new unique message to avoid log spam
    if not event_logs or message not in event_logs[0]:
        event_logs.insert(0, f"[{timestamp}] {message}")
        if len(event_logs) > 6: event_logs.pop()

def draw_tech_border(img, p1, p2, color):
    x1, y1 = p1
    x2, y2 = p2
    l = int((x2 - x1) * 0.2)
    # Top Left
    cv2.line(img, (x1, y1), (x1+l, y1), color, 2)
    cv2.line(img, (x1, y1), (x1, y1+l), color, 2)
    # Top Right
    cv2.line(img, (x2, y1), (x2-l, y1), color, 2)
    cv2.line(img, (x2, y1), (x2, y1+l), color, 2)
    # Bottom Left
    cv2.line(img, (x1, y2), (x1+l, y2), color, 2)
    cv2.line(img, (x1, y2), (x1, y2-l), color, 2)
    # Bottom Right
    cv2.line(img, (x2, y2), (x2-l, y2), color, 2)
    cv2.line(img, (x2, y2), (x2, y2-l), color, 2)

# ------------------- Main Pipeline -------------------

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    h, w = frame.shape[:2]

    # 1. Visual Base (Dark Overlay + Scanlines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (w,h), (5, 10, 5), -1)
    frame = cv2.addWeighted(frame, 0.85, overlay, 0.15, 0)
    for i in range(0, h, 4): cv2.line(frame, (0, i), (w, i), (0, 20, 0), 1)

    # 2. Run Detection
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()
    
    drone_count = 0
    radar_dots = []

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        if conf > 0.45:
            drone_count += 1
            cx, cy = int((x1+x2)/2), int((y1+y2)/2)
            
            # Target Visuals
            draw_tech_border(frame, (int(x1), int(y1)), (int(x2), int(y2)), CLR_ALERT)
            cv2.putText(frame, f"TRGT_DRONE_{int(conf*100)}", (int(x1), int(y1)-10), FONT, 0.5, CLR_ALERT, 1)
            cv2.line(frame, (w//2, h), (cx, cy), (0, 100, 255), 1) 
            
            # Radar Mapping (Center of radar is 110, h-110)
            rx = int(110 + ((cx / w) - 0.5) * 150)
            ry = int((h-110) + ((cy / h) - 0.5) * 150)
            radar_dots.append((rx, ry))

    # Alert Logic
    if drone_count > 0:
        add_log(f"ALERT: {drone_count} TARGET(S) DETECTED")
        play_alert()
    else:
        # Subtle scanning log if empty
        if random.random() < 0.01: add_log("SCANNING AREA...")

    # 3. UI - TOP HEADER
    cv2.rectangle(frame, (0, 0), (w, 75), (0, 0, 0), -1)
    cv2.line(frame, (0, 75), (w, 75), CLR_MAIN, 2)
    cv2.putText(frame, "BANGLADESH ARMY", (w//2 - 180, 45), cv2.FONT_HERSHEY_TRIPLEX, 1.2, CLR_MAIN, 2)
    cv2.putText(frame, f"DTG: {datetime.now().strftime('%d%H%M %b %y')}", (20, 45), FONT, 0.5, CLR_MAIN, 1)
    cv2.putText(frame, "AIR DEFENSE COMMAND // SECURE LINK", (w//2 - 160, 65), FONT, 0.4, CLR_MAIN, 1)

    # 4. UI - RADAR (Bottom Left)
    r_center = (110, h - 110)
    cv2.circle(frame, r_center, 80, CLR_MAIN, 1)
    cv2.circle(frame, r_center, 40, CLR_MAIN, 1)
    sweep_angle = (time.time() * 3) % (2 * math.pi)
    cv2.line(frame, r_center, (int(r_center[0]+80*math.cos(sweep_angle)), int(r_center[1]+80*math.sin(sweep_angle))), CLR_MAIN, 2)
    for dot in radar_dots: cv2.circle(frame, dot, 5, CLR_ALERT, -1)

    # 5. UI - THERMAL MINI (Next to Radar)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    thermal_mini = cv2.resize(thermal, (160, 110))
    cv2.rectangle(thermal_mini, (0,0), (159, 109), CLR_WHITE, 1)
    cv2.putText(thermal_mini, "IR-FEED", (5, 15), FONT, 0.4, CLR_WHITE, 1) # FIXED CLR_WHITE
    frame[h-165:h-55, 210:370] = thermal_mini

    # 6. UI - EVENT LOGS
    log_y = h - 30
    for log in event_logs:
        color = CLR_ALERT if "ALERT" in log else CLR_MAIN
        cv2.putText(frame, log, (400, log_y), FONT, 0.4, color, 1)
        log_y -= 20

    # 7. UI - TELEMETRY & THREAT (Right Side)
    cv2.putText(frame, f"LAT: {base_lat + random.uniform(-0.0001, 0.0001):.5f}", (w-220, 110), FONT, 0.5, CLR_MAIN, 1)
    cv2.putText(frame, f"LON: {base_lon + random.uniform(-0.0001, 0.0001):.5f}", (w-220, 130), FONT, 0.5, CLR_MAIN, 1)
    
    cv2.rectangle(frame, (w-220, h-50), (w-20, h-20), (0, 20, 0), -1)
    cv2.rectangle(frame, (w-220, h-50), (w-20, h-20), CLR_MAIN, 1)
    bar_w = int(200 * (min(drone_count, 5) / 5)) if drone_count > 0 else 0
    cv2.rectangle(frame, (w-215, h-45), (w-215+bar_w, h-25), CLR_ALERT, -1)
    cv2.putText(frame, "THREAT LEVEL", (w-220, h-60), FONT, 0.4, CLR_MAIN, 1)

    # Show Output
    cv2.imshow('BANGLADESH_ARMY_TACTICAL_HUD', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()