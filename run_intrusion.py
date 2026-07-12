import argparse
import json
import os
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import requests
from ultralytics import YOLO

# =============================
# TELEGRAM CONFIG (ВСТАВЬ СЮДА)
# =============================
TG_BOT_TOKEN = "8316881895:AAEejmLdjZCvkMyrM31aQshCUsNMTMlVGEc"
TG_CHAT_ID = "1542771283"

ALERT_COOLDOWN_SEC = 20

# Clip settings (seconds)
PRE_SECONDS = 2.0
POST_SECONDS = 2.0

# Continuous recording (segment length, seconds)
RECORD_SEGMENT_SEC = 10 * 60  # 10 минут


# =============================
# STYLE (DARK + B/W)
# =============================
# For B/W overlay we keep mainly white/gray + red for alert
WHITE = (255, 255, 255)
GRAY = (190, 190, 190)
DARK_BG = (25, 25, 25)
RED = (60, 60, 255)  # alert accent

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
FONT_THICKNESS = 2
AA = cv2.LINE_AA


# =============================
# TELEGRAM
# =============================
def tg_send_photo(image_path: str, caption: str):
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendPhoto"
    with open(image_path, "rb") as img:
        files = {"photo": img}
        data = {"chat_id": TG_CHAT_ID, "caption": caption}
        requests.post(url, files=files, data=data, timeout=30)


def tg_send_video_as_document(video_path: str, caption: str):
    # sendDocument = максимальная совместимость (видео не "размытое", открывается везде)
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendDocument"
    with open(video_path, "rb") as vid:
        files = {"document": vid}
        data = {"chat_id": TG_CHAT_ID, "caption": caption}
        requests.post(url, files=files, data=data, timeout=180)

# =============================
# CV + DRAW UTILS
# =============================
def safe_source(src: str):
    return int(src) if src.isdigit() else src

def point_in_polygon(point, polygon):
    poly = np.array(polygon, dtype=np.int32)
    return cv2.pointPolygonTest(poly, (float(point[0]), float(point[1])), False) >= 0

def center_of_bbox(xyxy):
    x1, y1, x2, y2 = xyxy
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def bw_base(frame_bgr):
    # nice B/W look: gray + slight contrast
    g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.convertScaleAbs(g, alpha=1.15, beta=0)  # contrast
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

def draw_transparent_polygon(frame, polygon, alpha=0.18):
    # grayscale fill (dark mode style)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon], (70, 70, 70))
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def draw_label(frame, text, x, y, bg=DARK_BG, fg=WHITE):
    (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)
    cv2.rectangle(frame, (x, y - th - 10), (x + tw + 10, y), bg, -1)
    cv2.putText(frame, text, (x + 5, y - 5), FONT, FONT_SCALE, fg, FONT_THICKNESS, AA)

def open_segment_writer(out_dir: str, fps: float, w: int, h: int):
    os.makedirs(out_dir, exist_ok=True)
    name = time.strftime("record_%Y%m%d_%H%M%S.mp4")
    path = os.path.join(out_dir, name)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    return writer, path, time.time()

# =============================
# MAIN
# =============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="input.mp4", help='0/1/2 for webcam, or "rtsp://...", "http://...", or file.mp4')
    parser.add_argument("--zones", default="zones.json")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--events", default="events.jsonl")
    parser.add_argument("--out_dir", default="out", help="output directory (records, alerts, etc.)")
    parser.add_argument("--show", action="store_true", help="show live window (useful for webcam)")
    parser.add_argument("--bw", action="store_true", help="black&white mode for camera frame")
    args = parser.parse_args()

    if TG_BOT_TOKEN.startswith("PASTE") or TG_CHAT_ID.startswith("PASTE"):
        print("❗ Set TG_BOT_TOKEN and TG_CHAT_ID at the top of the file before running.")
        return

    source = safe_source(str(args.source))
    zones = json.loads(Path(args.zones).read_text(encoding="utf-8"))["zones"]

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 25.0
    dt = 1.0 / fps

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

    out_dir = args.out_dir
    alerts_dir = os.path.join(out_dir, "alerts")
    records_dir = os.path.join(out_dir, "records")
    os.makedirs(alerts_dir, exist_ok=True)
    os.makedirs(records_dir, exist_ok=True)

    # Continuous recording writer (rotates by segments)
    rec_writer, rec_path, rec_t0 = open_segment_writer(records_dir, fps, w, h)

    model = YOLO(args.model)

    # intrusion state
    in_zone_time = {}          # (zone_id, pseudo_id) -> seconds
    fired = set()              # one-shot per (zone, pseudo_id)
    last_alert_time = {}       # zone_id -> timestamp cooldown

    # events file
    events_file = open(os.path.join(out_dir, args.events), "w", encoding="utf-8")

    # prebuffer for clip
    pre_frames_max = max(1, int(PRE_SECONDS * fps))
    prebuffer = deque(maxlen=pre_frames_max)

    # active clip recordings per zone
    active_clips = {}  # zone_id -> {writer, path, remaining, caption}

    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        # base look
        if args.bw:
            frame = bw_base(frame)

        results = model.predict(frame, conf=args.conf, verbose=False)[0]

        # ----- draw zones (smooth edges) -----
        for z in zones:
            poly = np.array(z["polygon"], dtype=np.int32)
            poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
            poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)

            draw_transparent_polygon(frame, poly, alpha=0.16)
            cv2.polylines(frame, [poly], True, GRAY, 2, lineType=AA)  # "идеальные" AA контуры

        draw_label(frame, "SMART CV | INTRUSION", 20, 35, bg=DARK_BG, fg=WHITE)

        # ----- detections -----
        if results.boxes is not None and len(results.boxes) > 0:
            for i in range(len(results.boxes)):
                if int(results.boxes.cls[i]) != 0:
                    continue

                score = float(results.boxes.conf[i])
                x1, y1, x2, y2 = map(int, results.boxes.xyxy[i].tolist())
                cx, cy = center_of_bbox((x1, y1, x2, y2))

                bbox_color = GRAY

                for z in zones:
                    zone_id = z["id"]
                    min_seconds = float(z.get("min_seconds", 2.0))
                    inside = point_in_polygon((cx, cy), z["polygon"])
                    key = (zone_id, i)

                    if inside:
                        in_zone_time[key] = in_zone_time.get(key, 0.0) + dt
                        draw_label(
                            frame,
                            f"{zone_id.upper()}  {in_zone_time[key]:.1f}s",
                            x1,
                            max(20, y1 - 5),
                            bg=DARK_BG,
                            fg=WHITE
                        )
                    else:
                        in_zone_time[key] = 0.0

                    alert = in_zone_time[key] >= min_seconds
                    if alert:
                        bbox_color = RED

                        # trigger (one-shot per key, but recording always continues)
                        if key not in fired:
                            fired.add(key)

                            event = {
                                "type": "intrusion",
                                "zone_id": zone_id,
                                "frame": frame_idx,
                                "time_sec": round(frame_idx / fps, 2),
                                "bbox": [x1, y1, x2, y2],
                                "confidence": round(score, 2),
                            }
                            events_file.write(json.dumps(event, ensure_ascii=False) + "\n")

                            now = time.time()
                            prev = last_alert_time.get(zone_id, 0)

                            if now - prev >= ALERT_COOLDOWN_SEC:
                                last_alert_time[zone_id] = now

                                caption = (
                                    f"🟥 ALERT: INTRUSION\n"
                                    f"📍 Zone: {zone_id}\n"
                                    f"⏱ In zone: {in_zone_time[key]:.2f}s (min {min_seconds:.2f}s)\n"
                                    f"🎯 Confidence: {score:.2f}\n"
                                    f"🕒 {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                                    f"🎞 t={frame_idx/fps:.2f}s | frame={frame_idx}\n"
                                    f"📦 bbox=[{x1},{y1},{x2},{y2}]"
                                )

                                # snapshot
                                snap_path = os.path.join(alerts_dir, f"alert_{zone_id}_{frame_idx}.jpg")
                                cv2.imwrite(snap_path, frame)
                                try:
                                    tg_send_photo(snap_path, caption)
                                except Exception as e:
                                    print("TG photo error:", e)

                                # start clip (pre + post)
                                if zone_id not in active_clips:
                                    clip_path = os.path.join(alerts_dir, f"clip_{zone_id}_{frame_idx}.mp4")
                                    clip_writer = cv2.VideoWriter(
                                        clip_path,
                                        cv2.VideoWriter_fourcc(*"mp4v"),
                                        fps,
                                        (w, h)
                                    )

                                    for pf in prebuffer:
                                        clip_writer.write(pf)
                                    clip_writer.write(frame)

                                    remaining = max(1, int(POST_SECONDS * fps))
                                    active_clips[zone_id] = {
                                        "writer": clip_writer,
                                        "path": clip_path,
                                        "remaining": remaining,
                                        "caption": caption,
                                    }

                # bbox (AA)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2, lineType=AA)

        # timestamp
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, ts, (w - 260, h - 15), FONT, 0.45, GRAY, 1, AA)

        # ----- active clips post frames -----
        finished = []
        for zone_id, clip in active_clips.items():
            clip["writer"].write(frame)
            clip["remaining"] -= 1
            if clip["remaining"] <= 0:
                clip["writer"].release()
                try:
                    tg_send_video_as_document(clip["path"], clip["caption"])
                except Exception as e:
                    print("TG video(doc) error:", e)
                finished.append(zone_id)

        for zone_id in finished:
            active_clips.pop(zone_id, None)

        # ----- prebuffer -----
        prebuffer.append(frame.copy())

        # ----- continuous recording rotation -----
        # always record full stream
        rec_writer.write(frame)
        if time.time() - rec_t0 >= RECORD_SEGMENT_SEC:
            rec_writer.release()
            rec_writer, rec_path, rec_t0 = open_segment_writer(records_dir, fps, w, h)

        # ----- show -----
        if args.show:
            cv2.imshow("SmartCV (Q to quit)", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break

    # cleanup
    events_file.close()
    cap.release()

    try:
        rec_writer.release()
    except Exception:
        pass

    for zone_id, clip in active_clips.items():
        try:
            clip["writer"].release()
        except Exception:
            pass

    if args.show:
        cv2.destroyAllWindows()

    print("DONE")


if __name__ == "__main__":
    main()
