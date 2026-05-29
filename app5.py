import base64
import cv2
import json
import numpy as np
import os
import sys
from tensorflow.keras.models import load_model
import time
from datetime import datetime
from urllib import error, request


def resource_path(relative_path: str) -> str:
    """Resolve path to a bundled resource (works both frozen and unfrozen)."""
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)  # type: ignore[attr-defined]
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)


def _app_support_dir() -> str:
    """~/Library/Application Support/DrowsinessApp2 — writable on macOS."""
    path = os.path.join(os.path.expanduser('~'), 'Library', 'Application Support', 'DrowsinessApp2')
    os.makedirs(path, exist_ok=True)
    return path


def load_env_file(file_name='.env'):
    candidates = []

    if getattr(sys, 'frozen', False):
        # 1. Next to the .app bundle in MacOS dir (Contents/MacOS/.env)
        candidates.append(os.path.join(os.path.dirname(sys.executable), file_name))
        # 2. ~/Library/Application Support/DrowsinessApp2/.env
        candidates.append(os.path.join(_app_support_dir(), file_name))
    else:
        candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name))

    env_path = next((p for p in candidates if os.path.exists(p)), None)

    if env_path is None:
        return

    with open(env_path, 'r', encoding='utf-8') as env_file:
        for raw_line in env_file:
            line = raw_line.strip()

            if line == '' or line.startswith('#') or '=' not in line:
                continue

            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if key != '':
                os.environ.setdefault(key, value)


def get_int_env(name, default=''):
    value = os.getenv(name, default).strip()

    if value.isdigit():
        return int(value)

    return None


load_env_file()

# =========================
# LOAD MODEL
# =========================
model_eye = load_model(resource_path('model_eye_mobilenet.h5'))
model_mouth = load_model(resource_path('model_mouth_mobilenet.h5'))

eye_labels = ['Closed', 'Open']
mouth_labels = ['No_yawn', 'Yawn']

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

# =========================
# CONFIG
# =========================
closed_frame_count = 0
eye_closed_start = None     # timestamp saat mata pertama kali tertutup
eye_open_start = None       # timestamp saat mata pertama kali terbuka (untuk toleransi reset)

YAWN_COUNTER = 0
YAWN_COOLDOWN = False
yawn_frame_count = 0

last_yawn_time = time.time()

EYE_CLOSED_THRESHOLD = 15  # frame (referensi display saja)
EYE_CLOSED_SECS_1 = 15     # detik → level 1: micro-sleep (alert pertama)
EYE_CLOSED_SECS_2 = 30     # detik → level 2: kritis + bunyi + border merah
EYE_OPEN_RESET_SECS = 0.4  # detik → mata harus terbuka minimal ini sebelum counter direset
YAWN_LIMIT = 3 # mencapai 3 kali menguap dalam TIME_WINDOW deteksi, bisa disesuaikan sesuai kebutuhan
TIME_WINDOW = 60 

YAWN_FRAME_THRESHOLD = 5 # jumlah frame berturut-turut mulut terbuka untuk deteksi yawn, untuk mengurangi false positive saat bicara atau tersenyum
MIN_CONFIDENCE = 0.80 # threshold confidence minimum untuk mempertimbangkan prediksi valid, bisa disesuaikan berdasarkan pengujian model dan kebutuhan lapangan

BACKEND_EVENT_URL = os.getenv('DROWSINESS_API_URL', 'http://127.0.0.1:8000/api/realtime/kelelahan/events').strip()
BACKEND_DEVICE_TOKEN = os.getenv('DROWSINESS_DEVICE_TOKEN', '').strip()
SOURCE_DEVICE = os.getenv('DROWSINESS_SOURCE_DEVICE', 'raspberry-pi').strip()
_default_pending_file = os.path.join(
    _app_support_dir() if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__)),
    'pending_drowsiness_events.jsonl',
)
PENDING_EVENTS_FILE = os.getenv('DROWSINESS_PENDING_FILE', _default_pending_file).strip()
REQUEST_TIMEOUT_SECONDS = 5
EVENT_COOLDOWN_SECONDS = 30
PENDING_FLUSH_INTERVAL_SECONDS = 15

# export DROWSINESS_API_URL="http://127.0.0.1:8000/api/realtime/kelelahan/events"
# export DROWSINESS_DEVICE_TOKEN="isi-token-yang-sama-dengan-env-laravel"
# export DROWSINESS_SOURCE_DEVICE="raspberrypi-01"
last_sent_at = {
    'micro_sleep': 0.0,
    'yawn_alert': 0.0,
    'fatigue_alert': 0.0,
}
active_alerts = set()
last_pending_flush = 0.0

print("Sistem Deteksi Kantuk Berjalan... Tekan 'q' untuk berhenti.")

if BACKEND_EVENT_URL and BACKEND_DEVICE_TOKEN:
    print(f"Realtime backend aktif: {BACKEND_EVENT_URL}")
else:
    print("Realtime backend nonaktif. Isi file .env atau set DROWSINESS_API_URL dan DROWSINESS_DEVICE_TOKEN untuk mengirim event.")

# =========================
# FUNCTION
# =========================
def get_prediction(img, model, labels, threshold, safe_index):
    try:
        img_resized = cv2.resize(img, (160, 160))
        img_resized = img_resized.astype("float32") / 255.0
        img_resized = np.expand_dims(img_resized, axis=0)

        pred = model.predict(img_resized, verbose=0)[0]
        idx = np.argmax(pred)
        conf = pred[idx]

        # DEBUG (optional)
        # print("Pred:", pred)

        if conf < threshold:
            return labels[safe_index], conf

        return labels[idx], conf

    except Exception:
        return labels[safe_index], 0.0


def is_backend_ready():
    return BACKEND_EVENT_URL != '' and BACKEND_DEVICE_TOKEN != ''


def store_pending_event(payload):
    with open(PENDING_EVENTS_FILE, 'a', encoding='utf-8') as pending_file:
        pending_file.write(json.dumps(payload) + '\n')


def store_pending_event_no_image(payload):
    payload_copy = {k: v for k, v in payload.items() if k != 'image_base64'}
    with open(PENDING_EVENTS_FILE, 'a', encoding='utf-8') as pending_file:
        pending_file.write(json.dumps(payload_copy) + '\n')


def encode_frame_jpeg(frame) -> str | None:
    try:
        if frame is None:
            return None
        small = cv2.resize(frame, (640, 360))
        ok, buf = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 60])
        return base64.b64encode(buf.tobytes()).decode('utf-8') if ok else None
    except Exception:
        return None


def send_event_payload(payload):
    if not is_backend_ready():
        return False

    data = json.dumps(payload).encode('utf-8')
    req = request.Request(
        BACKEND_EVENT_URL,
        data=data,
        headers={
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'X-Device-Token': BACKEND_DEVICE_TOKEN,
        },
        method='POST',
    )

    try:
        with request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            return 200 <= response.status < 300
    except (error.URLError, error.HTTPError, TimeoutError):
        return False


def flush_pending_events(current_time):
    global last_pending_flush

    if current_time - last_pending_flush < PENDING_FLUSH_INTERVAL_SECONDS:
        return

    last_pending_flush = current_time

    if not is_backend_ready() or not os.path.exists(PENDING_EVENTS_FILE):
        return

    with open(PENDING_EVENTS_FILE, 'r', encoding='utf-8') as pending_file:
        rows = [line.strip() for line in pending_file.readlines() if line.strip()]

    if not rows:
        os.remove(PENDING_EVENTS_FILE)
        return

    remaining_rows = []

    for row in rows:
        try:
            payload = json.loads(row)
        except json.JSONDecodeError:
            continue

        if not send_event_payload(payload):
            remaining_rows.append(row)

    if remaining_rows:
        with open(PENDING_EVENTS_FILE, 'w', encoding='utf-8') as pending_file:
            pending_file.write('\n'.join(remaining_rows) + '\n')
    else:
        os.remove(PENDING_EVENTS_FILE)


def build_event_payload(event_type, severity, status_text, eye_label, conf_eye, mouth_label, conf_mouth, image_base64=None):
    payload = {
        'event_type': event_type,
        'severity': severity,
        'status_text': status_text,
        'eye_label': eye_label,
        'mouth_label': mouth_label,
        'eye_confidence': round(float(conf_eye), 4),
        'mouth_confidence': round(float(conf_mouth), 4),
        'closed_frame_count': int(closed_frame_count),
        'yawn_count': int(YAWN_COUNTER),
        'source_device': SOURCE_DEVICE,
        'detected_at': datetime.now().astimezone().isoformat(timespec='seconds'),
        'metadata': {
            'time_window_seconds': TIME_WINDOW,
            'eye_closed_threshold': EYE_CLOSED_THRESHOLD,
            'yawn_limit': YAWN_LIMIT,
        },
    }
    if image_base64 is not None:
        payload['image_base64'] = image_base64
    return payload


def maybe_send_event(event_type, severity, status_text, eye_label, conf_eye, mouth_label, conf_mouth, current_time, image_to_send=None):
    if not is_backend_ready():
        return

    if current_time - last_sent_at[event_type] < EVENT_COOLDOWN_SECONDS:
        return

    image_b64 = encode_frame_jpeg(image_to_send)
    payload = build_event_payload(
        event_type,
        severity,
        status_text,
        eye_label,
        conf_eye,
        mouth_label,
        conf_mouth,
        image_b64,
    )

    if not send_event_payload(payload):
        store_pending_event_no_image(payload)

    last_sent_at[event_type] = current_time

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    # Reset Yawn Counter tiap TIME_WINDOW
    current_time = time.time()
    flush_pending_events(current_time)

    if current_time - last_yawn_time > TIME_WINDOW:
        YAWN_COUNTER = 0
        last_yawn_time = current_time

    status_text = "Status: Siaga (Aman)"
    alert_color = (0, 255, 0)

    eye_closed_duration = 0.0
    if len(faces) == 0:
        closed_frame_count = 0
        eye_closed_start = None
        eye_open_start = None
        yawn_frame_count = 0
        YAWN_COOLDOWN = False

    for (x, y, w, h) in faces:

        # =========================
        # ROI LEBIH PRESISI
        # =========================
        eye_roi = frame[y:y + int(h * 0.5), x:x + w]

        mouth_roi = frame[
            y + int(h * 0.7): y + int(h * 0.9),
            x + int(w * 0.2): x + int(w * 0.8)
        ]

        # =========================
        # PREDICTION
        # =========================
        label_eye, conf_eye = get_prediction(
            eye_roi, model_eye, eye_labels, MIN_CONFIDENCE, safe_index=1
        )

        label_mouth, conf_mouth = get_prediction(
            mouth_roi, model_mouth, mouth_labels, MIN_CONFIDENCE, safe_index=0
        )

        # =========================
        # MICRO-SLEEP (EYE)
        # =========================
        if label_eye == 'Closed':
            eye_open_start = None          # reset timer buka
            if eye_closed_start is None:
                eye_closed_start = current_time
            closed_frame_count += 1
            eye_closed_duration = current_time - eye_closed_start
        else:
            if eye_open_start is None:
                eye_open_start = current_time
            open_duration = current_time - eye_open_start
            if open_duration >= EYE_OPEN_RESET_SECS:
                # Mata sudah cukup lama terbuka → reset counter
                eye_closed_start = None
                closed_frame_count = 0
                eye_closed_duration = 0.0
                eye_open_start = None
            else:
                # Mata terbuka sebentar → pertahankan durasi tutup
                eye_closed_duration = (current_time - eye_closed_start) if eye_closed_start else 0.0

        # =========================
        # YAWN SMOOTHING
        # =========================
        if label_mouth == 'Yawn':
            yawn_frame_count += 1
        else:
            yawn_frame_count = 0

        if yawn_frame_count >= YAWN_FRAME_THRESHOLD:
            if not YAWN_COOLDOWN:
                YAWN_COUNTER += 1
                YAWN_COOLDOWN = True
        else:
            YAWN_COOLDOWN = False

        # =========================
        # VISUALISASI (gambar dulu agar frame ter-anotasi sebelum dikirim)
        # =========================
        e_color = (0, 0, 255) if label_eye == 'Closed' else (0, 255, 0)
        m_color = (0, 0, 255) if label_mouth == 'Yawn' else (255, 120, 0)

        # Eye box
        cv2.rectangle(frame, (x, y), (x + w, y + int(h * 0.5)), e_color, 2)
        cv2.putText(frame,
                    f"EYE: {label_eye} ",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, e_color, 2)

        # Mouth box
        cv2.rectangle(frame,
                      (x + int(w * 0.2), y + int(h * 0.7)),
                      (x + int(w * 0.8), y + int(h * 0.9)),
                      m_color, 2)
        cv2.putText(frame,
                    f"MOUTH: {label_mouth} ",
                    (x, y + h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, m_color, 2)

        # =========================
        # STATUS FINAL
        # =========================
        if eye_closed_duration >= EYE_CLOSED_SECS_2:
            status_text = f"!!! BAHAYA: TERTIDUR ({int(eye_closed_duration)}s) !!!"
            alert_color = (0, 0, 180)
        elif eye_closed_duration >= EYE_CLOSED_SECS_1:
            status_text = f"!!! BAHAYA: MICRO-SLEEP ({int(eye_closed_duration)}s) !!!"
            alert_color = (0, 0, 255)
        elif YAWN_COUNTER >= YAWN_LIMIT:
            status_text = f"PERINGATAN: LELAH ({YAWN_COUNTER}x Menguap)"
            alert_color = (0, 165, 255)

        # =========================
        # TANDA ALERT DI FRAME (untuk gambar yang dikirim)
        # =========================
        if eye_closed_duration >= EYE_CLOSED_SECS_1 or YAWN_COUNTER >= YAWN_LIMIT:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 55), (0, 0, 0), -1)
            cv2.putText(frame, status_text, (10, 38),
                        cv2.FONT_HERSHEY_DUPLEX, 0.75, alert_color, 2)
        if eye_closed_duration >= EYE_CLOSED_SECS_2:
            cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), (0, 0, 255), 10)

        if eye_closed_duration >= EYE_CLOSED_SECS_2:
            active_alerts.discard('micro_sleep')
            if 'fatigue_alert' not in active_alerts:
                print('\a', end='', flush=True)  # bunyi terminal bell
                maybe_send_event(
                    'fatigue_alert',
                    'danger',
                    status_text,
                    label_eye,
                    conf_eye,
                    label_mouth,
                    conf_mouth,
                    current_time,
                    image_to_send=frame,
                )
                active_alerts.add('fatigue_alert')
        elif eye_closed_duration >= EYE_CLOSED_SECS_1:
            active_alerts.discard('fatigue_alert')
            if 'micro_sleep' not in active_alerts:
                maybe_send_event(
                    'micro_sleep',
                    'danger',
                    status_text,
                    label_eye,
                    conf_eye,
                    label_mouth,
                    conf_mouth,
                    current_time,
                    image_to_send=frame,
                )
                active_alerts.add('micro_sleep')
        else:
            active_alerts.discard('micro_sleep')
            active_alerts.discard('fatigue_alert')

        if YAWN_COUNTER >= YAWN_LIMIT:
            if 'yawn_alert' not in active_alerts:
                maybe_send_event(
                    'yawn_alert',
                    'warning',
                    f"PERINGATAN: LELAH ({YAWN_COUNTER}x Menguap)",
                    label_eye,
                    conf_eye,
                    label_mouth,
                    conf_mouth,
                    current_time,
                    image_to_send=frame,
                )
                active_alerts.add('yawn_alert')
        else:
            active_alerts.discard('yawn_alert')

    # =========================
    # DASHBOARD
    # =========================
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 85), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, status_text,
                (20, 35),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                alert_color, 2)

    closed_info = f"{int(eye_closed_duration)}s" if eye_closed_duration > 0 else f"{closed_frame_count}f"
    cv2.putText(frame,
                f"Yawn: {YAWN_COUNTER}/{YAWN_LIMIT} | Mata tutup: {closed_info}",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255), 1)

    if eye_closed_duration >= EYE_CLOSED_SECS_2:
        cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), (0, 0, 255), 10)

    cv2.imshow('Fatigue Detection - MobileNetV2', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()