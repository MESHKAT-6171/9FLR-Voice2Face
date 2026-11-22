import csv
import gc
import math
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import cv2
import imageio_ffmpeg
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import noisereduce as nr
import numpy as np
import soundfile as sf
import yaml
from tqdm import tqdm

# --- Load Configuration ---
CONFIG_PATH = Path("configs/config.yaml")
with CONFIG_PATH.open("r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

RAW_VIDEO_DIR = Path(CFG["paths"]["raw_video_dir"])
PROCESSED_DIR = Path(CFG["paths"]["processed_dir"])
AUDIO_OUT_DIR = PROCESSED_DIR / "audio"
FACE_OUT_DIR = PROCESSED_DIR / "faces"
REGION_OUT_DIR = PROCESSED_DIR / "regions_9flr"
MANIFEST_PATH = PROCESSED_DIR / "manifest.csv"

SAMPLE_RATE = int(CFG["preprocessing"]["sample_rate"])          # 16000
CLIP_DURATION = float(CFG["preprocessing"]["clip_duration_sec"])  # 4.0
CLIPS_PER_VIDEO = int(CFG["preprocessing"]["clips_per_video"])    # 5
IMG_SIZE = int(CFG["preprocessing"]["image_size"])                # 128
REGION_SIZE = int(CFG["preprocessing"]["region_size"])          # 48

FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

TASK_MODEL_PATH = Path("configs/face_landmarker.task")
TASK_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

# 9 Facial Local Regions (MediaPipe Face Mesh Landmark Indices 0-467)
LANDMARK_REGIONS = {
    "0_forehead":    [10, 338, 297, 332, 284, 251, 389, 356, 70, 63, 105, 66, 107, 9],
    "1_left_eye":    [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7, 70, 63, 105, 66, 107],
    "2_right_eye":   [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382, 336, 296, 334, 293, 300],
    "3_nose_bridge": [168, 6, 197, 195, 5, 122, 351, 196, 419],
    "4_nose_tip":    [4, 1, 19, 94, 2, 98, 327, 48, 278, 115, 344],
    "5_left_cheek":  [116, 123, 147, 213, 192, 214, 212, 202, 204, 187, 207, 205, 50, 101],
    "6_right_cheek": [345, 352, 376, 433, 416, 434, 432, 422, 424, 411, 427, 425, 280, 330],
    "7_mouth":       [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146],
    "8_chin_jaw":    [152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 148, 176, 149, 150, 136, 172, 58, 132, 93],
}


def ensure_dirs_and_model():
    for d in [AUDIO_OUT_DIR, FACE_OUT_DIR, REGION_OUT_DIR, TASK_MODEL_PATH.parent]:
        d.mkdir(parents=True, exist_ok=True)
    if not TASK_MODEL_PATH.exists():
        print(f"Downloading MediaPipe FaceLandmarker model to {TASK_MODEL_PATH}...")
        urllib.request.urlretrieve(TASK_MODEL_URL, str(TASK_MODEL_PATH))


def create_face_landmarker():
    base_options = mp_python.BaseOptions(model_asset_path=str(TASK_MODEL_PATH))
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


def load_processed_videos() -> set[str]:
    if not MANIFEST_PATH.exists():
        return set()
    done = set()
    with MANIFEST_PATH.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            done.add(row["video_id"])
    return done


def extract_clean_audio(video_path: Path, wav_out: Path) -> bool:
    """Extracts 16kHz mono PCM WAV via bundled FFmpeg."""
    cmd = [
        FFMPEG_EXE, "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", str(SAMPLE_RATE), "-ac", "1",
        str(wav_out)
    ]
    res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return res.returncode == 0 and wav_out.exists()


def align_and_extract_9flr(frame_bgr: np.ndarray, landmarker) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Detects face via MediaPipe FaceLandmarker, aligns eyes horizontally,
    returns (aligned_face_128x128_rgb, regions_9x48x48x3_rgb).
    """
    h, w, _ = frame_bgr.shape
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = landmarker.detect(mp_image)

    if not results.face_landmarks:
        return None

    landmarks = results.face_landmarks[0]
    pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32)

    # 1. Eye-center alignment
    left_eye_center = pts[[33, 133]].mean(axis=0)
    right_eye_center = pts[[362, 263]].mean(axis=0)
    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]
    angle = math.degrees(math.atan2(dy, dx))

    eyes_mid = (
        float((left_eye_center[0] + right_eye_center[0]) / 2.0),
        float((left_eye_center[1] + right_eye_center[1]) / 2.0),
    )

    rot_mat = cv2.getRotationMatrix2D(eyes_mid, angle, 1.0)
    aligned_rgb = cv2.warpAffine(rgb, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    # Transform landmarks into aligned coordinate space
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    pts_hom = np.hstack([pts, ones])
    aligned_pts = pts_hom @ rot_mat.T

    # 2. Crop tightly around aligned face with 15% margin
    x_min, y_min = aligned_pts.min(axis=0)
    x_max, y_max = aligned_pts.max(axis=0)
    box_w, box_h = x_max - x_min, y_max - y_min
    if box_w < 40 or box_h < 40:
        return None

    pad_x, pad_y = box_w * 0.15, box_h * 0.15
    cx1 = max(0, int(x_min - pad_x))
    cy1 = max(0, int(y_min - pad_y))
    cx2 = min(w, int(x_max + pad_x))
    cy2 = min(h, int(y_max + pad_y))

    face_crop = aligned_rgb[cy1:cy2, cx1:cx2]
    if face_crop.size == 0:
        return None
    face_resized = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    # 3. Extract 9 Facial Local Regions (9FLR)
    region_patches = []
    for _, idx_list in LANDMARK_REGIONS.items():
        r_pts = aligned_pts[idx_list]
        rx1, ry1 = r_pts.min(axis=0)
        rx2, ry2 = r_pts.max(axis=0)
        rw, rh = max(12.0, rx2 - rx1), max(12.0, ry2 - ry1)

        px1 = max(0, int(rx1 - rw * 0.25))
        py1 = max(0, int(ry1 - rh * 0.25))
        px2 = min(w, int(rx2 + rw * 0.25))
        py2 = min(h, int(ry2 + rh * 0.25))

        patch = aligned_rgb[py1:py2, px1:px2]
        if patch.size == 0:
            patch = face_resized
        patch_resized = cv2.resize(patch, (REGION_SIZE, REGION_SIZE), interpolation=cv2.INTER_AREA)
        region_patches.append(patch_resized)

    regions_stack = np.stack(region_patches, axis=0).astype(np.uint8)  # Shape: (9, 48, 48, 3)
    return face_resized.astype(np.uint8), regions_stack


def process_single_video(video_path: Path, landmarker) -> list[dict]:
    video_id = video_path.stem
    records = []
    clip_samples = int(CLIP_DURATION * SAMPLE_RATE)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_wav = Path(tmpdir) / f"{video_id}.wav"
        if not extract_clean_audio(video_path, tmp_wav):
            return records

        audio, sr = sf.read(tmp_wav, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        total_samples = len(audio)
        if total_samples < clip_samples:
            return records

        try:
            audio = nr.reduce_noise(y=audio, sr=SAMPLE_RATE, stationary=True, prop_decrease=0.75)
        except Exception:
            pass

        margin_samples = min(int(2.0 * SAMPLE_RATE), max(0, (total_samples - clip_samples) // 4))
        usable_start = margin_samples
        usable_end = total_samples - clip_samples - margin_samples
        if usable_end <= usable_start:
            usable_start, usable_end = 0, total_samples - clip_samples

        starts = np.linspace(usable_start, usable_end, num=CLIPS_PER_VIDEO, dtype=int)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return records

        for idx, start_samp in enumerate(starts):
            seg = audio[start_samp : start_samp + clip_samples]

            rms = float(np.sqrt(np.mean(seg ** 2)))
            if rms < 0.008:
                continue

            peak = float(np.max(np.abs(seg)))
            if peak > 0:
                seg = (seg / peak) * 0.95

            mid_sec = (start_samp + clip_samples / 2.0) / SAMPLE_RATE
            cap.set(cv2.CAP_PROP_POS_MSEC, mid_sec * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            extracted = align_and_extract_9flr(frame, landmarker)
            if extracted is None:
                continue

            face_rgb, regions_stack = extracted
            sample_id = f"{video_id}_c{idx:02d}"

            aud_rel = f"audio/{sample_id}.wav"
            face_rel = f"faces/{sample_id}.png"
            reg_rel = f"regions_9flr/{sample_id}.npy"

            sf.write(PROCESSED_DIR / aud_rel, seg, SAMPLE_RATE, subtype="PCM_16")
            cv2.imwrite(str(PROCESSED_DIR / face_rel), cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR))
            np.save(PROCESSED_DIR / reg_rel, regions_stack)

            records.append({
                "sample_id": sample_id,
                "video_id": video_id,
                "clip_idx": idx,
                "mid_sec": round(mid_sec, 2),
                "audio_path": aud_rel,
                "face_path": face_rel,
                "regions_path": reg_rel,
            })

        cap.release()

    return records


def main():
    ensure_dirs_and_model()
    video_files = sorted(RAW_VIDEO_DIR.glob("*.mp4"))
    print(f"Found {len(video_files)} raw videos in {RAW_VIDEO_DIR}")

    done_videos = load_processed_videos()
    if done_videos:
        print(f"Resuming: {len(done_videos)} videos already processed.")

    file_exists = MANIFEST_PATH.exists()
    landmarker = create_face_landmarker()

    total_clips_saved = 0
    with MANIFEST_PATH.open("a", newline="", encoding="utf-8") as f:
        fieldnames = ["sample_id", "video_id", "clip_idx", "mid_sec", "audio_path", "face_path", "regions_path"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for vpath in tqdm(video_files, desc="Extracting Audio + 9FLR Faces"):
            if vpath.stem in done_videos:
                continue

            try:
                rows = process_single_video(vpath, landmarker)
                for r in rows:
                    writer.writerow(r)
                    total_clips_saved += 1
                f.flush()
            except Exception as e:
                print(f"\n[WARN] Skipped {vpath.name}: {e}")
            finally:
                gc.collect()

    landmarker.close()
    print(f"\nDone! Saved {total_clips_saved} new paired samples to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()