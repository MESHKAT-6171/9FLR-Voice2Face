from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset


class Voice2FaceDataset(Dataset):
    def __init__(self, csv_path: str | Path, processed_dir: str | Path, sample_rate: int = 16000, clip_sec: float = 4.0):
        self.df = pd.read_csv(csv_path, dtype={"video_id": str, "sample_id": str})
        self.processed_dir = Path(processed_dir)
        self.target_len = int(sample_rate * clip_sec)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        # 1. Load 16kHz mono audio waveform -> Tensor (target_len,)
        wav_path = self.processed_dir / row["audio_path"]
        audio, _ = sf.read(wav_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if len(audio) < self.target_len:
            audio = np.pad(audio, (0, self.target_len - len(audio)))
        else:
            audio = audio[: self.target_len]
        audio_tensor = torch.from_numpy(audio)

        # 2. Load aligned 128x128 RGB face -> Tensor (3, 128, 128) in [-1, 1]
        face_path = self.processed_dir / row["face_path"]
        import cv2
        bgr = cv2.imread(str(face_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        face_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 127.5 - 1.0

        # 3. Load 9FLR patches (9, 48, 48, 3) -> Tensor (9, 3, 48, 48) in [-1, 1]
        reg_path = self.processed_dir / row["regions_path"]
        regions = np.load(reg_path)
        regions_tensor = torch.from_numpy(regions).permute(0, 3, 1, 2).float() / 127.5 - 1.0

        return {
            "sample_id": row["sample_id"],
            "video_id": row["video_id"],
            "audio": audio_tensor,
            "face": face_tensor,
            "regions": regions_tensor,
        }