import zipfile
from pathlib import Path
from tqdm import tqdm

BUNDLE_PATH = Path("9flr_dataset_bundle.zip")
TARGETS = [Path("data/processed"), Path("data/splits")]


def main():
    files_to_pack = []
    for folder in TARGETS:
        if folder.exists():
            files_to_pack.extend([p for p in folder.rglob("*") if p.is_file()])

    print(f"Packing {len(files_to_pack)} files into {BUNDLE_PATH}...")
    with zipfile.ZipFile(BUNDLE_PATH, mode="w", compression=zipfile.ZIP_STORED) as zf:
        for file_path in tqdm(files_to_pack, desc="Creating Dataset Archive"):
            zf.write(file_path, arcname=file_path.as_posix())

    size_gb = BUNDLE_PATH.stat().st_size / (1024 ** 3)
    print(f"\nDone! Created {BUNDLE_PATH} ({size_gb:.2f} GB)")
    print("Upload this single file to Google Drive at: MyDrive/9FLR-Voice2Face/9flr_dataset_bundle.zip")


if __name__ == "__main__":
    main()