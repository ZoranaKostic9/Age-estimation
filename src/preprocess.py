"""Predobrada: detekcija, poravnanje i isecanje lica na 224x224."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.datasets import METADATA_PATH, PROJECT_ROOT

MODEL_PATH = PROJECT_ROOT / "models" / "face_detection_yunet_2023mar.onnx"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_METADATA_PATH = PROJECT_ROOT / "data" / "metadata_processed.csv"

OUTPUT_SIZE = 224   # ulaz za ResNet50 i ConvNeXt
PAD_RATIO = 0.2     # crni okvir oko slike pre detekcije (pomaže kod tesno isečenih lica)
FACE_SCALE = 0.9  # manja vrednost = više konteksta (čelo, kosa) oko lica

# Referentne pozicije 5 tačaka za sliku 112x112 (standardni ArcFace šablon):
# levo oko, desno oko, vrh nosa, levi ugao usana, desni ugao usana (gledano na slici)
ARCFACE_112 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def make_template(size: int = OUTPUT_SIZE, face_scale: float = FACE_SCALE) -> np.ndarray:
    """Šablon skaliran na izlaznu veličinu i malo smanjen da stane više konteksta."""
    points = ARCFACE_112 * (size / 112)
    center = size / 2
    return ((points - center) * face_scale + center).astype(np.float32)


TEMPLATE = make_template()


def read_image(path: Path):
    """cv2.imread ne radi sa nekim Windows putanjama, pa čitamo bajtove pa dekodiramo."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path: Path, image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if ok:
        buffer.tofile(str(path))


def detect_landmarks(detector, image):
    """Vraća 5 tačaka najvećeg lica na slici, ili None ako lice nije nađeno."""
    h, w = image.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(image)
    if faces is None or len(faces) == 0:
        return None
    # Svaki red: x, y, širina, visina, 5 tačaka (10 brojeva), skor
    largest = max(faces, key=lambda f: f[2] * f[3])
    return largest[4:14].reshape(5, 2).astype(np.float32)


def align_face(image, landmarks):
    """Rotira, skalira i pomera sliku tako da se tačke poklope sa šablonom."""
    matrix, _ = cv2.estimateAffinePartial2D(landmarks, TEMPLATE, method=cv2.LMEDS)
    if matrix is None:
        return None
    # Delove van originalne slike popunjava ponavljanjem rubnih piksela umesto crnom
    return cv2.warpAffine(image, matrix, (OUTPUT_SIZE, OUTPUT_SIZE),
                          borderMode=cv2.BORDER_REFLECT_101)


def center_crop(image):
    """Rezervni plan: centralni kvadrat promenjen na 224x224."""
    h, w = image.shape[:2]
    side = min(h, w)
    top, left = (h - side) // 2, (w - side) // 2
    crop = image[top:top + side, left:left + side]
    return cv2.resize(crop, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_AREA)


def process_image(detector, image_path: Path):
    """Vraća (obrađena_slika, da_li_je_lice_detektovano)."""
    image = read_image(image_path)
    if image is None:
        return None, False

    # Crni okvir samo za detekciju (pomaže kod tesno isečenih lica)
    pad = int(PAD_RATIO * max(image.shape[:2]))
    padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)

    landmarks = detect_landmarks(detector, padded)
    if landmarks is not None:
        landmarks = landmarks - pad   # koordinate vraćamo na originalnu sliku
        aligned = align_face(image, landmarks)
        if aligned is not None:
            return aligned, True

    return center_crop(image), False

def main():
    parser = argparse.ArgumentParser(description="Predobrada lica")
    parser.add_argument("--limit", type=int, default=None,
                        help="obradi samo prvih N slika iz svakog skupa (za probu)")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Nema modela detektora: {MODEL_PATH}")

    detector = cv2.FaceDetectorYN.create(str(MODEL_PATH), "", (320, 320),
                                         score_threshold=0.6)

    df = pd.read_csv(METADATA_PATH)
    if args.limit:
        df = df.groupby("dataset").head(args.limit).reset_index(drop=True)

    processed_paths, detected_flags = [], []
    for image_path in tqdm(df["image_path"], desc="Predobrada"):
        # data/raw/fgnet/images/001A02.JPG -> data/processed/fgnet/images/001A02.jpg
        relative = Path(image_path).relative_to("data/raw").with_suffix(".jpg")
        out_path = PROCESSED_DIR / relative

        result, detected = process_image(detector, PROJECT_ROOT / image_path)
        if result is None:
            processed_paths.append(None)
            detected_flags.append(False)
            continue

        write_image(out_path, result)
        processed_paths.append(out_path.relative_to(PROJECT_ROOT).as_posix())
        detected_flags.append(detected)

    df["processed_path"] = processed_paths
    df["face_detected"] = detected_flags
    df.to_csv(PROCESSED_METADATA_PATH, index=False)

    print("\nProcenat slika sa detektovanim licem:")
    print((df.groupby("dataset")["face_detected"].mean() * 100).round(1).to_string())
    print(f"\nNečitljivih slika: {df['processed_path'].isna().sum()}")
    print(f"Sačuvano: {PROCESSED_METADATA_PATH}")


if __name__ == "__main__":
    main()