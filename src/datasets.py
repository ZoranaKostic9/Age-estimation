"""Učitavanje skupova podataka u zajednički format.

Svaki skup se pretvara u tabelu sa kolonama:
    image_path | age | person_id | dataset
"""

import re
from pathlib import Path

import pandas as pd

# Koren projekta = folder iznad src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
METADATA_PATH = PROJECT_ROOT / "data" / "metadata.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# 001A02.JPG, 001A43a.JPG  ->  osoba=001, godine=02
FGNET_PATTERN = re.compile(r"^(\d+)A(\d+)[a-z]?$", re.IGNORECASE)

# 00013_00M19.JPG  ->  osoba=00013, slika=00, pol=M, godine=19
MORPH_PATTERN = re.compile(r"^(\d+)_(\d+)([MF])(\d+)$", re.IGNORECASE)


def _list_images(folder: Path):
    """Vraća sve slike u folderu i podfolderima, sortirano."""
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _to_dataframe(rows, name):
    df = pd.DataFrame(rows, columns=["image_path", "age", "person_id"])
    # Putanje relativno u odnosu na koren projektat
    df["image_path"] = df["image_path"].map(
        lambda p: Path(p).relative_to(PROJECT_ROOT).as_posix()
    )
    df["dataset"] = name
    return df

def load_fgnet(root: Path = RAW_DIR / "fgnet") -> pd.DataFrame:
    rows = []
    for path in _list_images(root / "images"):
        match = FGNET_PATTERN.match(path.stem)
        if match is None:
            print(f"[fgnet] preskočen fajl: {path.name}")
            continue
        person_id, age = match.groups()
        rows.append((str(path), int(age), f"fgnet_{person_id}"))
    return _to_dataframe(rows, "fgnet")


def load_morph(root: Path = RAW_DIR / "morph") -> pd.DataFrame:
    rows = []
    # Učitava Train, Validation i Test zajedno
    for path in _list_images(root / "Dataset" / "Images"):
        match = MORPH_PATTERN.match(path.stem)
        if match is None:
            print(f"[morph] preskočen fajl: {path.name}")
            continue
        person_id, _image_idx, _gender, age = match.groups()
        rows.append((str(path), int(age), f"morph_{person_id}"))
    return _to_dataframe(rows, "morph")


def load_appa(root: Path = RAW_DIR / "appa") -> pd.DataFrame:
    labels = pd.read_csv(root / "labels.csv")
    rows = []
    for file_name, age in zip(labels["file_name"], labels["real_age"]):
        path = root / "final_files" / file_name
        if not path.exists():
            print(f"[appa] ne postoji slika: {file_name}")
            continue
        # Nema ID osobe -> svaka slika je posebna "osoba"
        rows.append((str(path), int(age), f"appa_{path.stem}"))
    return _to_dataframe(rows, "appa")


def prepare_utkface(root: Path = RAW_DIR / "utkface") -> None:
    """Jednom raspakuje slike iz .parquet fajlova u images/ i pravi labels.csv."""
    images_dir = root / "images"
    labels_path = root / "labels.csv"
    if labels_path.exists():
        return  # već raspakovano

    images_dir.mkdir(exist_ok=True)
    parts = []
    for parquet_path in sorted(root.glob("*.parquet")):
        df = pd.read_parquet(parquet_path)
        for image, file_name in zip(df["image"], df["file_name"]):
            # Hugging Face čuva sliku kao {"bytes": ..., "path": ...}
            data = image["bytes"] if isinstance(image, dict) else image
            (images_dir / file_name).write_bytes(data)
        parts.append(df[["file_name", "age"]])
        print(f"[utkface] raspakovano {len(df)} slika iz {parquet_path.name}")

    pd.concat(parts, ignore_index=True).to_csv(labels_path, index=False)


def load_utkface(root: Path = RAW_DIR / "utkface") -> pd.DataFrame:
    prepare_utkface(root)
    labels = pd.read_csv(root / "labels.csv")
    rows = []
    for file_name, age in zip(labels["file_name"], labels["age"]):
        path = root / "images" / file_name
        # Nema ID osobe -> svaka slika je posebna "osoba"
        rows.append((str(path), int(age), f"utkface_{file_name}"))
    return _to_dataframe(rows, "utkface")
   

LOADERS = {
    "fgnet": load_fgnet,
    "morph": load_morph,
    "appa": load_appa,
    "utkface": load_utkface,
}

def load_all() -> pd.DataFrame:
    return pd.concat([loader() for loader in LOADERS.values()], ignore_index=True)


def print_summary(df: pd.DataFrame) -> None:
    summary = df.groupby("dataset").agg(
        slika=("image_path", "count"),
        osoba=("person_id", "nunique"),
        min_god=("age", "min"),
        max_god=("age", "max"),
        prosek_god=("age", "mean"),
    ).round(1)
    print(summary.to_string())


if __name__ == "__main__":
    df = load_all()
    print_summary(df)
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(METADATA_PATH, index=False)
    print(f"\nSačuvano: {METADATA_PATH}")