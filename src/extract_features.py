"""Ekstrakcija karakteristika pomoću pretreniranih CNN mreža (bez poslednjeg sloja)."""

import argparse
import time

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from timm.data import resolve_model_data_config
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from src.datasets import PROJECT_ROOT

PROCESSED_METADATA_PATH = PROJECT_ROOT / "data" / "metadata_processed.csv"
FEATURES_DIR = PROJECT_ROOT / "features"

# Kratko ime -> tačan naziv težina u timm biblioteci (obe pretrenirane na ImageNet-1k)
MODELS = {
    "resnet50": "resnet50.tv2_in1k",
    "convnext": "convnext_tiny.fb_in1k",
}


class FaceDataset(Dataset):
    """Učitava obrađene slike jednu po jednu i priprema ih za mrežu."""

    def __init__(self, paths, transform):
        self.paths = list(paths)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(PROJECT_ROOT / self.paths[idx]).convert("RGB")
        return self.transform(image)


def build_model(name: str):
    # num_classes=0 uklanja poslednji (klasifikacioni) sloj -> izlaz je vektor karakteristika
    model = timm.create_model(MODELS[name], pretrained=True, num_classes=0)
    model.eval()

    # Slike su već 224x224, pa radimo samo normalizaciju koju mreža očekuje
    config = resolve_model_data_config(model)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=config["mean"], std=config["std"]),
    ])
    return model, transform


@torch.inference_mode()
def extract(model, loader) -> np.ndarray:
    batches = []
    for images in tqdm(loader, desc="Ekstrakcija", leave=False):
        batches.append(model(images).numpy())
    return np.concatenate(batches).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Ekstrakcija karakteristika")
    parser.add_argument("--model", choices=list(MODELS), required=True)
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="npr. --datasets fgnet morph (podrazumevano: svi)")
    parser.add_argument("--limit", type=int, default=None,
                        help="samo prvih N slika po skupu, za merenje brzine (ne čuva rezultat)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(PROCESSED_METADATA_PATH)
    df = df[df["face_detected"]].reset_index(drop=True)  # izbacujemo slike bez lica

    model, transform = build_model(args.model)
    out_dir = FEATURES_DIR / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in args.datasets or sorted(df["dataset"].unique()):
        subset = df[df["dataset"] == name].reset_index(drop=True)
        features_path = out_dir / f"{name}.npy"
        meta_path = out_dir / f"{name}_meta.csv"

        if args.limit:
            subset = subset.head(args.limit)
        elif features_path.exists() and not args.overwrite:
            print(f"[{name}] već postoji, preskačem")
            continue

        loader = DataLoader(FaceDataset(subset["processed_path"], transform),
                            batch_size=args.batch_size, num_workers=args.workers)

        start = time.perf_counter()
        features = extract(model, loader)
        elapsed = time.perf_counter() - start
        print(f"[{name}] {len(subset)} slika -> {features.shape}, "
              f"{len(subset) / elapsed:.1f} slika/s")

        if args.limit:
            continue  # proba brzine, ne čuvamo

        np.save(features_path, features)
        subset[["processed_path", "age", "person_id", "dataset"]].to_csv(meta_path, index=False)


if __name__ == "__main__":
    main()