"""Optimizacija hiperparametara (Optuna + GroupKFold) i treniranje MLP / XGBoost modela."""

import argparse
import json
import time
import warnings

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.decomposition import PCA

from src.datasets import PROJECT_ROOT

FEATURES_DIR = PROJECT_ROOT / "features"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"   # .joblib je u .gitignore
HPO_DIR = RESULTS_DIR / "hpo"         # .json ide na Git

CNNS = ["resnet50", "convnext"]
ML_MODELS = ["mlp", "xgboost"]
DATASETS = ["appa", "fgnet", "morph", "utkface"]
SEED = 42

warnings.filterwarnings("ignore", category=ConvergenceWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_features(cnn: str, dataset: str):
    """Vraća X (karakteristike), y (godine) i groups (ID osobe), istim redosledom."""
    X = np.load(FEATURES_DIR / cnn / f"{dataset}.npy")
    meta = pd.read_csv(FEATURES_DIR / cnn / f"{dataset}_meta.csv")
    return X, meta["age"].to_numpy(dtype=np.float32), meta["person_id"].to_numpy()


def subsample_by_person(X, y, groups, max_samples: int, seed: int = SEED):
    """Nasumično bira CELE osobe dok ne skupi oko max_samples slika."""
    if len(y) <= max_samples:
        return X, y, groups
    rng = np.random.default_rng(seed)
    counts = pd.Series(groups).value_counts()
    selected, total = [], 0
    for person in rng.permutation(counts.index.to_numpy()):
        if total >= max_samples:
            break
        selected.append(person)
        total += counts[person]
    mask = np.isin(groups, selected)
    return X[mask], y[mask], groups[mask]

def suggest_params(trial, ml: str) -> dict:
    """Prostor pretrage hiperparametara (uključujući broj PCA komponenti)."""
    params = {
        "pca_components": trial.suggest_categorical("pca_components", [128, 256, 512]),
    }
    if ml == "mlp":
        params.update({
            "hidden_layers": trial.suggest_categorical(
                "hidden_layers", ["128", "256", "256-64", "512-128"]),
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        })
    else:
        params.update({
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        })
    return params


def build_model(ml: str, params: dict):
    """Pipeline: skaliranje -> PCA -> regresor."""
    params = dict(params)                        # kopija, da ne menjamo original
    n_components = params.pop("pca_components")  # izvadi PCA parametar iz rečnika

    steps = [
        StandardScaler(),
        PCA(n_components=n_components, whiten=True, random_state=SEED),
    ]
    if ml == "mlp":
        hidden = tuple(int(n) for n in params["hidden_layers"].split("-"))  # "256-64" -> (256, 64)
        steps.append(MLPRegressor(
            hidden_layer_sizes=hidden,
            alpha=params["alpha"],
            learning_rate_init=params["learning_rate_init"],
            batch_size=params["batch_size"],
            early_stopping=True,
            max_iter=300,
            random_state=SEED,
        ))
    else:
        steps.append(XGBRegressor(**params, tree_method="hist", n_jobs=-1, random_state=SEED))
    return make_pipeline(*steps)


def optimize(ml: str, X, y, groups, n_trials: int, n_folds: int):
    cv = GroupKFold(n_splits=n_folds)

    def objective(trial):
        model = build_model(ml, suggest_params(trial, ml))
        scores = cross_val_score(model, X, y, groups=groups, cv=cv,
                                 scoring="neg_mean_absolute_error")
        return -scores.mean()  # MAE, manje je bolje

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study


def main():
    parser = argparse.ArgumentParser(description="Optimizacija i treniranje")
    parser.add_argument("--cnn", nargs="+", default=CNNS, choices=CNNS)
    parser.add_argument("--ml", nargs="+", default=ML_MODELS, choices=ML_MODELS)
    parser.add_argument("--train", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--max-hpo-samples", type=int, default=5000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    HPO_DIR.mkdir(parents=True, exist_ok=True)

    for cnn in args.cnn:
        for train_name in args.train:
            X, y, groups = load_features(cnn, train_name)
            X_hpo, y_hpo, g_hpo = subsample_by_person(X, y, groups, args.max_hpo_samples)

            for ml in args.ml:
                tag = f"{cnn}_{ml}_{train_name}"
                model_path = MODELS_DIR / f"{tag}.joblib"
                if model_path.exists() and not args.overwrite:
                    print(f"[{tag}] već postoji, preskačem")
                    continue

                print(f"\n[{tag}] optimizacija na {len(y_hpo)} slika, "
                      f"{len(np.unique(g_hpo))} osoba")
                start = time.perf_counter()
                study = optimize(ml, X_hpo, y_hpo, g_hpo, args.trials, args.folds)

                model = build_model(ml, study.best_params)
                model.fit(X, y)  # konačno treniranje na CELOM skupu
                joblib.dump(model, model_path)

                elapsed = time.perf_counter() - start
                info = {
                    "cnn": cnn, "ml": ml, "train_dataset": train_name,
                    "best_params": study.best_params,
                    "cv_mae": round(study.best_value, 3),
                    "n_trials": args.trials, "n_folds": args.folds,
                    "hpo_samples": int(len(y_hpo)), "train_samples": int(len(y)),
                    "seconds": round(elapsed, 1),
                }
                with open(HPO_DIR / f"{tag}.json", "w", encoding="utf-8") as f:
                    json.dump(info, f, indent=2)

                print(f"[{tag}] CV MAE = {study.best_value:.2f} god., "
                      f"vreme {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()