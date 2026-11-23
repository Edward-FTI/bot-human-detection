from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_CANDIDATES = [
    # BASE_DIR / "models" / "decision_tree.pkl",
    BASE_DIR / "models" / "best_model.pkl",
    # BASE_DIR / "models" / "best_model_500k.pkl",
]


def load_model():
    """Memuat model yang disimpan di folder models."""
    for path in MODEL_CANDIDATES:
        if path.exists():
            return joblib.load(path)
    raise FileNotFoundError(
        "File model tidak ditemukan. Letakkan best_model.pkl "
        "(atau best_model_500k.pkl) di folder models/."
    )


def get_model_features(model):
    """
    Mengembalikan daftar fitur yang dibutuhkan model, sesuai urutan training.

    Model yang dilatih dari DataFrame menyimpan nama fitur pada
    feature_names_in_. Jika tidak ada, daftar ditentukan dari jumlah fitur.
    """
    from src.feature_engineering import MODEL_FEATURES, LEGACY_MODEL_FEATURES

    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        return list(names)

    n_features = getattr(model, "n_features_in_", None)
    for candidate in (MODEL_FEATURES, LEGACY_MODEL_FEATURES):
        if n_features == len(candidate):
            return list(candidate)
    raise ValueError(
        f"Model membutuhkan {n_features} fitur dan tidak menyimpan nama fitur, "
        "sehingga urutan fitur tidak dapat dipastikan."
    )


def describe_model(model):
    """Informasi singkat model untuk ditampilkan di dashboard."""
    from src.behavior_features import BEHAVIOR_FEATURES

    features = get_model_features(model)
    uses_behavior = all(column in features for column in BEHAVIOR_FEATURES)
    return {
        "n_features": len(features),
        "uses_behavior": uses_behavior,
        "version": "Baru (fitur perilaku per IP)" if uses_behavior else "Lama (tanpa fitur perilaku)",
    }


def predict_traffic(df_parsed: pd.DataFrame, model):
    """
    Membuat fitur dari hasil parsing, menyelaraskan kolom fitur dengan model,
    lalu mengembalikan data request beserta hasil prediksi.
    """
    from src.feature_engineering import create_features

    df_features, X_all = create_features(df_parsed.copy())

    expected_features = get_model_features(model)
    missing = [column for column in expected_features if column not in X_all.columns]
    if missing:
        raise ValueError(
            "Kolom fitur yang dibutuhkan model tidak tersedia: "
            + ", ".join(missing)
        )
    X = X_all.reindex(columns=expected_features)

    predictions = model.predict(X)

    results = df_parsed.copy()
    results["prediction"] = predictions
    results["prediction_label"] = results["prediction"].map(
        {0: "Human", 1: "Bot"}
    ).fillna(results["prediction"].astype(str))

    return results, df_features, X
