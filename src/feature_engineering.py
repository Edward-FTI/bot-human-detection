
import numpy as np
import pandas as pd

from src.behavior_features import BEHAVIOR_FEATURES, create_behavior_features


URL_CATEGORIES = [
    "article",
    "basket",
    "blog",
    "browse",
    "customer",
    "filter",
    "home",
    "image",
    "login",
    "order",
    "order_administration",
    "other",
    "product",
    "search",
    "settings",
    "site",
    "static",
]

URL_CATEGORY_COLUMNS = [f"url_category_{category}" for category in URL_CATEGORIES]

# Urutan fitur model baru = urutan kolom train.csv dari Notebook 01 (baru):
# fitur numerik sesuai FEATURE_COLUMNS, lalu kolom one-hot url_category_*.
# Fitur duplikat (is_home, is_login, is_product, is_static_resource,
# response_size) tidak lagi dipakai model baru.
MODEL_FEATURES = [
    "hour",
    "day_of_week",
    "is_get",
    "is_post",
    "status_code",
    "log_response_size",
    "url_length",
    "url_depth",
    "is_admin",
    "is_image",
    "requests_per_minute",
] + BEHAVIOR_FEATURES + URL_CATEGORY_COLUMNS

# Fitur model lama (Notebook 01 _lama, 33 fitur). Disimpan agar model lama
# masih bisa dijalankan untuk perbandingan.
LEGACY_MODEL_FEATURES = [
    "hour",
    "day_of_week",
    "is_get",
    "is_post",
    "status_code",
    "response_size",
    "log_response_size",
    "url_length",
    "url_depth",
    "is_home",
    "is_login",
    "is_product",
    "is_admin",
    "is_static_resource",
    "requests_per_minute",
    "is_image",
] + URL_CATEGORY_COLUMNS


def categorize_url(url):
    # Mengikuti Notebook 01 Tahap 6: pencocokan prefix bersifat case-sensitive
    # (tanpa .lower()), sehingga "/orderAdministration" dapat dikenali.
    url = str(url)
    if url == "/":
        return "home"

    categories = [
        ("static", "/static"),
        ("image", "/image"),
        ("filter", "/filter"),
        ("product", "/product"),
        ("settings", "/settings"),
        ("site", "/site"),
        ("browse", "/browse"),
        ("blog", "/blog"),
        ("order_administration", "/orderAdministration"),
        ("customer", "/customer"),
        ("article", "/article"),
        ("login", "/login"),
        ("basket", "/basket"),
        ("search", "/search"),
        ("order", "/order"),
    ]

    for category, prefix in categories:
        if url.startswith(prefix):
            return category

    return "other"


def create_features(df):
    df = df.copy()

    # Pecah request menjadi method, URL, dan versi HTTP
    parts = df["request"].fillna("").str.split(" ", n=2, expand=True)

    df["http_method"] = parts[0]
    df["url"] = parts[1] if 1 in parts.columns else ""
    df["http_version"] = parts[2] if 2 in parts.columns else ""

    # Timestamp
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        format="%d/%b/%Y:%H:%M:%S %z",
        errors="coerce",
    )

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    # Requests per minute berdasarkan IP dan 60 detik sebelumnya
    # (mengikuti Notebook 01 Tahap 5). Timestamp dikonversi ke UTC dan
    # satuan nanodetik secara eksplisit, sehingga konstanta 60_000_000_000
    # selalu berarti 60 detik apa pun resolusi datetime bawaan pandas.
    # Kolom "timestamp" asli tetap dipakai untuk hour/day_of_week (waktu lokal).
    df["_timestamp_ns"] = df["timestamp"].dt.tz_convert("UTC").dt.as_unit("ns")
    df["requests_per_minute"] = 0
    valid_timestamp = df["_timestamp_ns"].notna()
    for ip, group in df[valid_timestamp].groupby("ip_address", sort=False):
        group = group.sort_values("_timestamp_ns", kind="mergesort")
        timestamps = group["_timestamp_ns"].astype("int64").to_numpy()
        window_start = timestamps - 60_000_000_000
        left_positions = np.searchsorted(
            timestamps,
            window_start,
            side="left"
        )
        current_positions = np.arange(len(group))
        counts = current_positions - left_positions
        df.loc[group.index, "requests_per_minute"] = counts
    df = df.drop(columns=["_timestamp_ns"])

    # Method HTTP
    df["is_get"] = (df["http_method"] == "GET").astype(int)
    df["is_post"] = (df["http_method"] == "POST").astype(int)

    # Ukuran respons
    df["status_code"] = pd.to_numeric(
        df["status_code"], errors="coerce"
    )
    df["response_size"] = pd.to_numeric(
        df["response_size"], errors="coerce"
    ).fillna(0)

    df["log_response_size"] = np.log1p(df["response_size"])

    # URL
    df["url"] = df["url"].fillna("").astype(str)
    df["url_path"] = df["url"].str.split("?", n=1).str[0]

    df["url_length"] = df["url_path"].str.len()
    df["url_depth"] = df["url_path"].apply(
        lambda url: 0 if not url or url == "/" else
        len([part for part in url.strip("/").split("/") if part])
    )

    # Pola URL
    df["is_home"] = (df["url_path"] == "/").astype(int)
    df["is_login"] = df["url_path"].str.startswith("/login").astype(int)
    df["is_product"] = df["url_path"].str.startswith("/product").astype(int)
    df["is_admin"] = df["url_path"].str.startswith("/admin").astype(int)
    df["is_static_resource"] = (
        df["url_path"].str.startswith("/static").astype(int)
    )

    # Mengikuti Notebook 01 Tahap 7:
    # deteksi URL gambar berdasarkan segmen /image/ atau /images/
    # maupun ekstensi file gambar. Query string diabaikan.
    image_extensions = (
        ".jpg", ".jpeg", ".png", ".gif", ".svg",
        ".webp", ".ico", ".bmp",
    )
    url_lower = df["url_path"].str.lower()
    has_image_path = url_lower.str.contains(
        r"(?:^|/)images?(?:/|$)", regex=True, na=False
    )
    has_image_extension = url_lower.str.endswith(image_extensions)
    df["is_image"] = (has_image_path | has_image_extension).astype("int8")

    df["url_category"] = df["url_path"].apply(categorize_url)

    # Encoding harus mengikuti kolom yang dipakai model saat training
    df_encoded = pd.get_dummies(
        df,
        columns=["url_category"],
        dtype=int,
    )

    # Pastikan seluruh kategori yang diharapkan model tersedia.
    for category in URL_CATEGORIES:
        column = f"url_category_{category}"
        if column not in df_encoded.columns:
            df_encoded[column] = 0

    # Fitur perilaku per IP (Notebook 01 Tahap 5), dihitung dari seluruh
    # request pada log yang diunggah.
    behavior = create_behavior_features(df)
    for column in BEHAVIOR_FEATURES:
        df_encoded[column] = behavior[column]

    # X berisi gabungan kolom model baru dan model lama. predict_traffic()
    # memilih kolom sesuai model.feature_names_in_, sehingga dashboard tetap
    # bisa memakai model lama (33 fitur) maupun model baru (34 fitur).
    feature_columns = list(dict.fromkeys(MODEL_FEATURES + LEGACY_MODEL_FEATURES))

    X = df_encoded[feature_columns].copy()

    # Gagal lebih baik daripada mengirim nilai tidak valid ke model.
    if X.isna().any().any():
        raise ValueError(
            "Ada nilai kosong pada fitur model. "
            "Periksa timestamp dan status_code pada data log."
        )

    return df, X
