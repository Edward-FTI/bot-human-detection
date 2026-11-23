"""
Fitur perilaku per IP (behavioral features).

Fitur dihitung per IP dari seluruh request IP tersebut di dalam log,
lalu ditempelkan ke setiap baris request milik IP itu. Dengan begitu
model per-request tetap bisa "melihat" pola perilaku IP-nya.

Semua fitur dibuat agar tidak bergantung pada panjang log (log training
±5 hari, log dashboard ±1 bulan): jumlah request dihitung per hari aktif,
cakupan jam dihitung rata-rata per hari aktif, dan jeda memakai median.

Input  : DataFrame dengan kolom ip_address, timestamp (string format
         access log), dan request.
Output : DataFrame berisi kolom BEHAVIOR_FEATURES dengan index yang sama.

Nilai -1 dipakai sebagai penanda "tidak terdefinisi", misalnya IP yang
hanya punya 1 request sehingga jeda antar-request tidak bisa dihitung.

Fungsi ini identik dengan fungsi pada Notebook 01 (Tahap 5) dan wajib
dipakai juga di dashboard agar fitur saat training dan prediksi sama.
"""

import numpy as np
import pandas as pd


BEHAVIOR_FEATURES = [
    "ip_requests_per_day_log",  # log1p(jumlah request / jumlah hari aktif)
    "ip_interval_median_log",   # log1p(median jeda antar-request, detik)
    "ip_interval_cv",           # std/mean jeda antar-request (≈0 = sangat teratur)
    "ip_asset_ratio",           # proporsi request ke gambar/static/css/js/font
    "ip_unique_url_ratio",      # jumlah URL unik / jumlah request
    "ip_hour_coverage",         # rata-rata per hari: jam aktif berbeda / 24
]

ASSET_PATH = r"(?:^|/)(?:images?|static|css|js|fonts?)(?:/|$)"
ASSET_EXT = (
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico", ".bmp",
    ".css", ".js", ".woff", ".woff2", ".ttf",
)


def create_behavior_features(df):
    url = df["request"].fillna("").astype(str).str.split(" ", n=2).str[1]
    url_path = url.fillna("").str.split("?", n=1).str[0]
    url_lower = url_path.str.lower()

    ts = pd.to_datetime(
        df["timestamp"], format="%d/%b/%Y:%H:%M:%S %z", errors="coerce"
    )

    work = pd.DataFrame({
        "ip": df["ip_address"].to_numpy(),
        "ts_utc": ts.dt.tz_convert("UTC").to_numpy(),
        "date": ts.dt.date.to_numpy(),        # tanggal lokal
        "hour": ts.dt.hour.to_numpy(),        # jam lokal, sama dengan fitur hour
        "url_path": url_path.to_numpy(),
        "is_asset": (
            url_lower.str.contains(ASSET_PATH, regex=True, na=False)
            | url_lower.str.endswith(ASSET_EXT)
        ).astype("int8").to_numpy(),
    })

    # 1. Jeda antar-request per IP (urut waktu)
    ordered = work.sort_values(["ip", "ts_utc"], kind="mergesort")
    delta = ordered.groupby("ip", sort=False)["ts_utc"].diff().dt.total_seconds()
    interval = delta.groupby(ordered["ip"]).agg(["median", "mean", "std"])
    interval["cv"] = interval["std"] / interval["mean"]

    # 2. Cakupan jam per hari aktif
    hours_per_day = work.groupby(["ip", "date"])["hour"].nunique()
    per_day = hours_per_day.groupby(level="ip").agg(["mean", "size"])

    # 3. Agregasi lain per IP
    per_ip = work.groupby("ip").agg(
        n=("url_path", "size"),
        n_unique_url=("url_path", "nunique"),
        asset_ratio=("is_asset", "mean"),
    )

    out = pd.DataFrame(index=per_ip.index)
    out["ip_requests_per_day_log"] = np.log1p(per_ip["n"] / per_day["size"])
    out["ip_interval_median_log"] = np.log1p(interval["median"]).fillna(-1)
    # mean = 0 (semua request pada detik yang sama) -> cv tidak terdefinisi
    out["ip_interval_cv"] = (
        interval["cv"].replace([np.inf, -np.inf], np.nan).fillna(-1)
    )
    out["ip_asset_ratio"] = per_ip["asset_ratio"]
    out["ip_unique_url_ratio"] = per_ip["n_unique_url"] / per_ip["n"]
    out["ip_hour_coverage"] = per_day["mean"] / 24

    # 4. Tempelkan kembali ke setiap baris request
    result = out.reindex(work["ip"]).set_axis(df.index)
    return result[BEHAVIOR_FEATURES].astype("float64")
