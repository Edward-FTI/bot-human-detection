import pandas as pd


def summarize_results(results: pd.DataFrame):
    """Menghasilkan ringkasan jumlah request, Human, Bot, dan IP unik."""
    total_requests = len(results)
    human_count = int((results["prediction_label"] == "Human").sum())
    bot_count = int((results["prediction_label"] == "Bot").sum())
    total_ip = int(results["ip_address"].nunique()) if "ip_address" in results else 0

    return {
        "total_requests": total_requests,
        "total_ip": total_ip,
        "human": human_count,
        "bot": bot_count,
        "human_pct": human_count / total_requests * 100 if total_requests else 0,
        "bot_pct": bot_count / total_requests * 100 if total_requests else 0,
    }


def build_ip_summary(results: pd.DataFrame):
    """Mengelompokkan hasil prediksi per IP."""
    if results.empty:
        return pd.DataFrame(
            columns=["IP Address", "Total Request", "Human", "Bot", "Bot %", "Dominan"]
        )

    grouped = results.groupby("ip_address")["prediction_label"].value_counts().unstack(fill_value=0)
    for label in ("Human", "Bot"):
        if label not in grouped.columns:
            grouped[label] = 0

    grouped["Total Request"] = grouped["Human"] + grouped["Bot"]
    grouped["Bot %"] = (grouped["Bot"] / grouped["Total Request"] * 100).round(1)
    grouped["Dominan"] = grouped.apply(
        lambda row: "Human" if row["Human"] > row["Bot"]
        else ("Bot" if row["Bot"] > row["Human"] else "Seimbang"),
        axis=1,
    )

    grouped = grouped.reset_index().rename(columns={"ip_address": "IP Address"})
    return grouped[["IP Address", "Total Request", "Human", "Bot", "Bot %", "Dominan"]].sort_values(
        ["Total Request", "Bot"], ascending=False
    ).reset_index(drop=True)


def build_top_bot_ip(results: pd.DataFrame, limit: int = 10):
    """Mengurutkan IP berdasarkan jumlah request yang diprediksi sebagai Bot."""
    if results.empty:
        return pd.DataFrame(columns=["IP Address", "Bot Request", "Total Request", "Bot %"])

    bot_rows = results[results["prediction_label"] == "Bot"]
    if bot_rows.empty:
        return pd.DataFrame(columns=["IP Address", "Bot Request", "Total Request", "Bot %"])

    bot_counts = bot_rows.groupby("ip_address").size().rename("Bot Request")
    total_counts = results.groupby("ip_address").size().rename("Total Request")
    top = pd.concat([bot_counts, total_counts], axis=1).fillna(0)
    top = top.sort_values(["Bot Request", "Total Request"], ascending=False).head(limit)
    top.index.name = "IP Address"
    top = top.reset_index()
    top["Bot %"] = (top["Bot Request"] / top["Total Request"] * 100).round(1)
    return top



def build_traffic_timeline(results: pd.DataFrame, interval: str = "h"):
    """Mengelompokkan request Human dan Bot berdasarkan waktu."""

    columns = ["Waktu", "Human", "Bot"]

    if results.empty or "timestamp" not in results:
        return pd.DataFrame(columns=columns)

    # Konversi timestamp sesuai format Apache Access Log
    timestamps = pd.to_datetime(
        results["timestamp"],
        format="%d/%b/%Y:%H:%M:%S %z",
        errors="coerce",
        utc=True
    )

    # Ambil data dengan timestamp valid
    valid = results.loc[timestamps.notna()].copy()

    if valid.empty:
        return pd.DataFrame(columns=columns)

    valid["Waktu"] = timestamps[timestamps.notna()].dt.floor(interval)

    # Hitung jumlah request Human dan Bot
    timeline = (
        valid.groupby(["Waktu", "prediction_label"])
        .size()
        .unstack(fill_value=0)
    )

    # Pastikan kedua kategori tersedia
    for label in ["Human", "Bot"]:
        if label not in timeline.columns:
            timeline[label] = 0

    timeline = timeline.reset_index()

    return timeline[["Waktu", "Human", "Bot"]].sort_values("Waktu")
