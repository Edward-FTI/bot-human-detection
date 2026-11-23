import io

import pandas as pd
import plotly.express as px
import streamlit as st

from src.log_parser import parse_access_log
from src.prediction import load_model, predict_traffic, describe_model
from src.analysis import (
    summarize_results,
    build_ip_summary,
    build_top_bot_ip,
    build_traffic_timeline,
)

HUMAN_COLOR = "#3B82F6"
BOT_COLOR = "#F97316"
MUTED = "#94A3B8"
PANEL = "#111B2D"
BORDER = "#293750"
DISPLAY_LIMIT = 100

st.set_page_config(
    page_title="Deteksi Trafik Bot dan Human",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Warna dasar diatur di .streamlit/config.toml; CSS hanya untuk detail kecil.
st.markdown(
    f"""
    <style>
    .main-title {{ font-size:30px; font-weight:700; margin-bottom:2px; }}
    .subtitle {{ color:{MUTED}; font-size:14px; margin-bottom:20px; }}
    .section-title {{ font-size:19px; font-weight:600; margin:18px 0 10px; }}
    div[data-testid="stMetric"] {{
        background:#172338; border:1px solid {BORDER};
        padding:14px; border-radius:11px;
    }}
    .legend {{ display:flex; justify-content:space-between; font-size:14px; margin-bottom:6px; }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model():
    return load_model()


def reset_results():
    for key in ("parsed_df", "parse_summary", "results", "ip_summary", "uploaded_name", "model_info"):
        st.session_state.pop(key, None)
    st.session_state["uploader_key"] = st.session_state.get("uploader_key", 0) + 1


def style_labels(df, columns):
    """Beri warna pada kolom label (Human/Bot/Seimbang)."""
    def color(value):
        c = {"Bot": BOT_COLOR, "Human": HUMAN_COLOR}.get(value, MUTED)
        return f"color:{c}; font-weight:600"
    styler = df.style
    apply = getattr(styler, "map", None) or styler.applymap
    return apply(color, subset=[c for c in columns if c in df.columns])


def format_timestamp(series):
    parsed = pd.to_datetime(series, format="%d/%b/%Y:%H:%M:%S %z", errors="coerce", utc=True)
    return parsed.dt.strftime("%d %b %Y %H:%M:%S").fillna(series)


def style_fig(fig):
    fig.update_layout(
        paper_bgcolor=PANEL, plot_bgcolor=PANEL, font_color="#CBD5E1",
        legend_title_text="", margin=dict(l=15, r=15, t=20, b=15),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor=BORDER)
    return fig


def download(label, df, name):
    st.download_button(label, df.to_csv(index=False).encode("utf-8-sig"),
                       file_name=name, mime="text/csv")


# ==========================================================
# SIDEBAR
# ==========================================================
with st.sidebar:
    st.markdown("## 🛡️ Traffic Detector")
    st.caption("Analisis web server log")
    page = st.radio("Menu", ["Dashboard", "Tentang Sistem"], label_visibility="collapsed")
    st.divider()
    st.markdown("**Model**")
    st.write("Decision Tree")
    model_info = st.session_state.get("model_info")
    if model_info:
        st.caption(f'{model_info["version"]} · {model_info["n_features"]} fitur')
    st.markdown("**File aktif**")
    st.write(st.session_state.get("uploaded_name", "Belum ada file"))
    st.markdown("**Status**")
    st.write("Hasil siap" if "results" in st.session_state else "Menunggu analisis")

if page == "Tentang Sistem":
    st.markdown('<div class="main-title">Tentang Sistem</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Informasi sistem deteksi trafik Bot dan Human.</div>',
                unsafe_allow_html=True)
    with st.container(border=True):
        st.subheader("Deskripsi")
        st.write("Sistem menerima web server access log, mem-parsing isinya, "
                 "lalu mengklasifikasikan setiap request sebagai Human atau Bot.")
        st.subheader("Fitur Dashboard")
        st.write("Unggah file log, lihat deteksi per request dan per IP, "
                 "grafik trafik berdasarkan waktu, dan daftar IP bot teratas.")
    st.stop()

# ==========================================================
# HEADER & UPLOAD
# ==========================================================
st.markdown('<div class="main-title">Deteksi Trafik Bot dan Human</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Analisis trafik berdasarkan web server access log.</div>',
            unsafe_allow_html=True)

with st.container(border=True):
    st.markdown("### Unggah Access Log")
    uploaded = st.file_uploader(
        "Pilih file access log",
        type=["log", "txt"],
        help="Format yang didukung: .log atau .txt (Apache/Nginx combined log).",
        key=f"uploader_{st.session_state.get('uploader_key', 0)}",
    )

    if uploaded is not None:
        st.caption(f"{uploaded.name} · {uploaded.size / (1024 * 1024):.2f} MB")
        c_run, c_reset = st.columns([4, 1])
        run = c_run.button("Analisis", type="primary", use_container_width=True)
        c_reset.button("Reset", on_click=reset_results, use_container_width=True)

        if run:
            try:
                with st.status("Menganalisis log...", expanded=True) as status:
                    st.write("1/2 Membaca dan mem-parsing file log")
                    parsed_df, parse_summary = parse_access_log(io.BytesIO(uploaded.getvalue()))
                    if parsed_df.empty:
                        raise ValueError("Tidak ada baris yang cocok dengan format access log.")
                    st.write("2/2 Membuat fitur dan menjalankan model deteksi")
                    model = get_model()
                    results, _, _ = predict_traffic(parsed_df, model)
                    st.session_state.update(
                        parsed_df=parsed_df, parse_summary=parse_summary,
                        results=results, ip_summary=build_ip_summary(results),
                        uploaded_name=uploaded.name,
                        model_info=describe_model(model),
                    )
                    status.update(label="Analisis selesai", state="complete", expanded=False)
            except FileNotFoundError as exc:
                st.error(f"Model belum tersedia. {exc}")
            except Exception as exc:
                st.error(f"Analisis gagal: {exc}")
    else:
        st.info("Unggah file log untuk memulai analisis.")

    if "parse_summary" in st.session_state:
        ps = st.session_state["parse_summary"]
        failed_pct = ps["failed_lines"] / ps["total_lines"] * 100 if ps["total_lines"] else 0
        with st.expander("Ringkasan parsing", expanded=False):
            p1, p2, p3 = st.columns(3)
            p1.metric("Baris dibaca", f'{ps["total_lines"]:,}')
            p2.metric("Berhasil diparsing", f'{ps["parsed_rows"]:,}')
            p3.metric("Gagal diparsing", f'{ps["failed_lines"]:,}')
            st.dataframe(st.session_state["parsed_df"].head(10),
                         use_container_width=True, hide_index=True)
            download("Unduh hasil parsing (CSV)", st.session_state["parsed_df"], "hasil_parsing.csv")
        if failed_pct > 0:
            st.warning(f"{ps['failed_lines']:,} baris ({failed_pct:.2f}%) tidak sesuai format log dan dilewati.")

    model_info = st.session_state.get("model_info")
    if model_info and not model_info["uses_behavior"]:
        st.warning(
            "Model yang dipakai masih model lama (tanpa fitur perilaku per IP). "
            "Bot yang request-nya mirip manusia, misalnya UptimeRobot dengan GET /, "
            "bisa terprediksi Human. Ganti models/best_model.pkl dengan model hasil "
            "training ulang Notebook 01_preprocessing_baru dan 03_tuning_baru."
        )

# ==========================================================
# HASIL
# ==========================================================
results = st.session_state.get("results")
if results is None:
    st.stop()

ip_summary = st.session_state["ip_summary"]
summary = summarize_results(results)

st.markdown('<div class="section-title">Ringkasan Hasil</div>', unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total request", f'{summary["total_requests"]:,}')
m2.metric("Total IP", f'{summary["total_ip"]:,}')
m3.metric("Trafik Human", f'{summary["human"]:,}')
m3.caption(f'{summary["human_pct"]:.2f}% dari total request')
m4.metric("Trafik Bot", f'{summary["bot"]:,}')
m4.caption(f'{summary["bot_pct"]:.2f}% dari total request')

hp, bp = summary["human_pct"], summary["bot_pct"]
st.markdown(
    f'<div class="legend"><span style="color:{HUMAN_COLOR}">Human {hp:.2f}%</span>'
    f'<span style="color:{BOT_COLOR}">Bot {bp:.2f}%</span></div>'
    f'<div style="display:flex; height:14px; background:{BORDER}; border-radius:20px; '
    f'overflow:hidden; margin-bottom:8px;">'
    f'<div style="width:{hp}%; background:{HUMAN_COLOR}"></div>'
    f'<div style="width:{bp}%; background:{BOT_COLOR}"></div></div>',
    unsafe_allow_html=True,
)

# tab_time, tab_req, tab_ip, tab_top = st.tabs(
#     ["Timeline", "Per Request", "Per IP", "Top Bot IP"]
# )
tab_time, tab_req, tab_ip = st.tabs(
    ["Timeline", "Per Request", "Per IP"]
)

# ---------- Timeline ----------
with tab_time:
    # Timeline selalu dikelompokkan per hari; sumbu Y menggunakan skala linear.
    timeline = build_traffic_timeline(results, interval="D")
    if timeline.empty:
        st.warning("Timestamp yang valid tidak ditemukan untuk membuat grafik.")
    else:
        chart_data = timeline.melt(id_vars="Waktu", value_vars=["Human", "Bot"],
                                   var_name="Jenis trafik", value_name="Jumlah request")
        fig = px.line(chart_data, x="Waktu", y="Jumlah request", color="Jenis trafik",
                      markers=True, color_discrete_map={"Human": HUMAN_COLOR, "Bot": BOT_COLOR})
        fig.update_layout(height=380, hovermode="x unified")
        fig.update_xaxes(title_text="Waktu (UTC)")
        st.plotly_chart(style_fig(fig), use_container_width=True)

# ---------- Per request ----------
with tab_req:
    names = {"ip_address": "IP Address", "timestamp": "Timestamp", "request": "Request",
             "status_code": "Status", "prediction_label": "Hasil Deteksi"}
    cols = [c for c in names if c in results.columns]

    # Tabel dibatasi 100 baris agar tampilan tetap ringan.
    table = results[cols].head(DISPLAY_LIMIT).rename(columns=names).copy()
    if "Timestamp" in table:
        table["Timestamp"] = format_timestamp(table["Timestamp"])
    st.dataframe(style_labels(table, ["Hasil Deteksi"]), use_container_width=True, hide_index=True)
    st.caption(f"Menampilkan {len(table):,} dari {len(results):,} request. Unduhan mencakup seluruh request.")

    # Unduhan tetap menggunakan seluruh hasil prediksi, bukan hanya 100 baris tabel.
    download("Unduh seluruh hasil per request", results[cols], "hasil_deteksi_per_request.csv")

# ---------- Per IP ----------
with tab_ip:
    # Tabel hanya menampilkan 100 IP pertama.
    ip_view = ip_summary.head(DISPLAY_LIMIT).copy()
    ip_view = ip_view[["IP Address", "Total Request", "Dominan"]].rename(
        columns={"Dominan": "Hasil Prediksi"}
    )
    st.dataframe(
        style_labels(ip_view, ["Hasil Prediksi"]),
        use_container_width=True, hide_index=True,
    )
    st.caption(f"Menampilkan {len(ip_view):,} dari {len(ip_summary):,} IP. Unduhan mencakup seluruh IP.")

    # Unduhan berisi seluruh IP dengan kolom yang sudah disederhanakan.
    ip_download = ip_summary[["IP Address", "Total Request", "Dominan"]].rename(
        columns={"Dominan": "Hasil Prediksi"}
    )
    download("Unduh seluruh hasil per IP", ip_download, "hasil_deteksi_per_ip.csv")

# ---------- Top Bot IP ----------
# with tab_top:
#     top_bot = build_top_bot_ip(results, limit=10)
#     if top_bot.empty:
#         st.info("Tidak ada request yang terdeteksi sebagai Bot.")
#     else:
#         fig = px.bar(top_bot.iloc[::-1], x="Bot Request", y="IP Address", orientation="h",
#                      color_discrete_sequence=[BOT_COLOR])
#         fig.update_layout(height=max(280, 32 * len(top_bot)))
#         fig.update_yaxes(type="category", title_text="")
#         st.plotly_chart(style_fig(fig), use_container_width=True)

st.divider()
st.caption("Deteksi Trafik Bot dan Human · Machine Learning Research Project")
