
import re
import pandas as pd


# Pola parsing yang digunakan pada notebook preprocessing
LOG_PATTERN = re.compile(
    r"^(\S+)\s+(\S+)\s+(\S+)\s+"
    r"\[([^\]]+)\]\s+"
    r'"([^"]*)"\s+'
    r"(\d{3})\s+"
    r"(\S+)\s+"
    r'"([^"]*)"\s+'
    r'"([^"]*)"\s*'
    r"(.*)$"
)

LOG_COLUMNS = [
    "ip_address",
    "ident",
    "auth_user",
    "timestamp",
    "request",
    "status_code",
    "response_size",
    "referer",
    "user_agent",
    "extra",
]


def parse_access_log(uploaded_file):
    """
    Membaca file access log dan mengubahnya menjadi DataFrame.

    Returns:
        df: DataFrame hasil parsing
        summary: Ringkasan proses parsing
    """

    parsed_data = []
    failed_lines = 0
    total_lines = 0

    # Membaca file baris demi baris untuk menghindari
    # membaca seluruh isi file sebagai satu string.
    for raw_line in uploaded_file:
        total_lines += 1

        if isinstance(raw_line, bytes):
            line = raw_line.decode("utf-8", errors="ignore")
        else:
            line = str(raw_line)

        line = line.rstrip("\r\n")

        if not line.strip():
            continue

        match = LOG_PATTERN.match(line)

        if match:
            parsed_data.append(match.groups())
        else:
            failed_lines += 1

    df = pd.DataFrame(parsed_data, columns=LOG_COLUMNS)

    # Tipe data awal mengikuti struktur log.
    # Validasi dan cleaning dilakukan pada tahap berikutnya.
    summary = {
        "total_lines": total_lines,
        "parsed_rows": len(df),
        "failed_lines": failed_lines,
    }

    return df, summary
