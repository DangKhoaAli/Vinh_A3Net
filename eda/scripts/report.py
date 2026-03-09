# scripts/report_structure.py

import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ===============================
# Paths
# ===============================

DATA_PATH = Path("data/iu_xray/annotation.json")
OUTPUT_DIR = Path(r"D:\HocTap\NCKH_ThayDoNhuTai\A3Net\eda\report")

OUTPUT_DIR.mkdir(exist_ok=True)

# ===============================
# Load dataset
# ===============================

with open(DATA_PATH) as f:
    data = json.load(f)

records = []

for split in data:
    for item in data[split]:

        report = item["report"]

        word_len = len(report.split())
        sent_len = len(report.split("."))

        records.append({
            "report": report,
            "word_length": word_len,
            "sentence_length": sent_len
        })

df = pd.DataFrame(records)

print("Total reports:", len(df))

# ===============================
# Statistics
# ===============================

stats = {
    "metric": [
        "avg_words",
        "max_words",
        "min_words",
        "avg_sentences"
    ],
    "value": [
        round(df["word_length"].mean(),2),
        df["word_length"].max(),
        df["word_length"].min(),
        round(df["sentence_length"].mean(),2)
    ]
}

stats_df = pd.DataFrame(stats)

csv_path = OUTPUT_DIR / "report_structure_stats.csv"

stats_df.to_csv(csv_path, index=False)

print("Saved:", csv_path)

# ===============================
# Plot length distribution
# ===============================

plt.figure(figsize=(8,5))

plt.hist(df["word_length"], bins=50)

plt.title("Report Length Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")

plt.tight_layout()

fig_path = OUTPUT_DIR / "report_length_distribution.png"

plt.savefig(fig_path)

print("Saved:", fig_path)