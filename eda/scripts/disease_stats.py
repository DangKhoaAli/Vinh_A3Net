# scripts/disease_stats.py

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
# Disease keywords
# ===============================

DISEASE_KEYWORDS = [
    "opacity",
    "effusion",
    "cardiomegaly",
    "atelectasis",
    "pneumonia",
    "nodule",
    "fracture",
    "edema",
]

# ===============================
# Load dataset
# ===============================

with open(DATA_PATH) as f:
    data = json.load(f)

records = []

for split in data:
    for item in data[split]:

        report = item["report"].lower()

        records.append({
            "report": report
        })

df = pd.DataFrame(records)

print("Total reports:", len(df))

# ===============================
# Compute disease frequency
# ===============================

results = []

for disease in DISEASE_KEYWORDS:

    count = df["report"].str.contains(disease).sum()

    results.append({
        "disease": disease,
        "count": int(count),
        "percentage": round(count / len(df) * 100, 2)
    })

disease_df = pd.DataFrame(results)

# sort descending
disease_df = disease_df.sort_values("count", ascending=False)

# ===============================
# Save CSV
# ===============================

csv_path = OUTPUT_DIR / "disease_stats.csv"

disease_df.to_csv(csv_path, index=False)

print("Saved:", csv_path)

# ===============================
# Plot distribution
# ===============================

plt.figure(figsize=(8,5))

plt.bar(
    disease_df["disease"],
    disease_df["count"]
)

plt.title("Disease Keyword Frequency")
plt.xlabel("Disease")
plt.ylabel("Number of Reports")

plt.tight_layout()

fig_path = OUTPUT_DIR / "disease_distribution.png"

plt.savefig(fig_path)

print("Saved:", fig_path)