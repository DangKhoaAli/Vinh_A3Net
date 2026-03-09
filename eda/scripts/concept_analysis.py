import pandas as pd
import itertools

# =========================
# Load dataset
# =========================

df = pd.read_csv(r"D:\HocTap\NCKH_ThayDoNhuTai\A3Net\eda\report\reports_cleaned.csv")

reports = df["report"].fillna("").str.lower()

# =========================
# Concept dictionary (từ paper)
# =========================

concepts = [
    "pneumothorax",
    "pleural",
    "spine",
    "heart",
    "hernia",
    "lung",
    "mediastinal",
    "cardiac",
    "bony",
    "emphysema",
    "atelectasis",
    "lobe",
    "clavicle",
    "cardiomediastinal",
    "osseous",
    "mediastinum",
    "aorta",
    "aortic",
    "diaphragm",
    "thoracic",
    "vascularity",
    "pulmonary"
]

# =========================
# 1️⃣ Concept Frequency
# =========================

rows = []

freq_dict = {}

for c in concepts:
    count = reports.str.contains(c).sum()
    freq_dict[c] = count

    rows.append({
        "section": "frequency",
        "concept": c,
        "value": int(count)
    })

# =========================
# 2️⃣ Coverage
# =========================

total_reports = len(reports)

reports_with_concept = reports.apply(
    lambda x: any(c in x for c in concepts)
).sum()

coverage_ratio = reports_with_concept / total_reports

rows.append({
    "section": "coverage",
    "concept": "total_reports",
    "value": total_reports
})

rows.append({
    "section": "coverage",
    "concept": "reports_with_concept",
    "value": reports_with_concept
})

rows.append({
    "section": "coverage",
    "concept": "coverage_ratio",
    "value": round(coverage_ratio, 4)
})

# =========================
# 3️⃣ Concept Co-occurrence
# =========================

for c1, c2 in itertools.combinations(concepts, 2):

    mask1 = reports.str.contains(c1)
    mask2 = reports.str.contains(c2)

    co_count = (mask1 & mask2).sum()

    if co_count > 0:
        rows.append({
            "section": "cooccurrence",
            "concept": f"{c1}|{c2}",
            "value": int(co_count)
        })

# =========================
# Save CSV
# =========================

result_df = pd.DataFrame(rows)

result_df.to_csv(
    r"D:\HocTap\NCKH_ThayDoNhuTai\A3Net\eda\report\concept_analysis.csv",
    index=False
)

print("Saved: reports/concept_analysis.csv")