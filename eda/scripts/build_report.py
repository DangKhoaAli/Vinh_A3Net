import json
import pandas as pd

# dataset json
json_path = r"D:\HocTap\NCKH_ThayDoNhuTai\A3Net\data\iu_xray\annotation.json"

with open(json_path, "r") as f:
    data = json.load(f)

rows = []

for split in data:

    for item in data[split]:

        rows.append({
            "id": item["id"],
            "report": item["report"],
            "split": split
        })

df = pd.DataFrame(rows)

# lowercase + clean
df["report"] = df["report"].str.lower()

# save
df.to_csv(
    r"D:\HocTap\NCKH_ThayDoNhuTai\A3Net\eda\report\reports_cleaned.csv",
    index=False
)

print("Saved: data/processed/reports_cleaned.csv")
print("Total reports:", len(df))