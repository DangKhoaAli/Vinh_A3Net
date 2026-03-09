import json
import pandas as pd

DATA_PATH = r"D:\HocTap\NCKH_ThayDoNhuTai\A3Net\data\iu_xray\annotation.json"

with open(DATA_PATH) as f:
    data = json.load(f)

records = []

for split in ["train","val","test"]:

    for item in data[split]:

        records.append({
            "id": item["id"],
            "split": split,
            "num_images": len(item["image_path"]),
            "report_length": len(item["report"].split())
        })

df = pd.DataFrame(records)

summary = {
    "Metric": [
        "Total Images",
        "Total Reports",
        "Unique Patients",
        "Avg Report Length",
        "Max Report Length",
        "Min Report Length"
    ],
    "Value": [
        len(df),
        len(df),
        df["id"].nunique(),
        df["report_length"].mean(),
        df["report_length"].max(),
        df["report_length"].min()
    ]
}

summary_df = pd.DataFrame(summary)


summary_df.to_csv(r"D:\HocTap\NCKH_ThayDoNhuTai\A3Net\eda\report\dataset_summary.csv", index=False)

print(summary_df)