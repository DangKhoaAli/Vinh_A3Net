import json
from collections import Counter
import pandas as pd

DATA_PATH = r"D:\HocTap\NCKH_ThayDoNhuTai\A3Net\data\iu_xray\annotation.json"

with open(DATA_PATH) as f:
    data = json.load(f)

texts = []

for split in data:
    for item in data[split]:
        texts.append(item["report"].lower())

words = []

for t in texts:
    words.extend(t.split())

counter = Counter(words)

vocab_df = pd.DataFrame(counter.most_common(100), columns=["word","count"])

vocab_df.to_csv(r"D:\HocTap\NCKH_ThayDoNhuTai\A3Net\eda\report\vocab_stats.csv", index=False)

print("Vocabulary size:", len(counter))