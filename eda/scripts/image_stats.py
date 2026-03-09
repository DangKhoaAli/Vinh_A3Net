import os
import json
import cv2
import pandas as pd

DATA_PATH = r"D:\HocTap\NCKH_ThayDoNhuTai\A3Net\data\iu_xray\annotation.json"
IMAGE_DIR = r"D:\HocTap\NCKH_ThayDoNhuTai\A3Net\data\iu_xray\images"

with open(DATA_PATH) as f:
    data = json.load(f)

sizes = []

for split in data:

    for item in data[split]:

        for img in item["image_path"]:

            path = os.path.join(IMAGE_DIR, img)

            image = cv2.imread(path)

            h,w,_ = image.shape

            sizes.append((h,w))

df = pd.DataFrame(sizes, columns=["height","width"])

print(df.describe())