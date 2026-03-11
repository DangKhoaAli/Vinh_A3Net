# Hướng Dẫn EDA Chi Tiết Cho Radiology Report Generation (IU X-Ray)

Dựa trên phân tích cấu trúc dữ liệu và project A3Net của bạn, dưới đây là hướng dẫn EDA toàn diện:

---

## **PHẦN 1: TỔNG QUAN DATASET**

### 1.1 Thống kê cơ bản (Đã có sẵn trong dataset_summary.csv)
- Tổng số ảnh: **2,955** ảnh X-ray
- Tổng số báo cáo: **2,955** báo cáo
- Số bệnh nhân duy nhất: **2,955** (mỗi bệnh nhân 1 nghiên cứu)
- Độ dài trung bình báo cáo: **31 từ**
- Độ dài báo cáo: Min=7, Max=149 từ

### 1.2 Kiểm tra data quality
```python
# Kiểm tra missing images
missing_images = []
for _, row in df.iterrows():
    for img_path in row['image_path']:
        full_path = os.path.join(IMAGE_DIR, img_path)
        if not os.path.exists(full_path):
            missing_images.append(img_path)

# Kiểm tra duplicate reports
duplicate_reports = df[df.duplicated(subset=['report'], keep=False)]

# Kiểm tra data leakage giữa các splits
patient_split = df.groupby('patient_id')['split'].nunique()
leakage_patients = patient_split[patient_split > 1]
```

---

## **PHẦN 2: PHÂN TÍCH ẢNH (IMAGE ANALYSIS)**

### 2.1 Kích thước và độ phân giải ảnh
```python
# Thu thập kích thước ảnh
widths, heights = [], []
for img in df["image"]:
    image = Image.open(path)
    w, h = image.size
    widths.append(w)
    heights.append(h)

print(f"Width: min={min(widths)}, max={max(widths)}, mean={np.mean(widths):.2f}")
print(f"Height: min={min(heights)}, max={max(heights)}, mean={np.mean(heights):.2f}")
```

### 2.2 Phân tích pixel intensity
```python
# Phân tích phân bố cường độ pixel (quan trọng cho X-ray)
pixels = []
for img in df["image"][:]:
    image =500 Image.open(path).convert("L")  # Grayscale
    pixels.extend(image.flatten())

plt.hist(pixels, bins=256, range=(0, 256))
plt.title("Pixel Intensity Distribution")
plt.xlabel("Intensity")
plt.ylabel("Frequency")
```

### 2.3 Kiểm tra số ảnh mỗi báo cáo
- Mỗi báo cáo có **2 ảnh** (frontal + lateral view)
- Phân bố: 100% có 2 ảnh

### 2.4 Phân tích nâng cao (Kỹ thuật chuyên sâu)

#### a) Image Quality Assessment
```python
# Tính toán các chỉ số chất lượng ảnh
from PIL import ImageStat

def calculate_image_stats(image_path):
    img = Image.open(image_path).convert("L")
    stat = ImageStat.Stat(img)
    return {
        'mean': stat.mean[0],
        'stddev': stat.stddev[0],
        'contrast': stat.stddev[0] / (stat.mean[0] + 1e-6),
        'sharpness': img.resize((256,256)).filter(ImageFilter.SHARPEN).getextrema()
    }
```

#### b) Brightness và Contrast Analysis
```python
# Phân tích độ sáng và độ tương phản
def analyze_brightness_contrast(image):
    img_array = np.array(image)
    brightness = np.mean(img_array)
    contrast = np.std(img_array)
    return brightness, contrast
```

#### c) Edge Detection Analysis
```python
# Phân tích cạnh (quan trọng cho việc phát hiện abnormalities)
import cv2

def edge_density(image_path):
    img = cv2.imread(image_path, 0)
    edges = cv2.Canny(img, 100, 200)
    return np.sum(edges > 0) / edges.size
```

#### d) Lung Region Analysis
```python
# Phân tích vùng phổi (sử dụng thresholding)
def lung_region_analysis(image_path):
    img = cv2.imread(image_path, 0)
    # Otsu's thresholding
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lung_pixels = np.sum(binary < 128)  # Vùng phổi thường tối hơn
    return lung_pixels / binary.size
```

---

## **PHẦN 3: PHÂN TÍCH VĂN BẢN (TEXT ANALYSIS)**

### 3.1 Phân tích độ dài báo cáo
- Trung bình: **31 từ/report**
- Phân bố: Lệch phải (right-skewed), tập trung 20-40 từ

### 3.2 Phân tích số câu
- Trung bình: **4.58 câu/report**
- Phần lớn: 3-6 câu

### 3.3 Phân tích từ vựng
- Tổng số từ: ~91,000 từ
- Vocabulary size: ~1,500 từ duy nhất
- Top words: "the", "no", "normal", "heart", "lungs", "pleural"

### 3.4 Phân tích từ khóa y khoa (Disease Keywords)
```python
diseases = [
    "atelectasis", "cardiomegaly", "consolidation", "edema",
    "effusion", "emphysema", "fibrosis", "hernia", "infiltration",
    "mass", "nodule", "pleural", "pneumonia", "pneumothorax"
]

for d in diseases:
    count = df["report"].str.contains(d, case=False).sum()
    print(f"{d}: {count} ({count/len(df)*100:.2f}%)")
```

### 3.5 Phân tích nâng cao (Kỹ thuật chuyên sâu)

#### a) Medical Concept Extraction
```python
# Trích xuất các khái niệm y khoa sử dụng medical NER
# Có thể dùng: scispacy, medspacy, ClinicalBERT

import spacy
nlp = spacy.load("en_core_sci_md")

def extract_medical_concepts(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ in ['DISORDER', 'ANATOMY']]
```

#### b) Report Structure Analysis
```python
# Phân tích cấu trúc báo cáo y khoa
# Thường có các section: Findings, Impression

def analyze_report_structure(report):
    sections = {}
    if "findings" in report.lower():
        sections['findings'] = True
    if "impression" in report.lower():
        sections['impression'] = True
    return sections
```

#### c) Negation Detection
```python
# Phát hiện negation (rất quan trọng trong X-ray reports)
# "No pleural effusion" vs "Pleural effusion present"

import re

negation_patterns = [
    r'\bno\b', r'\bnot\b', r'\bnegative\b', 
    r'\bwithout\b', r'\babsence\b', r'\bclear\b'
]

def detect_negation(text):
    return any(re.search(p, text.lower()) for p in negation_patterns)
```

#### d) N-gram Analysis
```python
# Phân tích bigrams và trigrams
from sklearn.feature_extraction.text import CountVectorizer

# Bigrams
vectorizer_bi = CountVectorizer(ngram_range=(2,2), stop_words='english')
X_bi = vectorizer_bi.fit_transform(df["report"])
bigrams = sorted(zip(vectorizer_bi.get_feature_names_out(), X_bi.sum(axis=0).A1), key=lambda x: x[1], reverse=True)

# Trigrams  
vectorizer_tri = CountVectorizer(ngram_range=(3,3), stop_words='english')
X_tri = vectorizer_tri.fit_transform(df["report"])
trigrams = sorted(zip(vectorizer_tri.get_feature_names_out(), X_tri.sum(axis=0).A1), key=lambda x: x[1], reverse=True)
```

#### e) Radiology-specific patterns
```python
# Các pattern đặc trưng trong báo cáo X-ray
patterns = {
    'normal_findings': r'(normal|within normal limits|unremarkable)',
    'location_mentions': r'(upper lobe|lower lobe|mid|lateral|medial|bilateral)',
    'severity_modifiers': r'(mild|moderate|severe|minimal)',
    'comparison': r'(unchanged|from prior|compared to)',
    'device_mention': r'(tube|catheter|pacemaker|line)'
}
```

---

## **PHẦN 4: PHÂN TÍCH MỐI QUAN HỆ ẢNH - VĂN BẢN**

### 4.1 Image-Text Alignment
```python
# Kiểm tra mối quan hệ giữa số ảnh và độ dài báo cáo
plt.scatter(df['num_images'], df['report_length'], alpha=0.5)
plt.xlabel('Number of Images')
plt.ylabel('Report Length')
```

### 4.2 Multi-view Consistency
```python
# Kiểm tra tính nhất quán giữa các view (frontal vs lateral)
# Mỗi report có 2 ảnh - cần đảm bảo consistency
```

### 4.3 Concept-level Alignment
```python
# Map concepts trong text với regions trong ảnh
# Sử dụng: bounding box annotations (nếu có)
# Hoặc: attention maps từ trained model
```

---

## **PHẦN 5: PHÂN TÍCH PHÂN BỐ BỆNH**

### 5.1 Disease Distribution (Từ file disease_stats.csv)
| Disease | Count | Percentage |
|---------|-------|------------|
| Effusion | 2,291 | 77.53% |
| Opacity | 322 | 10.9% |
| Atelectasis | 191 | 6.46% |
| Nodule | 179 | 6.06% |
| Edema | 175 | 5.92% |
| Fracture | 139 | 4.7% |
| Cardiomegaly | 114 | 3.86% |
| Pneumonia | 108 | 3.65% |

### 5.2 Class Imbalance Analysis
```python
# Phân tích class imbalance - rất quan trọng cho training
imbalance_ratio = disease_counts.max() / disease_counts.min()
print(f"Imbalance ratio: {imbalance_ratio:.2f}")
```

### 5.3 Multi-label Distribution
```python
# Một report có thể chứa nhiều bệnh
df['num_diseases'] = df['report'].apply(
    lambda x: sum(1 for d in diseases if d in x.lower())
)
plt.hist(df['num_diseases'])
plt.title("Number of Diseases per Report")
```

---

## **PHẦN 6: DATA PREPROCESSING CHO MODEL**

### 6.1 Text Preprocessing Pipeline
```python
import re
import string

def preprocess_report(text):
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove placeholders (XXXX thường dùng để ẩn PHI)
    text = re.sub(r'x{4,}', '', text)
    
    # 3. Remove extra whitespace
    text = ' '.join(text.split())
    
    # 4. Standardize medical terms
    text = text.replace('rt.', 'right')
    text = text.replace('lt.', 'left')
    
    return text

# Tokenization với nltk
from nltk.tokenize import word_tokenize
tokens = word_tokenize(preprocessed_text)
```

### 6.2 Image Preprocessing
```python
# Các bước chuẩn bị ảnh cho model
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# Với X-ray, có thể thêm các bước:
# - Histogram equalization
# - Contrast enhancement
# - Lung segmentation (nếu có mask)
```

### 6.3 Vocabulary Building
```python
# Xây dựng vocabulary từ training data
# Threshold từ xuất hiện ít nhất 3 lần (theo args.threshold = 3)
from collections import Counter

word_counts = Counter()
for report in train_reports:
    word_counts.update(report.split())

vocab = [word for word, count in word_counts.items() if count >= 3]
vocab = sorted(vocab)
```

---

## **PHẦN 7: CÁC KỸ THUẬT NÂNG CAO**

### 7.1 Data Augmentation cho Ảnh X-ray
```python
# Image augmentation techniques
from torchvision.transforms import functional as TF

def augment_image(image):
    # Random rotation (±15 độ)
    angle = random.uniform(-15, 15)
    image = TF.rotate(image, angle)
    
    # Random horizontal flip
    if random.random() > 0.5:
        image = TF.hflip(image)
    
    # Random brightness/contrast
    brightness = random.uniform(0.9, 1.1)
    contrast = random.uniform(0.9, 1.1)
    image = TF.adjust_brightness(image, brightness)
    image = TF.adjust_contrast(image, contrast)
    
    # Random noise
    noise = torch.randn_like(image) * 0.01
    image = image + noise
    
    return image
```

### 7.2 Text Data Augmentation
```python
# Back-translation augmentation
# Thay đổi từ đồng nghĩa y khoa
# Synonym replacement cho các medical terms

# Ví dụ:
# "pneumothorax" -> "air in pleural space"
# "effusion" -> "fluid in pleural space"
```

### 7.3 Class Rebalancing Techniques
```python
# Oversampling cho minority classes
# Hoặc: Weighted loss function

# Tính class weights
class_counts = df['num_diseases'].value_counts()
weights = 1.0 / class_counts
sample_weights = df['num_diseases'].map(weights)
```

### 7.4 Report Simplification
```python
# Chuẩn hóa báo cáo - loại bỏ các biến thể không cần thiết
def normalize_report(report):
    # Standardize các abbreviations
    abbrevs = {
        'pt': 'patient',
        'dx': 'diagnosis',
        'hx': 'history',
        'cxr': 'chest x-ray',
        'pa': 'posteroanterior',
        'ap': 'anteroposterior'
    }
    
    for abbr, full in abbrevs.items():
        report = re.sub(rf'\b{abbr}\b', full, report)
    
    return report
```

---

## **PHẦN 8: CHECKLIST EDA HOÀN CHỈNH**

### 8.1 Data Quality Checks
- [ ] Missing images
- [ ] Duplicate reports
- [ ] Data leakage giữa splits
- [ ] Invalid image formats
- [ ] Corrupted images
- [ ] Empty or very short reports

### 8.2 Statistical Analysis
- [ ] Distribution of report lengths
- [ ] Distribution of sentence counts
- [ ] Vocabulary size và frequency
- [ ] Top bigrams và trigrams
- [ ] Disease keyword frequencies
- [ ] Class imbalance analysis

### 8.3 Image Analysis
- [ ] Image dimensions distribution
- [ ] Pixel intensity distribution
- [ ] Brightness/contrast statistics
- [ ] Edge density analysis
- [ ] Number of images per report

### 8.4 Text-Image Alignment
- [ ] Image count vs report length correlation
- [ ] Concept coverage in reports
- [ ] Multi-view consistency

---

## **TÀI LIỆU THAM KHẢO THÊM**

1. **CheXpert**: https://stanfordmlgroup.github.io/projects/chexpert
2. **IU X-Ray Dataset**: https://openi.nlm.nih.gov/
3. **MIMIC-CXR**: https://physionet.org/content/mimic-cxr/
4. **RadLex**: https://radlex.org/ - Radiology vocabulary
5. **Medical NER**: https://github.com/glample/TagNER

---

## **LƯU Ý QUAN TRỌNG**

1. **Class Imbalance**: Effusion chiếm 77.53% - cần áp dụng class weights hoặc oversampling
2. **Negation Handling**: "No effusion" khác hoàn toàn với "Effusion present" - model cần học điều này
3. **Multi-view**: Mỗi sample có 2 images (frontal + lateral) - cần xử lý multi-image input
4. **Short Reports**: Một số report rất ngắn (7 từ) - có thể ảnh hưởng đến generation quality
5. **Medical Terminology**: Cần tokenizer hiểu medical terms - có thể fine-tune tokenizer

---

*File hướng dẫn này được tạo dựa trên phân tích cấu trúc project A3Net và dataset IU X-Ray của bạn.*

