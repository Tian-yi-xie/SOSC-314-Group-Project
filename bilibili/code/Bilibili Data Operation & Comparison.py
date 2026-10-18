#Preprocessing: tokenize Chinese comments with jieba, remove stopwords, 
#and keep only multi-character Chinese words for downstream vectorization (CountVectorizer / TfidfVectorizer) and classification.

import pandas as pd
import jieba
import re

# Load the labeled data
df = pd.read_csv('/content/bilibili_comments_labeled_clean.csv')
print(f"Loaded: {len(df)} comments")
print(f"Columns: {df.columns.tolist()}")

# Define Chinese stopwords
stopwords = set([
    '的', '了', '是', '在', '我', '你', '他', '她', '它', '这', '那',
    '就', '都', '也', '和', '与', '但', '而', '或', '及', '等',
    '一个', '什么', '怎么', '为什么', '这样', '那样', '这个', '那个',
    '真的', '就是', '还是', '但是', '因为', '所以', '如果', '虽然',
    '啊', '吧', '呢', '吗', '呀', '哦', '嗯', '哈', '哎',
    '可以', '觉得', '知道', '感觉', '可能', '应该', '现在',
])

# Tokenize: keep only Chinese, remove stopwords, keep words with len > 1
def tokenize(text):
    text = str(text)
    text = re.sub(r'[^\u4e00-\u9fff]', ' ', text)
    words = jieba.cut(text)
    return ' '.join([w for w in words if w not in stopwords and len(w) > 1])

df['tokens'] = df['content'].apply(tokenize)

# Show a few examples
print("\nTokenization examples:")
for i in range(5):
    print(f"Original: {df['content'].iloc[i]}")
    print(f"Tokens:   {df['tokens'].iloc[i]}")
    print()

print(f"Total rows with tokens: {len(df)}")

import pandas as pd
import jieba
import re
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Load data
df = pd.read_csv('/content/bilibili_comments_labeled_clean.csv')

# Tokenize if not present
if 'tokens' not in df.columns:
    stopwords = set(['的','了','是','在','我','你','他','她','它','这','那','就','都','也','和'])
    def tokenize(text):
        text = re.sub(r'[^\u4e00-\u9fff]', ' ', str(text))
        return ' '.join([w for w in jieba.cut(text) if w not in stopwords and len(w) > 1])
    df['tokens'] = df['content'].apply(tokenize)

# Vectorize
cv = CountVectorizer(max_features=1000)
X_count = cv.fit_transform(df['tokens'])

tfidf = TfidfVectorizer(max_features=1000)
X_tfidf = tfidf.fit_transform(df['tokens'])

dims = ['genuine_attachment', 'emotional_substitution',
        'playful_performance', 'dangerous_dependence']

# Compute F1 automatically
count_f1 = []
tfidf_f1 = []

for dim in dims:
    y = df[dim].astype(int)
    
    clf_c = LogisticRegression(max_iter=1000, class_weight='balanced')
    f1_c = cross_val_score(clf_c, X_count, y, cv=5, scoring='f1').mean()
    count_f1.append(f1_c)
    
    clf_t = LogisticRegression(max_iter=1000, class_weight='balanced')
    f1_t = cross_val_score(clf_t, X_tfidf, y, cv=5, scoring='f1').mean()
    tfidf_f1.append(f1_t)

print("Count F1:", [f"{x:.3f}" for x in count_f1])
print("TF-IDF F1:", [f"{x:.3f}" for x in tfidf_f1])

# Figure 3: Compare CountVectorizer vs. TfidfVectorizer on the same classification task.
#The classifier (LogisticRegression) and feature count (1000) are held constant.

labels = ['Genuine\nAttachment', 'Emotional\nSubstitution',
          'Playful\nPerformance', 'Dangerous\nDependence']

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
bars1 = ax.bar(x - width/2, count_f1, width, label='Count + LR', color='#4C72B0')
bars2 = ax.bar(x + width/2, tfidf_f1, width, label='TF-IDF + LR', color='#DD8452')

ax.set_ylabel('5-fold F1 Score')
ax.set_title('CountVectorizer vs. TfidfVectorizer: F1 per Dimension')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1.0)
ax.legend()

for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                f'{h:.3f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('/content/fig_representation_comparison.png', dpi=300)
plt.show()
