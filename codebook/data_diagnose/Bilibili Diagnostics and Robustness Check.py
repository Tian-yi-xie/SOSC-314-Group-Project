#Diagnose of two dimensions from comparison results
import pandas as pd
import jieba
import re
from sklearn.feature_extraction.text import CountVectorizer

# Load data
df = pd.read_csv('/content/bilibili_comments_labeled_clean.csv')
print(f"Loaded: {len(df)} comments")
print(f"Columns: {df.columns.tolist()}")

# Tokenize 
if 'tokens' not in df.columns:
    stopwords = set([
        '的', '了', '是', '在', '我', '你', '他', '她', '它', '这', '那',
        '就', '都', '也', '和', '与', '但', '而', '或', '及', '等',
        '一个', '什么', '怎么', '为什么', '这样', '那样', '这个', '那个',
        '真的', '就是', '还是', '但是', '因为', '所以', '如果', '虽然',
        '啊', '吧', '呢', '吗', '呀', '哦', '嗯', '哈', '哎',
        '可以', '觉得', '知道', '感觉', '可能', '应该', '现在',
    ])

    def tokenize(text):
        text = str(text)
        text = re.sub(r'[^\u4e00-\u9fff]', ' ', text)
        words = jieba.cut(text)
        return ' '.join([w for w in words if w not in stopwords and len(w) > 1])

    df['tokens'] = df['content'].apply(tokenize)
    print(f"\nTokens column created.")
    print(df[['content', 'tokens']].head(3))

# Robustness check — what drives Playful's Count advantage?

playful = df[df['playful_performance'] == True]['tokens']
non_playful = df[df['playful_performance'] == False]['tokens']

print(f"\nPlayful comments: {len(playful)}")
print(f"Non-playful comments: {len(non_playful)}")

cv = CountVectorizer(max_features=50)
cv.fit(pd.concat([playful, non_playful]))

playful_counts = cv.transform(playful).sum(axis=0).A1
non_playful_counts = cv.transform(non_playful).sum(axis=0).A1
words = cv.get_feature_names_out()

# Per-comment frequency difference
diff = playful_counts / len(playful) - non_playful_counts / len(non_playful)
top_idx = diff.argsort()[-20:][::-1]

print("\nTop 20 words distinguishing Playful from non-Playful:")
print(f"{'Word':<15} {'Playful':<12} {'Non-Playful':<12}")
print("-" * 40)
for i in top_idx:
print(f"{words[i]:<15} {playful_counts[i]/len(playful):<12.3f} {non_playful_counts[i]/len(non_playful):<12.3f}")

#Robustness check — dangerous dependence
dang = df[df['dangerous_dependence'] == True]['tokens']
non_dang = df[df['dangerous_dependence'] == False]['tokens']

cv = CountVectorizer(max_features=50)
cv.fit(pd.concat([dang, non_dang]))

dang_counts = cv.transform(dang).sum(axis=0).A1
non_dang_counts = cv.transform(non_dang).sum(axis=0).A1
words = cv.get_feature_names_out()

diff = dang_counts / len(dang) - non_dang_counts / len(non_dang)
top_idx = diff.argsort()[-20:][::-1]

print("Top 20 words distinguishing Dangerous from non-Dangerous:")
print(f"{'Word':<15} {'Dangerous':<12} {'Non-Dangerous':<12}")
print("-" * 45)
for i in top_idx:
print(f"{words[i]:<15} {dang_counts[i]/len(dang):<12.3f} {non_dang_counts[i]/len(non_dang):<12.3f}")

#Visualization of Diagnose
import matplotlib.pyplot as plt
import numpy as np
# Data: Playful top words 
playful_words = ['human', 'self', 'none', 'emotion', 'like',
                 'friend', 'when', 'truly', 'need', 'problem']
playful_diffs = [0.540-0.070, 0.640-0.207, 0.500-0.159, 0.400-0.070,
                 0.400-0.074, 0.360-0.035, 0.400-0.079, 0.320-0.021,
                 0.320-0.043, 0.320-0.052]

# Data: Dangerous top words 
dang_words = ['none/have-not', 'self', 'human', 'emotion', 'feeling',
              'not', 'problem', 'when', 'will not', 'chat']
dang_diffs = [1.149-0.000, 0.664-0.153, 0.418-0.039, 0.336-0.044,
              0.269-0.023, 0.306-0.067, 0.254-0.033, 0.269-0.066,
              0.261-0.062, 0.246-0.077]

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Playful
axes[0].barh(playful_words[::-1], playful_diffs[::-1], color='#4C72B0')
axes[0].set_xlabel('Frequency difference (Playful − non-Playful)')
axes[0].set_title('Top words distinguishing Playful comments')
axes[0].axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
for i, v in enumerate(playful_diffs[::-1]):
    axes[0].text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9)

# Right: Dangerous
axes[1].barh(dang_words[::-1], dang_diffs[::-1], color='#C44E52')
axes[1].set_xlabel('Frequency difference (Dangerous − non-Dangerous)')
axes[1].set_title('Top words distinguishing Dangerous comments')
axes[1].axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
for i, v in enumerate(dang_diffs[::-1]):
    axes[1].text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('/content/fig_robustness_top_words.png', dpi=300)
plt.show()

#New Dictionary
import os
import pandas as pd

# 1. Create directory
os.makedirs('dictionaries', exist_ok=True)

# 2. Refined dictionary
# Removed broad words: 没有, 帮助, 意义, 陪伴, 开心, 任何事
# Replaced with specific words that carry clear framing signals

data = {
    'dimension': (
        ['genuine_attachment'] * 5 +
        ['emotional_substitution'] * 5 +
        ['playful_performance'] * 6 +
        ['dangerous_dependence'] * 6
    ),
    'keyword': [
        # Genuine Attachment (5)
        '爱', '心动', '谈恋爱', '真的喜欢', '伴侣',
        # Emotional Substitution (5)
        '替代', '比真人', '情绪价值', '倾诉', '不如AI',
        # Playful Performance (6)
        '整活', '捏人', '乐子', '玩梗', '笑死', '哈哈哈哈',
        # Dangerous Dependence (6)
        '上瘾', '戒不掉', '停不下来', '离不开', '崩溃', '哭死',
    ],
    'type': [
        'verb', 'verb', 'verb', 'phrase', 'noun',
        'verb', 'phrase', 'noun', 'verb', 'phrase',
        'verb', 'verb', 'noun', 'verb', 'verb', 'interj',
        'adj', 'phrase', 'phrase', 'phrase', 'verb', 'verb',
    ],
    'notes': [''] * 22,
}

# 3. Save
df = pd.DataFrame(data)
df.to_csv('dictionaries/seed2.csv', index=False, encoding='utf-8-sig')

print(f"Total keywords: {len(df)}")
print(df.groupby('dimension')['keyword'].apply(list))

#Labeled new dictionary
import pandas as pd
import jieba
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Load data
df = pd.read_csv('/content/bilibili_comments_labeled_clean.csv')
print(f"Loaded: {len(df)} comments")

# Tokenize (if missing)

if 'tokens' not in df.columns:
    stopwords = set([
        '的', '了', '是', '在', '我', '你', '他', '她', '它', '这', '那',
        '就', '都', '也', '和', '与', '但', '而', '或', '及', '等',
        '一个', '什么', '怎么', '为什么', '这样', '那样', '这个', '那个',
        '真的', '就是', '还是', '但是', '因为', '所以', '如果', '虽然',
        '啊', '吧', '呢', '吗', '呀', '哦', '嗯', '哈', '哎',
        '可以', '觉得', '知道', '感觉', '可能', '应该', '现在',
    ])

    def tokenize(text):
        text = str(text)
        text = re.sub(r'[^\u4e00-\u9fff]', ' ', text)
        words = jieba.cut(text)
        return ' '.join([w for w in words if w not in stopwords and len(w) > 1])

    df['tokens'] = df['content'].apply(tokenize)
    print("Tokens column created.")

# Save tokens version for future use
df.to_csv('/content/bilibili_comments_with_tokens.csv', index=False, encoding='utf-8-sig')
print("Saved with tokens.")

#  Re-label with refined dictionary (seed2.csv)
dictionary = pd.read_csv('dictionaries/seed2.csv')
dims = dictionary.groupby('dimension')['keyword'].apply(list).to_dict()

for dim, words in dims.items():
    pattern = '|'.join(words)
    df[dim + '_v2'] = df['content'].str.contains(pattern, na=False)
    print(f"{dim}_v2: {df[dim + '_v2'].sum()} ({df[dim + '_v2'].mean()*100:.1f}%)")

# Multi-dimension distribution
dims_v2 = [d + '_v2' for d in dims.keys()]
df['n_dims_v2'] = df[dims_v2].sum(axis=1)
print(f"\nMulti-dimension distribution:")
print(df['n_dims_v2'].value_counts().sort_index())
# Diagnostic 1 — Top words in refined multi-dimension
multi_v2 = df[df['n_dims_v2'] >= 2]
single_v2 = df[df['n_dims_v2'] == 1]

print(f"\nMulti-dimension comments: {len(multi_v2)}")
print(f"Single-dimension comments: {len(single_v2)}")

cv = CountVectorizer(max_features=30)
cv.fit(pd.concat([multi_v2['tokens'], single_v2['tokens']]))

multi_counts = cv.transform(multi_v2['tokens']).sum(axis=0).A1
single_counts = cv.transform(single_v2['tokens']).sum(axis=0).A1
words = cv.get_feature_names_out()

diff = multi_counts / len(multi_v2) - single_counts / len(single_v2)
top_idx = diff.argsort()[-15:][::-1]

print("\nTop 15 words in refined multi-dimension comments:")
print(f"{'Word':<15} {'Multi':<10} {'Single':<10}")
print("-" * 35)
for i in top_idx:
    print(f"{words[i]:<15} {multi_counts[i]/len(multi_v2):<10.3f} {single_counts[i]/len(single_v2):<10.3f}")

# Diagnostic 2 — F1 with refined dictionary
print("\n=== F1 with refined dictionary (seed2) ===")

cv_count = CountVectorizer(max_features=1000)
X_count = cv_count.fit_transform(df['tokens'])

tfidf = TfidfVectorizer(max_features=1000)
X_tfidf = tfidf.fit_transform(df['tokens'])

for dim in dims_v2:
    y = df[dim].astype(int)
    n_pos = y.sum()

    f1_count = cross_val_score(
        LogisticRegression(max_iter=1000, class_weight='balanced'),
        X_count, y, cv=5, scoring='f1'
    ).mean()

    f1_tfidf = cross_val_score(
        LogisticRegression(max_iter=1000, class_weight='balanced'),
        X_tfidf, y, cv=5, scoring='f1'
    ).mean()

print(f"{dim}: n_pos={n_pos}, Count={f1_count:.3f}, TF-IDF={f1_tfidf:.3f}")

#Visualization of comparison of two dictionaries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Load
df = pd.read_csv('/content/bilibili_comments_with_tokens.csv')
df['tokens'] = df['tokens'].fillna('')

# Load both dictionaries
old_dict = pd.read_csv('/content/dictionaries/seed.csv')
new_dict = pd.read_csv('/content/bilibili_comment_scraper_webui/dictionaries/seed2.csv')

old_dims = old_dict.groupby('dimension')['keyword'].apply(list).to_dict()
new_dims = new_dict.groupby('dimension')['keyword'].apply(list).to_dict()

dim_order = ['genuine_attachment', 'emotional_substitution',
             'playful_performance', 'dangerous_dependence']
# Counts (auto)
old_counts = []
new_counts = []
for dim in dim_order:
    old_counts.append(df['content'].str.contains('|'.join(old_dims[dim]), na=False).sum())
    new_counts.append(df['content'].str.contains('|'.join(new_dims[dim]), na=False).sum())

print("Old counts:", old_counts)
print("New counts:", new_counts)

# F1 (auto)
cv = CountVectorizer(max_features=1000)
X = cv.fit_transform(df['tokens'])

old_f1 = []
new_f1 = []
for dim in dim_order:
    y_old = df['content'].str.contains('|'.join(old_dims[dim]), na=False).astype(int)
    old_f1.append(cross_val_score(
        LogisticRegression(max_iter=1000, class_weight='balanced'),
        X, y_old, cv=5, scoring='f1'
    ).mean())
    
    y_new = df['content'].str.contains('|'.join(new_dims[dim]), na=False).astype(int)
    new_f1.append(cross_val_score(
        LogisticRegression(max_iter=1000, class_weight='balanced'),
        X, y_new, cv=5, scoring='f1'
    ).mean())
    
    print(f"{dim}: old F1={old_f1[-1]:.3f} (n={y_old.sum()}), new F1={new_f1[-1]:.3f} (n={y_new.sum()})")

# Figure 1: Counts
labels = ['Genuine\nAttachment', 'Emotional\nSubstitution',
          'Playful\nPerformance', 'Dangerous\nDependence']
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
bars1 = ax.bar(x - width/2, old_counts, width, label='Original dictionary', color='#4C72B0')
bars2 = ax.bar(x + width/2, new_counts, width, label='Refined dictionary', color='#DD8452')
ax.set_ylabel('Number of positive comments')
ax.set_title('Dictionary Refinement: Positive Class Size Before and After')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 3, str(int(h)), ha='center', fontsize=10)
plt.tight_layout()
plt.savefig('/content/fig_dictionary_refinement.png', dpi=300)
plt.show()

# Figure 2: F1
fig, ax = plt.subplots(figsize=(10, 5))
bars1 = ax.bar(x - width/2, old_f1, width, label='Original dictionary', color='#4C72B0')
bars2 = ax.bar(x + width/2, new_f1, width, label='Refined dictionary', color='#DD8452')
ax.set_ylabel('5-fold F1 Score')
ax.set_title('Dictionary Refinement: F1 Before and After')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1.0)
ax.legend()
for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.3f}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('/content/fig_f1_refinement.png', dpi=300)
plt.show()

#Expand sample amount
import asyncio
import pandas as pd
from bilibili_api import comment, video, Credential

async def fetch_one_video(bvid, max_pages=50):
    """Fetch comments from one video, up to max_pages."""
    v = video.Video(bvid=bvid, credential=credential)
    aid = v.get_aid()
    all_comments = []
    
    for page in range(1, max_pages + 1):
        try:
            data = await comment.get_comments(
                oid=aid,
                type_=comment.CommentResourceType.VIDEO,
                page_index=page,
                order=comment.OrderType.LIKE,
                credential=credential
            )
            replies = data.get('replies') or []
            
            if not replies:
                print(f"  Page {page}: no more comments, stopping")
                break
            
            for r in replies:
                all_comments.append({
                    'content': r['content']['message'],
                    'like': r['like'],
                    'user': r['member']['uname'],
                    'uid': str(r['member']['mid']),
                    'rpid': str(r['rpid']),
                    'ctime': r['ctime'],
                })
            
            print(f"  Page {page}: {len(replies)} comments, total {len(all_comments)}")
            
            # 3-second delay to avoid rate limiting
            await asyncio.sleep(3)
        
        except Exception as e:
            print(f"  Page {page} error: {e}")
            break
    
    return all_comments

async def main():
    bv_list = ['BV18hJdzMEPf', 'BV1yv4y1H7Ev', 'BV1f3896bE7D']
    all_data = []
    
    for bv in bv_list:
        print(f"\n===== {bv} =====")
        try:
            comments = await fetch_one_video(bv, max_pages=50)
            for c in comments:
                c['source_video'] = bv
            all_data.extend(comments)
            print(f"√ {bv}: {len(comments)} comments")
        except Exception as e:
            print(f"× {bv} failed: {e}")
    
    df = pd.DataFrame(all_data)
    df.to_csv('/content/bilibili_comments_expanded.csv', index=False, encoding='utf-8-sig')
    print(f"\n===== DONE =====")
    print(f"Total comments: {len(df)}")
    print(f"Saved to: /content/bilibili_comments_expanded.csv")

await main()

# Clean new sample csv
import pandas as pd
import jieba
import re

df = pd.read_csv('/content/bilibili_comments_expanded.csv')
print(f"Raw expanded: {len(df)}")

# Clean
df = df.drop_duplicates(subset=['content'])
print(f"After dedup: {len(df)}")

df = df[df['content'].str.len() >= 5]
print(f"After length filter: {len(df)}")

df = df[df['content'].str.contains(r'[\u4e00-\u9fff]', na=False)]
print(f"After language filter: {len(df)}")

# Video label
label_map = {
    'BV1yv4y1H7Ev': 'AI伴侣',
    'BV1f3896bE7D': 'AI恋爱',
    'BV18hJdzMEPf': '爱上AI',
}
df['video_label'] = df['source_video'].map(label_map)

# Tokenize
stopwords = set([
    '的', '了', '是', '在', '我', '你', '他', '她', '它', '这', '那',
    '就', '都', '也', '和', '与', '但', '而', '或', '及', '等',
    '一个', '什么', '怎么', '为什么', '这样', '那样', '这个', '那个',
    '真的', '就是', '还是', '但是', '因为', '所以', '如果', '虽然',
    '啊', '吧', '呢', '吗', '呀', '哦', '嗯', '哈', '哎',
    '可以', '觉得', '知道', '感觉', '可能', '应该', '现在',
])

def tokenize(text):
    text = str(text)
    text = re.sub(r'[^\u4e00-\u9fff]', ' ', text)
    words = jieba.cut(text)
    return ' '.join([w for w in words if w not in stopwords and len(w) > 1])

df['tokens'] = df['content'].apply(tokenize)

df.to_csv('/content/bilibili_expanded_clean.csv', index=False, encoding='utf-8-sig')
print(f"\nSaved: {len(df)} comments")
print(df['video_label'].value_counts())

# Labeled new sample csv
import pandas as pd

# Load expanded clean data
df = pd.read_csv('/content/bilibili_expanded_clean.csv')
print(f"Loaded: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# Apply refined dictionary
new_dict = pd.read_csv('/content/bilibili_comment_scraper_webui/dictionaries/seed2.csv')
new_dims = new_dict.groupby('dimension')['keyword'].apply(list).to_dict()

# Label
for dim, words in new_dims.items():
    df[dim] = df['content'].str.contains('|'.join(words), na=False)
    print(f"{dim}: {df[dim].sum()} ({df[dim].mean()*100:.1f}%)")

# Save
df.to_csv('/content/bilibili_expanded_labeled.csv', index=False, encoding='utf-8-sig')
print(f"\nSaved: {len(df)}")

#Compare two verisons of sample with new dictionary
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Load refined dictionary
new_dict = pd.read_csv('/content/bilibili_comment_scraper_webui/dictionaries/seed2.csv')
new_dims = new_dict.groupby('dimension')['keyword'].apply(list).to_dict()

dim_order = ['genuine_attachment', 'emotional_substitution',
             'playful_performance', 'dangerous_dependence']

# Original sample (860)
df_old = pd.read_csv('/content/bilibili_comments_with_tokens.csv')
df_old['tokens'] = df_old['tokens'].fillna('')

old_n = []
old_f1 = []

cv_old = CountVectorizer(max_features=1000)
X_old_count = cv_old.fit_transform(df_old['tokens'])
tfidf_old = TfidfVectorizer(max_features=1000)
X_old_tfidf = tfidf_old.fit_transform(df_old['tokens'])

for dim in dim_order:
    y = df_old['content'].str.contains('|'.join(new_dims[dim]), na=False).astype(int)
    old_n.append(y.sum())
    
    f1_c = cross_val_score(
        LogisticRegression(max_iter=1000, class_weight='balanced'),
        X_old_count, y, cv=5, scoring='f1'
    ).mean()
    f1_t = cross_val_score(
        LogisticRegression(max_iter=1000, class_weight='balanced'),
        X_old_tfidf, y, cv=5, scoring='f1'
    ).mean()
    old_f1.append(max(f1_c, f1_t))

print("Original sample:")
print("  n =", old_n)
print("  F1 =", [f"{x:.3f}" for x in old_f1])

# Expanded sample (2302)
df_new = pd.read_csv('/content/bilibili_expanded_labeled.csv')
df_new['tokens'] = df_new['tokens'].fillna('')

new_n = []
new_f1 = []

cv_new = CountVectorizer(max_features=1000)
X_new_count = cv_new.fit_transform(df_new['tokens'])
tfidf_new = TfidfVectorizer(max_features=1000)
X_new_tfidf = tfidf_new.fit_transform(df_new['tokens'])

for dim in dim_order:
    y = df_new[dim].astype(int)
    new_n.append(y.sum())
    
    f1_c = cross_val_score(
        LogisticRegression(max_iter=1000, class_weight='balanced'),
        X_new_count, y, cv=5, scoring='f1'
    ).mean()
    f1_t = cross_val_score(
        LogisticRegression(max_iter=1000, class_weight='balanced'),
        X_new_tfidf, y, cv=5, scoring='f1'
    ).mean()
    new_f1.append(max(f1_c, f1_t))

print("\nExpanded sample:")
print("  n =", new_n)
print("  F1 =", [f"{x:.3f}" for x in new_f1])

# Figure
labels = ['Genuine\nAttachment', 'Emotional\nSubstitution',
          'Playful\nPerformance', 'Dangerous\nDependence']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
x = np.arange(len(labels))
width = 0.35

# Left: positive class size
axes[0].bar(x - width/2, old_n, width, label='Original sample (n=860)', color='#4C72B0')
axes[0].bar(x + width/2, new_n, width, label='Expanded sample (n=2302)', color='#DD8452')
axes[0].set_ylabel('Number of positive comments')
axes[0].set_title('Positive class size: original vs expanded')
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels)
axes[0].legend()
for i, (a, b) in enumerate(zip(old_n, new_n)):
    axes[0].text(i - width/2, a + 5, str(a), ha='center', fontsize=9)
    axes[0].text(i + width/2, b + 5, str(b), ha='center', fontsize=9)

# Right: F1
axes[1].bar(x - width/2, old_f1, width, label='Original sample (n=860)', color='#4C72B0')
axes[1].bar(x + width/2, new_f1, width, label='Expanded sample (n=2302)', color='#DD8452')
axes[1].set_ylabel('5-fold F1 (refined dictionary)')
axes[1].set_title('F1 after refinement: original vs expanded')
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels)
axes[1].set_ylim(0, 1.0)
axes[1].legend()
for i, (a, b) in enumerate(zip(old_f1, new_f1)):
    axes[1].text(i - width/2, a + 0.01, f'{a:.2f}', ha='center', fontsize=9)
    axes[1].text(i + width/2, b + 0.01, f'{b:.2f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('/content/fig_expansion_diagnostic.png', dpi=300)
plt.show()


# Text similarity clustering
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv('/content/bilibili_expanded_labeled.csv')
df['tokens'] = df['tokens'].fillna('')
print(f"Loaded: {len(df)}")

# Vectorize
tfidf = TfidfVectorizer(max_features=1000)
X = tfidf.fit_transform(df['tokens'])
print(f"TF-IDF: {X.shape}")

# K-Means K=4
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X)

print(f"\nCluster distribution:")
print(df['cluster'].value_counts().sort_index())

sil = silhouette_score(X, df['cluster'])
print(f"Silhouette: {sil:.3f}")

# Top 15 words per cluster
feature_names = tfidf.get_feature_names_out()
for c in range(4):
    center = kmeans.cluster_centers_[c]
    top_idx = center.argsort()[-15:][::-1]
    top_words = [feature_names[i] for i in top_idx]
    print(f"\nCluster {c} (n={sum(df['cluster']==c)}):")
    print(f"  {', '.join(top_words)}")

# Save for later
df.to_csv('/content/bilibili_expanded_clustered.csv', index=False, encoding='utf-8-sig')
print("\nSaved to bilibili_expanded_clustered.csv")

#Identify K-means pattern
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Compute Silhouette for each K
k_values = [3, 4, 5, 6, 7, 8, 10]
sil_scores = []

for k in k_values:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels)
    sil_scores.append(sil)
    print(f"K={k}: Silhouette={sil:.3f}")

# Plot
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(k_values, sil_scores, marker='o', color='#4C72B0', linewidth=2, markersize=8)
ax.set_xlabel('Number of clusters (K)')
ax.set_ylabel('Silhouette score')
ax.set_title('Silhouette score across different K values')
ax.set_xticks(k_values)
ax.set_ylim(0, 0.10)  
ax.axhline(y=0.10, color='gray', linestyle='--', linewidth=0.7, label='Weak structure threshold (0.10)')
ax.legend()

# Annotate each point
for k, s in zip(k_values, sil_scores):
    ax.text(k, s + 0.003, f'{s:.3f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('/content/fig_silhouette_by_k.png', dpi=300)
plt.show()
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score

# Load the labeled expanded sample
df = pd.read_csv('/content/bilibili_expanded_labeled.csv')
df['tokens'] = df['tokens'].fillna('')
print(f"Loaded: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# Vectorize: create the numeric representation used by the classifiers
cv = CountVectorizer(max_features=1000)
X_count = cv.fit_transform(df['tokens'])
print(f"X_count shape: {X_count.shape}")

tfidf = TfidfVectorizer(max_features=1000)
X_tfidf = tfidf.fit_transform(df['tokens'])
print(f"X_tfidf shape: {X_tfidf.shape}")

dims = ['genuine_attachment', 'emotional_substitution',
        'playful_performance', 'dangerous_dependence']

# Random seed stability: check whether F1 varies substantially across seeds
print("\n=== Random seed stability (all 4 dimensions, Count) ===")
for dim in dims:
    print(f"\n{dim}:")
    y = df[dim].astype(int)
    f1_scores = []
    for seed in [42, 0, 1, 7, 123]:
        kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
        clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=seed)
        f1 = cross_val_score(clf, X_count, y, cv=kfold, scoring='f1').mean()
        f1_scores.append(f1)
        print(f"  seed={seed}: F1={f1:.4f}")
    print(f"  → mean={sum(f1_scores)/len(f1_scores):.4f}, range={max(f1_scores)-min(f1_scores):.4f}")

# Classifier stability: check whether F1 depends on the choice of classifier
print("\n=== Classifier stability ===")
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    'SVM': SVC(class_weight='balanced', random_state=42),
    'RandomForest': RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100),
}

for name, model in models.items():
    print(f"\n{name}:")
    for dim in dims:
        y = df[dim].astype(int)
        f1 = cross_val_score(model, X_count, y, cv=kfold, scoring='f1').mean()
        print(f"  {dim}: F1={f1:.3f}")



