"""
Figure 1 presented the distribution of the four framing dimensions:
Playful Performance was the most frequent (26.0%), followed by
Genuine Attachment (15.0%), Dangerous Dependence (5.8%), and
Emotional Substitution (4.8%). Together, the four dimensions covered
only 51.6% of the corpus, which raised a follow-up question: where
did the remaining comments go?
"""
import pandas as pd
import matplotlib.pyplot as plt

# Use the clean labeled file
df = pd.read_csv('/content/bilibili_comments_labeled_clean.csv')
print(f"Total comments: {len(df)}")

dims = ['genuine_attachment', 'emotional_substitution',
        'playful_performance', 'dangerous_dependence']

df['n_dims'] = df[dims].sum(axis=1)

combo_counts = df['n_dims'].value_counts().sort_index()
print("\nDistribution of matched dimensions:")
print(combo_counts)

# Figure 2
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(combo_counts.index.astype(str), combo_counts.values, color='#4C72B0')
ax.set_xlabel('Number of dimensions matched')
ax.set_ylabel('Number of comments')
ax.set_title(f'How Many Dimensions Each Comment Matches (n={len(df)})')

for bar, count in zip(bars, combo_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            str(count), ha='center', fontsize=11)

plt.tight_layout()
plt.savefig('/content/fig2_dimension_count.png', dpi=300)
plt.show()

*Overview dimension data*
import pandas as pd

df = pd.read_csv('/content/bilibili_comments_labeled_clean.csv')
dims = ['genuine_attachment', 'emotional_substitution',
        'playful_performance', 'dangerous_dependence']
df['n_dims'] = df[dims].sum(axis=1)

# 0 dimensions
print("=" * 60)
print(f"0 DIMENSIONS (n={(df['n_dims']==0).sum()}): 20 random comments")
print("=" * 60)
for c in df[df['n_dims'] == 0]['content'].sample(20, random_state=42).tolist():
    print(f"- {c}")

# 1 dimension
print("\n" + "=" * 60)
print(f"1 DIMENSION (n={(df['n_dims']==1).sum()}): 20 random comments")
print("=" * 60)
for c in df[df['n_dims'] == 1]['content'].sample(20, random_state=42).tolist():
    print(f"- {c}")

# 2 dimensions
print("\n" + "=" * 60)
print(f"2 DIMENSIONS (n={(df['n_dims']==2).sum()}): 15 random comments")
print("=" * 60)
for c in df[df['n_dims'] == 2]['content'].sample(15, random_state=42).tolist():
    print(f"- {c}")

*Check 0 dimesion data*
import pandas as pd

df = pd.read_csv('/content/bilibili_comments_labeled_clean.csv')
dims = ['genuine_attachment', 'emotional_substitution',
        'playful_performance', 'dangerous_dependence']
df['n_dims'] = df[dims].sum(axis=1)

# Look at 0-dimension comments
zero_dim = df[df['n_dims'] == 0]

# Search for external critique markers
critique_words = ['精神病', '有病', '离谱', '可笑', '嘲讽', '幼稚', '沉迷']
critique_hits = zero_dim[zero_dim['content'].str.contains('|'.join(critique_words), na=False)]
print(f"External critique candidates: {len(critique_hits)}")
for c in critique_hits['content'].head(10).tolist():
    print(f"- {c}")

# Search for philosophical reflection markers
philosophy_words = ['本质', '存在', '意义', '哲学', '理性', '感性', '关系']
philosophy_hits = zero_dim[zero_dim['content'].str.contains('|'.join(philosophy_words), na=False)]
print(f"\nPhilosophical reflection candidates: {len(philosophy_hits)}")
for c in philosophy_hits['content'].head(10).tolist():
    print(f"- {c}")

*Look at missing data*
import pandas as pd

df = pd.read_csv('/content/bilibili_comments_labeled_clean.csv')
dims = ['genuine_attachment', 'emotional_substitution',
        'playful_performance', 'dangerous_dependence']
df['n_dims'] = df[dims].sum(axis=1)

zero_dim = df[df['n_dims'] == 0]

# Look for affective vocabulary that may be missing from the dictionary
affect_words = ['喜欢', '心安', '独一无二', '温柔', '陪着', '懂我', 
                '理解', '包容', '真诚', '舍不得', '想念']
affect_hits = zero_dim[zero_dim['content'].str.contains('|'.join(affect_words), na=False)]

print(f"0-dimension comments with affective vocabulary: {len(affect_hits)}")
for c in affect_hits['content'].head(15).tolist():
    print(f"- {c}")
