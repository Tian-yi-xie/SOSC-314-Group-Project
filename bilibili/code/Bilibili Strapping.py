# Colone the repo
!git clone https://github.com/ManiaAmaeOvo/bilibili_comment_scraper_webui
%cd bilibili_comment_scraper_webui

# Install
!pip install -r requirements.txt

# Install the browser driver for Playwright
!playwright install chromium
!playwright install-deps chromium

# Start the FastAPI web server in the background
!nohup python run.py > server.log 2>&1 &

# print the startup log
!cat server.log

#Install the bilibili-api library and aiohttp for async HTTP requests
!pip install bilibili-api-python aiohttp

import asyncio
import pandas as pd
from bilibili_api import comment, video, Credential

#  Fetch comments for one video
async def fetch_one_video(bvid, max_pages=15):
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
                print(f"  Page {page}: no more comments")
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
            await asyncio.sleep(2)
        except Exception as e:
            print(f"  Page {page} error: {e}")
            break
    return all_comments

# Strap the comments
async def main():
    bv_list = ['BV18hJdzMEPf', 'BV1yv4y1H7Ev', 'BV1f3896bE7D']
    all_data = []
    for bv in bv_list:
        print(f"\n===== {bv} =====")
        try:
            comments = await fetch_one_video(bv, max_pages=15)
            for c in comments:
                c['source_video'] = bv
            all_data.extend(comments)
            print(f"✅ {bv}: {len(comments)} comments")
        except Exception as e:
            print(f"❌ {bv} failed: {e}")

    df = pd.DataFrame(all_data)
    # Use ABSOLUTE path so it's easy to find
    df.to_csv('/content/bilibili_comments.csv', index=False, encoding='utf-8-sig')
    print(f"\n===== DONE =====")
    print(f"Total comments: {len(df)}")
    print(f"Saved to: /content/bilibili_comments.csv")

await main()

import pandas as pd

#  Load the combined comments CSV
df = pd.read_csv('/content/bilibili_comments.csv')

# Map each BV number to a human-readable label
label_map = {
    'BV1yv4y1H7Ev': 'AI伴侣',
    'BV1f3896bE7D': 'AI恋爱',
    'BV18hJdzMEPf': '爱上AI',
}

#  Add a new column with the readable label
df['video_label'] = df['source_video'].map(label_map)

# Check how many comments per label
print("Comments per video label:")
print(df['video_label'].value_counts())
print()

# Save the relabeled CSV
df.to_csv('/content/bilibili_comments_labeled.csv', index=False, encoding='utf-8-sig')
print("Saved to /content/bilibili_comments_labeled.csv")
print(f"Total rows: {len(df)}")

# Preview
print()
print(df[['video_label', 'content']].head(10))

import os
import pandas as pd

# 1. Create directory
os.makedirs('dictionaries', exist_ok=True)

# 2. Dictionary
# NOTE: keyword has 23 items, so every other list must also have 23 items.

data = {
    'dimension': (
        ['genuine_attachment'] * 5 +
        ['emotional_substitution'] * 6 +
        ['playful_performance'] * 6 +
        ['dangerous_dependence'] * 6
    ),
    'keyword': [
        # Genuine Attachment (5)
        '爱', '心动', '谈恋爱', '真的喜欢', '伴侣',
        # Emotional Substitution (6)
        '替代', '比真人', '任何事', '情绪价值', '倾诉', '不如AI',
        # Playful Performance (6)
        '帮助', '奇妙', '浪漫', '哈哈哈', '意义', '开心',
        # Dangerous Dependence (6)
        '眼泪', '陪伴', '没有', '影响生活', '痛苦', '崩溃',
    ],
    'type': [
        # 23 items total
        'verb', 'verb', 'adj', 'phrase', 'noun',
        'verb', 'phrase', 'phrase', 'noun', 'phrase', 'phrase',
        'adj', 'verb', 'verb', 'interj', 'phrase', 'noun',
        'adj', 'noun', 'phrase', 'phrase', 'phrase', 'verb',
    ],
    'notes': [''] * 23,
}

# 3. Save
df = pd.DataFrame(data)
df.to_csv('dictionaries/seed.csv', index=False, encoding='utf-8-sig')

import pandas as pd
import matplotlib.pyplot as plt

#  Load both CSV files using their actual paths
df = pd.read_csv('/content/bilibili_comments.csv')
dictionary = pd.read_csv('/content/dictionaries/seed.csv')

# Print basic info about both datasets
print(f"Number of comments: {len(df)}")
print(f"Number of dictionary keywords: {len(dictionary)}")
print(f"Dictionary columns: {dictionary.columns.tolist()}")
print(f"Comment columns: {df.columns.tolist()}")
print()

# Inspect the dictionary structure
print("Dictionary contents:")
print(dictionary.groupby('dimension')['keyword'].apply(list))
print()

#  Sample 20 real comments to see their style
print("20 random comments:")
for i, c in enumerate(df['content'].sample(20, random_state=42).tolist(), 1):
    print(f"{i}. {c}")

mport pandas as pd

#  Load the combined comments CSV
df = pd.read_csv('/content/bilibili_comments.csv')

# Map each BV number to a human-readable label
label_map = {
    'BV1yv4y1H7Ev': 'AI伴侣',
    'BV1f3896bE7D': 'AI恋爱',
    'BV18hJdzMEPf': '爱上AI',
}

#  Add a new column with the readable label
df['video_label'] = df['source_video'].map(label_map)

# Check how many comments per label
print("Comments per video label:")
print(df['video_label'].value_counts())
print()

# Save the relabeled CSV
df.to_csv('/content/bilibili_comments_labeled.csv', index=False, encoding='utf-8-sig')
print("Saved to /content/bilibili_comments_labeled.csv")
print(f"Total rows: {len(df)}")

# Preview
print()
print(df[['video_label', 'content']].head(10))

import pandas as pd
import matplotlib.pyplot as plt

# Load the raw comments scraped from Bilibili
df = pd.read_csv('/content/bilibili_comments.csv')
print(f"Raw comments: {len(df)}")

# Remove duplicate comments to avoid over-representing viral phrases
df = df.drop_duplicates(subset=['content'])
print(f"After deduplication: {len(df)}")

# Remove comments shorter than 5 characters, which typically carry no analysable framing content
df = df[df['content'].str.len() >= 5]
print(f"After length filter: {len(df)}")

# Keep only comments containing Chinese characters, removing pure-English or emoji-only comments
df = df[df['content'].str.contains(r'[\u4e00-\u9fff]', na=False)]
print(f"After language filter: {len(df)}")

# Load the seed dictionary that maps each framing dimension to its keywords
dictionary = pd.read_csv('/content/dictionaries/seed.csv')

# Group keywords by dimension to get a dict like {'genuine_attachment': ['爱', '心动', ...], ...}
dims = dictionary.groupby('dimension')['keyword'].apply(list).to_dict()

# Helper function: return the list of keywords from `words` that actually appear in the comment text
def find_matched_keywords(text, words):
    return [w for w in words if w in str(text)]

# For each dimension, create two columns:
#   (1) a boolean column indicating whether the comment matches this dimension
#   (2) a string column listing the exact keywords that triggered the match
for dim, words in dims.items():
    df[dim] = df['content'].apply(lambda x: bool(find_matched_keywords(x, words)))
    df[dim + '_keywords'] = df['content'].apply(
        lambda x: ','.join(find_matched_keywords(x, words))
    )# Print summary statistics per dimension
for dim in dims:
    count = df[dim].sum()
    pct = count / len(df) * 100
    print(f"{dim}: {count} ({pct:.1f}%)")

print(f"\nTotal comments: {len(df)}")

# Count how many comments match at least one dimension
matched_any = df[list(dims.keys())].any(axis=1).sum()
print(f"Comments matching at least one dimension: {matched_any}")

# Save the cleaned AND labeled corpus to a new CSV
df.to_csv('/content/bilibili_comments_labeled_clean.csv', index=False, encoding='utf-8-sig')


# Preview: show comment text and the matched keywords for each dimension
preview_cols = ['content'] + [d + '_keywords' for d in dims]
print("\nPreview of labeled comments:")
print(df[preview_cols].head(10))

# Figure 1: bar chart showing the distribution of the four framing dimensions
counts = [df[d].sum() for d in dims]
pcts = [c / len(df) * 100 for c in counts]
labels = ['Genuine\nAttachment', 'Emotional\nSubstitution',
          'Playful\nPerformance', 'Dangerous\nDependence']

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(labels, pcts, color=['#4C72B0', '#DD8452', '#55A868', '#C44E52'])
ax.set_ylabel('Percentage of comments (%)')
ax.set_title(f'Framing Dimensions in Bilibili Comments (n={len(df)})')
ax.set_ylim(0, max(pcts) * 1.3)

for bar, pct in zip(bars, pcts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{pct:.1f}%', ha='center', fontsize=11)

plt.tight_layout()
plt.savefig('/content/fig1_dimension_distribution.png', dpi=300)
plt.show()
