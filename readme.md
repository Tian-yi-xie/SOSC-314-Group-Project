# Cyber Lovers in China

How Chinese social media users talk about their relationships with AI companions on Baidu Tieba and Bilibili.

**Course:** SOSC-314 - Computational Social Science
**Term:** Fall, 2026

| Team member | Contribution |
|---|---|
| My Thuan Dang | Baidu Tieba: Data collection and preprocessing; frame analysis; semantic clustering and theme development; analysis of thematic and linguistic trends over time in policy context; repository maintenance |
| Tianyi Xie | Data collection and preprocessing; dictionary development and refinement; classification evaluation; clustering and lexical robustness checks; qualitative comment analysis; final report website development |
| Yuwen Zhou | Literature review and theoretical framework |

---

## Overview

Chinese news coverage of AI companions, and the rules issued on 10 April 2026, mostly treat these apps as emotionally harmful: users become dependent, replace human relationships, or suffer when an app changes. From this coverage and the literature we built four relational frames:

| Frame | Example seed words |
|---|---|
| Genuine attachment | 爱, 心动, 谈恋爱, 真的喜欢 |
| Emotional substitution | 替代, 比真人, 情绪价值, 倾诉 |
| Playful performance | 整活, 捏人, 玩梗, 笑死 |
| Dangerous dependence | 上瘾, 戒不掉, 离不开, 崩溃 |

---

## Research Question

1. How much of the online conversation about AI companions fits these four frames?
2. What do users actually talk about when they discuss AI companions?
3. How did the conversation change after the rules were issued?
4. Can these frames be measured from words alone, and does that depend on the platform?

The project started as a plan to code posts on the four frames. Tests on both platforms showed that the frames, measured through keywords, cover only a small part of the conversation. We therefore mapped the discourse bottom-up on Tieba and examined why lexical measurement fails on Bilibili. Early pilot work is kept in `archive/`.

---

## Data

| | Baidu Tieba | Bilibili |
|---|---|---|
| Source | 21 search keywords and 9 forums (5 companion, 4 general AI) | 3 videos about AI companions |
| Unit of analysis | Post | Comment |
| Final corpus | 48,127 posts from 1,974 threads | 2,302 comments |

---

## Methods

Both platforms start from the same seed dictionary of words for each frame (dictionaries/seed.csv)
- **Tieba:** a quick check of how many posts match each frame showed that the frames cover little of the conversation, so the main analysis is bottom-up (exploratory approach): BGE sentence embeddings are reduced with UMAP and clustered using HDBSCAN; the resulting clusters are grouped into nine broader themes through hierarchical clustering of cluster centroids. Fightin’ Words is then used to identify distinctive vocabulary before and after the policy rules.
- **Bilibili:** dictionary labels used to train logistic regression classifiers (5-fold F1); dictionary refinement and corpus expansion; clustering diagnostics with TF-IDF and K-Means.

Details and results are in the final report.

---

## Repository structure

```
├── README.md
├── index.html                              final report, served as the project website by GitHub Pages
├── final_project_report_submission.ipynb   notebook the report is built from
├── requirements.txt                        Python packages needed to run the code
├── docs/                                   Project documentation, AI-use record
├── dictionaries/                           frame seed words
├── tieba/
│   ├── code/                               
│   ├── data/                               
│   └── figures/
├── bilibili/
│   ├── code/
│   ├── data/
│   └── figures/
└── archive/
    ├── tieba_pilot/                        early 15-thread pilot: dictionary coding, LDA, first clustering
    └── frame_classifier_draft/             supervised classifier draft, not used in the final analysis
```
---

### Ethical Handling

- Only publicly accessible content is collected.
- Usernames and user IDs are hashed before storage.
- No personally identifiable information is included in the repository.

---

## Limitations

- Tieba and Bilibili users are not representative of all Chinese internet users, and our choice of keywords and forums shapes which posts were collected.
- The post-policy period is short, limiting analysis of longer-term effects. Other events and platform changes may also shape discussions, so we cannot attribute before-and-after differences to the rules alone.
- Clustering results and theme labels depend on model settings and researcher interpretation; the nine-theme structure is an analytical choice, not an objective ground truth.
- We can only measure how relationships with AI are talked about, not users' psychological states.

---

## Generative AI disclosure

[To be added - key literature on AI companionship, framing analysis, and Chinese social media research]


