<p align="center">
  <img src="assets/banner.png" alt="Embeat Banner" width="100%">
</p>

<p align="center">
  <b>English</b> | <a href="README_zh.md">简体中文</a>
</p>

<p align="center">
  <a href="https://github.com/gdstudio-org/Embeat"><img src="https://img.shields.io/github/stars/gdstudio-org/Embeat?style=social" alt="Stars"></a>
  <a href="https://github.com/gdstudio-org/Embeat/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-CC--BY--NC%204.0-blue" alt="License"></a>
</p>

---

# Embeat: A Music Recommendation System Based on Acoustic Features

## Introduction

Embeat is a music recommendation system built on Spotify acoustic feature data. It encodes audio features into vectors via a **contrastive learning model** and combines them with a **multi-channel recall** strategy to deliver high-quality music recommendations.

Key Features:

- **Acoustic Similarity**: The EmbeatMLP model, trained on Spotify Audio Features (key, tempo, energy, valence, etc.), encodes acoustic features into 64-dim vectors
- **Genre Awareness**: Leverages 6,000+ micro-genre tags to precisely assign genres to 2M+ artists, preventing "acoustically similar but stylistically different" recommendations
- **Multi-Channel Recall**: 5 parallel recall channels (Acoustic Similarity / Same-Genre Popular / Same Artist / Similar Artists / Playlist Collaborative Filtering), merged and scored for final output
- **Playlist Collaborative Filtering**: Track2Vec (Word2Vec-inspired) learns track co-occurrence patterns from 1.88M Spotify playlists
- **Millisecond-Level Response**: Powered by the Qdrant vector database, retrieval across 45M tracks completes in 30–100ms


## Roadmap

> If you find this project helpful, please give it a ⭐️. It means a lot to a personal project, thanks!

- [x] **2026-06-26**: Open-source initial codebase + [EmbeatMLP model weights](checkpoints/EmbeatMLP/)
- [x] **2026-06-26**: Open-source [45M tracks dataset](https://huggingface.co/datasets/GD-Studio/embeat_45m_spotify_tracks) + [technical documentation](https://www.bilibili.com/opus/1218087093501165591)
- [ ] **100 Stars**: Open-source Qdrant database
- [ ] **1K Stars**: Open-source Track2Vec model weights + 1.8M playlists dataset


## Demo

> Below are example recommendation results from Embeat (please unmute before playing)

<details open>
<summary><b>Uptown Funk - Bruno Mars [dance pop, pop]</b></summary>
<table>
<tr>
<th width="25%">Seed Track</th>
<th width="25%">Embeat #1</th>
<th width="25%">Embeat #2</th>
<th width="25%">Embeat #3</th>
</tr>
<tr>
<td>Uptown Funk - Bruno Mars</td>
<td>CAN'T STOP THE FEELING! - Justin Timberlake</td>
<td>Happy - Pharrell Williams</td>
<td>I Like to Move It - will.i.am</td>
</tr>
<tr>
<td>
<video src="https://github.com/user-attachments/assets/e22f726e-eb61-49c1-ab17-91cda503293d" controls width="100%"></video>
</td>
<td>
<video src="https://github.com/user-attachments/assets/a571d075-1527-4d57-8ca7-a2f285e04de6" controls width="100%"></video>
</td>
<td>
<video src="https://github.com/user-attachments/assets/8ba236f0-f28e-4083-8561-00fcbfb49a98" controls width="100%"></video>
</td>
<td>
<video src="https://github.com/user-attachments/assets/1758195d-34c0-485c-b627-5b6d26e00b17" controls width="100%"></video>
</td>
</tr>
</table>
</details>

<details>
<summary><b>杀死那个石家庄人 - 万能青年旅店 [chinese indie rock]</b></summary>
<table>
<tr>
<th width="25%">Seed Track</th>
<th width="25%">Embeat #1</th>
<th width="25%">Embeat #2</th>
<th width="25%">Embeat #3</th>
</tr>
<tr>
<td>杀死那个石家庄人 - 万能青年旅店</td>
<td>大石碎胸口 - 万能青年旅店</td>
<td>凄美地 - 郭顶</td>
<td>不要停止我的音乐 - 痛仰乐队</td>
</tr>
<tr>
<td>
<video src="https://github.com/user-attachments/assets/48eb51b0-0796-42e1-a54f-8a46a49d2916" controls width="100%"></video>
</td>
<td>
<video src="https://github.com/user-attachments/assets/e98e555f-8ac9-4fd0-9254-f440a6e0008e" controls width="100%"></video>
</td>
<td>
<video src="https://github.com/user-attachments/assets/535fec8c-3da1-4ba0-8bfa-bb0889cd698d" controls width="100%"></video>
</td>
<td>
<video src="https://github.com/user-attachments/assets/857666db-45f8-47df-86d2-32e51a679855" controls width="100%"></video>
</td>
</tr>
</table>
</details>

<details>
<summary><b>Sis puella magica! - 梶浦由記 [anime score, japanese vgm]</b></summary>
<table>
<tr>
<th width="25%">Seed Track</th>
<th width="25%">Embeat #1</th>
<th width="25%">Embeat #2</th>
<th width="25%">Embeat #3</th>
</tr>
<tr>
<td>Sis puella magica! - 梶浦由記</td>
<td>Decretum - 梶浦由記</td>
<td>Zoltraak - Evan Call</td>
<td>Arrietty's Song - Cécile Corbel</td>
</tr>
<tr>
<td>
<video src="https://github.com/user-attachments/assets/caa287b2-d477-443b-84e6-2135c8b2b4be" controls width="100%"></video>
</td>
<td>
<video src="https://github.com/user-attachments/assets/f8caff21-35e3-4620-958e-a2ea1ed4eec5" controls width="100%"></video>
</td>
<td>
<video src="https://github.com/user-attachments/assets/bc3048e6-279d-4010-bc7d-81b705dc2b2c" controls width="100%"></video>
</td>
<td>
<video src="https://github.com/user-attachments/assets/7b197556-9ea6-4c1b-a04b-6221975a9273" controls width="100%"></video>
</td>
</tr>
</table>
</details>

<details>
<summary><b>Gizeh - Oskar Schuster [compositional ambient]</b></summary>
<table>
<tr>
<th width="25%">Seed Track</th>
<th width="25%">Embeat #1</th>
<th width="25%">Embeat #2</th>
<th width="25%">Embeat #3</th>
</tr>
<tr>
<td>Gizeh - Oskar Schuster</td>
<td>Vleurgat - Oskar Schuster</td>
<td>Sleeping Lotus - Joep Beving</td>
<td>Travelling - James Spiteri</td>
</tr>
<tr>
<td>
<video src="https://github.com/user-attachments/assets/f10fad24-0f7b-43e3-ad11-879a7961a86a" controls width="100%"></video>
</td>
<td>
<video src="https://github.com/user-attachments/assets/1ce856b6-0e40-4d42-98a1-af458480e89f" controls width="100%"></video>
</td>
<td>
<video src="https://github.com/user-attachments/assets/b1413a46-243d-480c-a3d7-890789ff34f2" controls width="100%"></video>
</td>
<td>
<video src="https://github.com/user-attachments/assets/992e7662-2fe3-445b-8d14-6cb6417bbed5" controls width="100%"></video>
</td>
</tr>
</table>
</details>

### LLM Blind Evaluation

Using the LLM-as-a-Judge method (GPT-5.5 / Gemini Flash 3.5 / Claude Sonnet 4.6), Embeat was blindly evaluated against Netease Cloud Music in AB tests:

| Judge Model | Embeat Wins | Netease Wins | Tie |
|-------------|:-----------:|:------------:|:---:|
| Claude Sonnet 4.6 | **8** | 2 | 0 |
| Gemini Flash 3.5 | **9** | 1 | 0 |
| GPT 5.5 | **6** | 4 | 0 |

**Conclusions:**

- Embeat's core strength lies in its balance between style precision and artist diversity, with a particularly notable advantage in niche-style scenarios that span across languages and cultures
- Netease Cloud Music retains some reference value only in its deep mining of Mandarin-language local content
- For detailed comparison, please refer to the [technical documentation](https://www.bilibili.com/opus/1218087093501165591)


## System Architecture

### Model Details

**EmbeatMLP** - Acoustic Feature Encoding Model

- Input: 64-dim discrete features (key, mode, tempo, time_signature) + 64-dim continuous features (energy, valence, danceability, etc., 7 dimensions)
- Architecture: Dual-tower MLP (Discrete Tower + Acoustic Tower -> Backbone)
- Output: 64-dim L2-normalized vectors
- Training: Masked InfoNCE Loss, batch_size=4096, converges in ~70 steps
- Extremely small parameter count, supports real-time CPU-only inference

**Track2Vec** - Playlist Collaborative Filtering Model

- Based on Word2Vec Skip-Gram, treating playlists as "sentences" and tracks as "words"
- Training data: 1.88M Spotify playlists
- Vocabulary: 1.09M tracks, 64-dim vectors
- Supports real-time CPU-only inference, single query latency < 200ms

### Multi-Channel Recall

```
Input seed track: track_id / track_name + artist_name
  │
  ├─ Channel 1: Acoustic Similarity Recall (genre filtering + EmbeatMLP cosine similarity)
  ├─ Channel 2: Same-Genre Popular Recall (genre filtering + popularity ranking)
  ├─ Channel 3: Same Artist Recall (same artist + EmbeatMLP cosine similarity)
  ├─ Channel 4: Similar Artists Recall (similar artists + EmbeatMLP cosine similarity)
  ├─ Channel 5: Playlist Collaborative Filtering (Track2Vec cosine similarity)
  │
  ├─ ISRC Deduplication / Re-ranking / Same-Artist Ratio Control
  │
  └─ Output: Top-K Recommendation List
```

### Project Structure

```
Embeat/
├── assets/                 # Static assets folder
├── checkpoints/            # Model weights folder
│   ├── EmbeatMLP/          # EmbeatMLP model weights
│   └── Track2Vec/          # Track2Vec model weights (requires separate download)
├── data/                   # Data processing folder (not fully organized)
├── infer/                  # Inference code folder
│   ├── Embeat.py           # Embeat recommendation system core
│   ├── EmbeatUtils.py      # Embeat extension utilities
│   ├── infer.py            # EmbeatMLP inference entry point
│   ├── eval_infer.py       # EmbeatMLP evaluation utilities
│   └── hf_to_qdrant.py     # HF Dataset to Qdrant database
├── train/                  # Training code folder
│   ├── model.py            # EmbeatMLP model definition
│   ├── dataset.py          # HF Dataset processing
│   ├── sampler.py          # Positive/negative sample sampler
│   ├── loss.py             # Masked InfoNCE Loss
│   ├── trainer.py          # EmbeatMLP trainer
│   ├── train.py            # EmbeatMLP training entry point
│   └── train_track2vec.py  # Track2Vec training entry point
├── requirements.txt
└── LICENSE
```


## Getting Started

### Requirements (recommended)

- Python >= 3.10
- PyTorch >= 2.6, < 2.7 (required for training)
- CUDA >= 12.0 (required for training)
- [Qdrant](https://github.com/qdrant/qdrant/releases) (required for inference)

### Installation

```bash
conda create -n embeat python=3.10
conda activate embeat

# Install PyTorch (CUDA 12.x), see https://pytorch.org/get-started/previous-versions/
pip install "torch>=2.6,<2.7" --index-url https://download.pytorch.org/whl/cu126

pip install -r requirements.txt
```

### Train EmbeatMLP

```bash
# Prepare training data in HuggingFace Dataset format under data/datasets/, rename to spotify_45m_tracks_metadata
python -m train.train \
    --dataset data/datasets/spotify_45m_tracks_metadata@10000000 \
    --batch-size 4096 \
    --max-steps 200 \
    --lr 1e-4 \
    --tau 0.05 \
    --ckpt-dir checkpoints
```

### Train Track2Vec

```bash
# Prepare playlist training data (txt format, one playlist per line, space-separated track_ids)
cd train
python train_track2vec.py
```

### Inference: Compute Acoustic Similarity Between Two Tracks

```python
from infer.infer import infer

song_a = {"key": 7, "mode": 1, "tempo": 137, "time_signature": 4,
          "danceability": 0.54, "energy": 0.56, "speechiness": 0.02,
          "instrumentalness": 0.0, "valence": 0.41, "acousticness": 0.23,
          "liveness": 0.1}

song_b = {"key": 5, "mode": 0, "tempo": 87, "time_signature": 4,
          "danceability": 0.67, "energy": 0.65, "speechiness": 0.05,
          "instrumentalness": 0.03, "valence": 0.57, "acousticness": 0.27,
          "liveness": 0.19}

similarity = infer(sample_a=song_a, sample_b=song_b,
                   checkpoint_path="checkpoints/EmbeatMLP/model.pt")

print(f"Similarity: {similarity}")
```

### Inference: Qdrant-Based Music Recommendation

```bash
# 1. Start the Qdrant service and import the database
# 2. Query recommendations via command line
cd infer
python Embeat.py -t 5pIcwtJYNJx93l420oR2Vm   # Query by Spotify Track ID
python Embeat.py -s "晴天 - Jay Chou"   # Query by track name and artist
python Embeat.py -a "Jay Chou"   # Query by artist name
```


## Related Links

<p align="center">
  <img src="assets/gdmusic_embeat.png" alt="GDMusic Embeat" width="100%">
</p>

- GD Music (Live Demo): [https://music.gdstudio.xyz](https://music.gdstudio.xyz)
- Bilibili: [https://space.bilibili.com/13715770](https://space.bilibili.com/13715770)
- Telegram: [https://t.me/gdstudio_music](https://t.me/gdstudio_music)


## Acknowledgements

- [Anna's Archive](https://annas-archive.org)
- [Every Noise at Once](https://everynoise.com)


## License

| Scope | License |
|-------|---------|
| Code, Model Weights | MIT |
| Datasets, Database | [CC-BY-NC 4.0](LICENSE) |
