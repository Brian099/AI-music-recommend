<p align="center">
  <img src="assets/banner.png" alt="Embeat Banner" width="100%">
</p>

<p align="center">
  <a href="README.md">English</a> | <b>简体中文</b>
</p>

<p align="center">
  <a href="https://github.com/gdstudio-org/Embeat">主页</a> •
  <a href="https://www.bilibili.com/opus/1218087093501165591">技术文档</a> •
  <a href="https://github.com/gdstudio-org/Embeat/tree/main/checkpoints">模型</a> •
  <a href="https://huggingface.co/datasets/GD-Studio/embeat_45m_spotify_tracks">数据集</a> •
  <a href="https://pan.baidu.com/s/1CWFzgM75Z4YjP1tZnGCZKg?pwd=0616">数据库</a>
</p>

<p align="center">
  <a href="https://github.com/gdstudio-org/Embeat"><img src="https://img.shields.io/github/stars/gdstudio-org/Embeat?style=social" alt="Stars"></a>
  <a href="https://github.com/gdstudio-org/Embeat/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-CC--BY--NC%204.0-blue" alt="License"></a>
</p>

---

# Embeat：基于声学特征的音乐推荐系统

## 简介

Embeat 是一个基于 Spotify 声学特征数据构建的歌曲推荐系统，通过**对比学习模型**将音频特征编码为向量，结合**多路召回**策略实现高质量的音乐推荐。

**核心特点：**

- **声学相似**：基于 Spotify Audio Features（调性、节拍、能量、情绪等）训练的 EmbeatMLP 模型，将声学特征编码为 64 维向量
- **流派感知**：引入 6000+ 微流派标签，为 200 万+ 歌手精准分配流派，避免"声学相似但风格迥异"的推荐
- **多路召回**：5 路并行召回（声学相似 / 同流派热门 / 同歌手 / 相似歌手 / 歌单协同过滤），融合打分后输出
- **歌单协同过滤**：通过 Track2Vec（Word2Vec 思路）学习 188 万份 Spotify 歌单中歌曲的共现关系
- **毫秒级响应**：基于 Qdrant 向量数据库，4500 万首歌曲的检索响应在 30~100ms 内完成


## 开源路线

> 如果觉得项目有帮助，请帮忙点个⭐️。这对个人项目来说很重要，谢谢！

- [x] **2026-06-26**：开源 初版代码 + [EmbeatMLP 模型权重](checkpoints/EmbeatMLP/)
- [x] **2026-06-26**：开源 [45M 单曲数据集](https://huggingface.co/datasets/GD-Studio/embeat_45m_spotify_tracks) + [技术文档](https://www.bilibili.com/opus/1218087093501165591)
- [x] **2026-07-02**：开源 [Qdrant 数据库](https://pan.baidu.com/s/1CWFzgM75Z4YjP1tZnGCZKg?pwd=0616) (提取码：0616)
- [ ] **1K Stars**：开源 Track2Vec 模型权重 + 1.8M 歌单数据集


## 效果展示

> 以下是 Embeat 的推荐结果示例（播放前请关闭静音）

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

### LLM 盲评对比

使用 LLM-as-a-Judge 方法（GPT-5.5 / Gemini Flash 3.5 / Claude Sonnet 4.6），对 Embeat 与网易云音乐进行 AB 盲测：

| 评估模型 | Embeat胜 | 网易云胜 | 平局 |
|---------|:---------:|:-------:|:----:|
| Claude Sonnet 4.6 | **8** | 2 | 0 |
| Gemini Flash 3.5 | **9** | 1 | 0 |
| GPT 5.5 | **6** | 4 | 0 |

**结论：**

- Embeat 的核心优势在于风格精准度和艺人多样性的平衡，尤其在跨语言、跨文化的小众风格场景中优势突出
- 网易云音乐仅在华语本土内容的深度挖掘上仍有一定参考价值
- 具体对比细节请阅读 [技术文档](https://www.bilibili.com/opus/1218087093501165591)


## 系统架构

### 模型说明

**EmbeatMLP** - 声学特征编码模型

- 输入：64 维离散特征（key, mode, tempo, time_signature）+ 64 维连续特征（energy, valence, danceability 等 7 维）
- 架构：双塔 MLP（Discrete Tower + Acoustic Tower -> Backbone）
- 输出：64 维 L2 归一化向量
- 训练：Masked InfoNCE Loss，batch_size=4096，~70 steps 即可收敛
- 参数量极小，支持纯 CPU 实时推理

**Track2Vec** - 歌单协同过滤模型

- 基于 Word2Vec Skip-Gram，将歌单视为"句子"，歌曲视为"单词"
- 训练数据：188 万份 Spotify 歌单
- 词表：109 万首歌曲，64 维向量
- 支持纯 CPU 实时推理，单次查询耗时 < 200ms

### 多路召回

```
输入种子曲目: track_id / track_name + artist_name
  │
  ├─ 第1路 [similar]: 声学相似召回 (流派过滤 + EmbeatMLP余弦相似)
  ├─ 第2路 [popular]: 同流派热门召回 (流派过滤 + 热度排序)
  ├─ 第3路 [same_artist]: 同歌手召回 (相同歌手 + EmbeatMLP余弦相似)
  ├─ 第4路 [related_artist]: 相似歌手召回 (相似歌手 + EmbeatMLP余弦相似)
  ├─ 第5路 [related_track]: 歌单协同过滤 (Track2Vec余弦相似)
  │
  ├─ ISRC去重 / 重排打分 / 同歌手比例控制
  │
  └─ 输出: Top-K 推荐列表
```

### 项目结构

```
Embeat/
├── assets/                 # 资源文件目录
├── checkpoints/            # 模型目录目录
│   ├── EmbeatMLP/          # EmbeatMLP 模型权重
│   └── Track2Vec/          # Track2Vec 模型权重 (需额外下载)
├── data/                   # 数据处理目录 (未完全整理)
├── infer/                  # 推理代码目录
│   ├── Embeat.py           # Embeat 推荐系统核心
│   ├── EmbeatUtils.py      # Embeat 扩展辅助工具
│   ├── infer.py            # EmbeatMLP 模型推理
│   ├── eval_infer.py       # EmbeatMLP 模型评估工具
│   └── hf_to_qdrant.py     # HF Dataset 转 Qdrant 数据库
├── train/                  # 训练代码目录
│   ├── model.py            # EmbeatMLP 模型定义
│   ├── dataset.py          # HF Dataset 数据集处理
│   ├── sampler.py          # Masked InfoNCE 正负样本采样器
│   ├── loss.py             # Masked InfoNCE Loss
│   ├── trainer.py          # EmbeatMLP 训练器
│   ├── train.py            # EmbeatMLP 训练入口
│   └── train_track2vec.py  # Track2Vec 训练入口
├── .env.example            # .env 环境变量配置示例
├── requirements.txt
└── LICENSE
```


## 快速开始

### 环境要求（推荐）

- Python >= 3.10
- PyTorch >= 2.6, < 2.7（训练需要）
- CUDA >= 12.0（训练需要）
- [Qdrant](https://github.com/qdrant/qdrant/releases)（推理需要）

### 安装依赖

```bash
conda create -n embeat python=3.10
conda activate embeat

# 安装 PyTorch (CUDA 12.x)，参考 https://pytorch.org/get-started/previous-versions/
pip install "torch>=2.6,<2.7" --index-url https://download.pytorch.org/whl/cu126

pip install -r requirements.txt
```

### 训练 EmbeatMLP

```bash
# 1. 下载 HuggingFace 单曲数据集，放到 `data/datasets/` 目录下，重命名为 `spotify_45m_tracks_metadata`
# 2. 如果你希望做更细致的训练参数调整，请阅读代码
cd train
python train.py
```

### 训练 Track2Vec

```bash
# 1. 准备歌单训练数据 (txt格式，每行一个歌单，空格分隔track_id)
# 2. 重命名为 `spotify_playlists.txt`，并放到 `train` 文件夹下
# 3. 如果你希望做更细致的训练参数调整，请阅读代码
cd train
python train_track2vec.py
```

### 推理：计算两首歌的声学相似度

```python
# 1. 从 HuggingFace 单曲数据集中获取声学特征数据
# 2. 或者你可以从 infer/eval_infer.py 找到些现成例子
from infer.infer import infer

# 晴天 - Jay Chou (G大调快节奏)
song_a = {"key": 7, "mode": 1, "tempo": 137, "time_signature": 4,
          "danceability": 0.54, "energy": 0.56, "speechiness": 0.02,
          "instrumentalness": 0.0, "valence": 0.41, "acousticness": 0.23,
          "liveness": 0.1}

# 夜曲 - Jay Chou (F小调慢节奏)
song_b = {"key": 5, "mode": 0, "tempo": 87, "time_signature": 4,
          "danceability": 0.67, "energy": 0.65, "speechiness": 0.05,
          "instrumentalness": 0.03, "valence": 0.57, "acousticness": 0.27,
          "liveness": 0.19}

# 使用 EmbeatMLP 计算声学相似度
similarity = infer(sample_a=song_a, sample_b=song_b,
                   checkpoint_path="checkpoints/EmbeatMLP/model.pt")

# 相似度：0.7312
print(f"Similarity: {similarity:.4f}")
```

### 推理：基于 Qdrant 的歌曲推荐

```bash
# 1. 启动 Qdrant 服务并导入数据库
# 2. 通过命令行查询种子曲目的推荐结果
cd infer
python Embeat.py -t 5pIcwtJYNJx93l420oR2Vm   # 通过 Spotify Track ID 查询
python Embeat.py -s "晴天 - Jay Chou"   # 通过歌名和歌手查询
python Embeat.py -a "Jay Chou"   # 通过歌手名查询


# 输出例子《晴天 - Jay Chou》：
Query track_id: 5pIcwtJYNJx93l420oR2Vm
Query track info: 晴天 - Jay Chou
Query artist genres: ['mandopop', 'taiwan pop', 'c-pop', 'zhongguo feng']
-> Find query record used time: 21ms
-> Similar recall used time: 48ms
-> Popular recall used time: 24ms
-> Same artist recall used time: 4ms
-> Related artist recall used time: 5ms
-> Related track recall used time: 123ms
-> Re-ranking used time: 2ms
Result artist genres: ['taiwan indie', 'mandopop', 'chinese viral pop', 'cantopop']
======= Top 20 items =======
index   track_id                track_name      artist_name     album_name      sources         score
1       3Qj9Fy8BPbWmICTiNkuqB7  珊瑚海  Jay Chou        11月的蕭邦      ['same_artist', 'related_track']        1.0
2       10VuSw48iPN2UK2xX9Y6P0  青花瓷  Jay Chou        我很忙  ['same_artist', 'related_track']        1.0
3       0IAgufC1FlOg1nZMmRZxRr  突然好想你      Mayday  後 青春期的詩   ['popular', 'related_artist']   1.0
4       2zB7NKVnzRh7xSUSPLErFr  明明就  Jay Chou        十二新作        ['same_artist', 'related_track']        1.0
5       5WtMlbTDNZlbN8xZ5zfXva  Our Singapore   JJ Lin  My August 9th - 50 Wonderful Years (2016 Edition)       ['similar', 'related_artist']   1.0
6       5cU1O9P0EDA0rPkPDykhIm  怎麼了  Eric Chou       終於了解自由 (Deluxe)   ['popular', 'related_track']    1.0
7       4daA20tBusVX29bUWgd8Dw  交換餘生        JJ Lin  交換餘生        ['popular', 'related_track']    1.0
8       1EgGTmmFGtlWuqgXFLrp9x  溫柔    Mayday  愛情萬歲        ['related_artist']      0.87
9       3ZuyyfGJqx9qhWTVtdMCWz  生命線 - 電視劇《院長爸爸》片頭曲       Bii     生命線 (電視劇《院長爸爸》片頭曲)       ['similar']     0.85
10      3p4UTiSIIpP4LFn0KEyEOj  十面埋伏        Eason Chan      Live For Today  ['related_artist']      0.85
11      4lhbajK3dvUcJ0UNEeCdMn  飞鸟和蝉        Ren Ran         Ren然   ['related_track']       0.84
12      3e8uw7YMiKVcIakItBENqm  天天晴朗（蘇打綠版）    sodagreen       秋：故事（蘇打綠版）    ['similar']     0.83
13      26O8PmJ32hwAbZnIhbJJwZ  天使    Mayday  為愛而生        ['related_artist']      0.82
14      3LgoekU3dE5ZMLvuL3NIt9  清醒 (戲劇《淺情人不知》片尾曲)         Ariel Tsai      清醒 (戲劇《淺情人不知》片尾曲)         ['similar']     0.81
15      1WnTw4Tzpc5q9dHMjs4aHu  陰天快樂        Eason Chan      rice & shine    ['related_artist']      0.8
16      14GFYAUxkeXranhS2qrYIZ  我想要佔據你    告五人  帶你飛  ['related_track']       0.8
17      1ylx8p71GKQy5g1t4gzuEz  抱歉    Sam Lee         原諒我沒有說    ['similar']     0.79
18      7fAdinC2UTc0Y9GiKrkTtu  字字句句        卢卢快闭嘴      字字句句        ['related_track']       0.78
19      1VG8o5rUZQZ0wjs7Bi4siU  最熟悉的陌生人  Elva Hsiao      蕭亞軒 (最熟悉的)       ['similar']     0.77
20      1lM4cYuhJHSsDRfD0ZCRN7  你的背包        Eason Chan      陳奕迅 國語精選 (HQCDII)        ['related_artist']      0.77

Query used time: 0.229s
```


## 相关链接

<p align="center">
  <img src="assets/gdmusic_embeat.png" alt="GDMusic Embeat" width="100%">
</p>

- GD音乐台（在线体验）：[https://music.gdstudio.xyz](https://music.gdstudio.xyz)
- B站：[https://space.bilibili.com/13715770](https://space.bilibili.com/13715770)
- TG群：[https://t.me/gdstudio_music](https://t.me/gdstudio_music)


## 特别鸣谢

- [Anna's Archive](https://annas-archive.org)
- [Every Noise at Once](https://everynoise.com)


## 开源协议

| 范围 | 协议 |
|------|------|
| 代码、模型权重 | MIT |
| 数据集、数据库 | [CC-BY-NC 4.0](LICENSE) |

> Made with ❤️ by GD Studio
