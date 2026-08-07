# -*- coding: utf-8 -*-
# Written by GD Studio
# Date: 2026-04-23

import cloudscraper
import json
import os
import random
import sys
import time
from natsort import natsorted

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from infer.model_infer import infer

song_dict = {
    "周杰伦 - 晴天": {'key': 7, 'mode': 1, 'tempo': 137, 'time_signature': 4, 'danceability': 0.54, 'energy': 0.56, 'speechiness': 0.02, 'instrumentalness': 0.0, 'valence': 0.41, 'acousticness': 0.23, 'liveness': 0.1},
    "周杰伦 - 夜曲": {'key': 5, 'mode': 0, 'tempo': 87, 'time_signature': 4, 'danceability': 0.67, 'energy': 0.65, 'speechiness': 0.05, 'instrumentalness': 0.03, 'valence': 0.57, 'acousticness': 0.27, 'liveness': 0.19},
    "Lia - 鳥の詩": {'key': 11, 'mode': 0, 'tempo': 122, 'time_signature': 4, 'danceability': 0.47, 'energy': 0.84, 'speechiness': 0.03, 'instrumentalness': 0.93, 'valence': 0.54, 'acousticness': 0.04, 'liveness': 0.21},
    "彩菜 - Last regrets": {'key': 11, 'mode': 0, 'tempo': 93, 'time_signature': 4, 'danceability': 0.47, 'energy': 0.79, 'speechiness': 0.03, 'instrumentalness': 0.73, 'valence': 0.17, 'acousticness': 0.0, 'liveness': 0.22},
    "Abel Korzeniowski - Dance For Me Wallis": {'key': 7, 'mode': 0, 'tempo': 86, 'time_signature': 4, 'danceability': 0.4, 'energy': 0.36, 'speechiness': 0.03, 'instrumentalness': 0.91, 'valence': 0.16, 'acousticness': 0.89, 'liveness': 0.15},
    "Abel Korzeniowski - Charms": {'key': 8, 'mode': 0, 'tempo': 90, 'time_signature': 4, 'danceability': 0.24, 'energy': 0.31, 'speechiness': 0.04, 'instrumentalness': 0.9, 'valence': 0.07, 'acousticness': 0.93, 'liveness': 0.13},
    "The Beatles - While My Guitar Gently Weeps Remastered 2009": {'key': 4, 'mode': 0, 'tempo': 115, 'time_signature': 4, 'danceability': 0.45, 'energy': 0.65, 'speechiness': 0.03, 'instrumentalness': 0.0, 'valence': 0.7, 'acousticness': 0.02, 'liveness': 0.17},
    "Regina Spektor - While My Guitar Gently Weeps": {'key': 11, 'mode': 0, 'tempo': 113, 'time_signature': 4, 'danceability': 0.7, 'energy': 0.32, 'speechiness': 0.03, 'instrumentalness': 0.0, 'valence': 0.22, 'acousticness': 0.24, 'liveness': 0.13},
    "Satoshi Takebe - Summer of Farewells": {'key': 4, 'mode': 0, 'tempo': 105, 'time_signature': 3, 'danceability': 0.48, 'energy': 0.33, 'speechiness': 0.03, 'instrumentalness': 0.0, 'valence': 0.25, 'acousticness': 0.83, 'liveness': 0.12},
    "曲锦楠 - 霞光": {'key': 5, 'mode': 0, 'tempo': 125, 'time_signature': 3, 'danceability': 0.66, 'energy': 0.17, 'speechiness': 0.04, 'instrumentalness': 0.0, 'valence': 0.17, 'acousticness': 0.62, 'liveness': 0.08},
    "Satoshi Takebe - What Is a Youth?": {'key': 4, 'mode': 0, 'tempo': 119, 'time_signature': 3, 'danceability': 0.5, 'energy': 0.09, 'speechiness': 0.04, 'instrumentalness': 0.0, 'valence': 0.16, 'acousticness': 0.96, 'liveness': 0.1},
    "Satoshi Takebe - Beauty and the Beast": {'key': 0, 'mode': 1, 'tempo': 128, 'time_signature': 4, 'danceability': 0.6, 'energy': 0.84, 'speechiness': 0.06, 'instrumentalness': 0.0, 'valence': 0.53, 'acousticness': 0.0, 'liveness': 0.07},
    "梶浦 由記 - Sis puella magica!": {'key': 7, 'mode': 0, 'tempo': 148, 'time_signature': 3, 'danceability': 0.44, 'energy': 0.41, 'speechiness': 0.03, 'instrumentalness': 0.0, 'valence': 0.4, 'acousticness': 0.62, 'liveness': 0.08},
    "梶浦 由記 - Decretum": {'key': 0, 'mode': 0, 'tempo': 160, 'time_signature': 3, 'danceability': 0.14, 'energy': 0.61, 'speechiness': 0.04, 'instrumentalness': 0.92, 'valence': 0.12, 'acousticness': 0.2, 'liveness': 0.2},
    "梶浦 由記 - sand dream": {'key': 7, 'mode': 0, 'tempo': 82, 'time_signature': 4, 'danceability': 0.4, 'energy': 0.67, 'speechiness': 0.04, 'instrumentalness': 0.39, 'valence': 0.38, 'acousticness': 0.24, 'liveness': 0.27},
    "Eleni Karaindrou - Karaindrou: Waltz By The River": {'key': 2, 'mode': 0, 'tempo': 135, 'time_signature': 3, 'danceability': 0.28, 'energy': 0.08, 'speechiness': 0.06, 'instrumentalness': 0.05, 'valence': 0.08, 'acousticness': 0.74, 'liveness': 0.12},
    "Eleni Karaindrou - Karaindrou: Dance Theme": {'key': 2, 'mode': 0, 'tempo': 136, 'time_signature': 3, 'danceability': 0.24, 'energy': 0.14, 'speechiness': 0.04, 'instrumentalness': 0.84, 'valence': 0.12, 'acousticness': 0.94, 'liveness': 0.11},
    "Eleni Karaindrou - Eternity And A Day:3. Eternity Theme": {'key': 10, 'mode': 0, 'tempo': 144, 'time_signature': 3, 'danceability': 0.22, 'energy': 0.23, 'speechiness': 0.03, 'instrumentalness': 0.85, 'valence': 0.37, 'acousticness': 0.87, 'liveness': 0.12},
    "Oskar Schuster - Gizeh": {'key': 2, 'mode': 0, 'tempo': 142, 'time_signature': 3, 'danceability': 0.53, 'energy': 0.2, 'speechiness': 0.08, 'instrumentalness': 0.87, 'valence': 0.69, 'acousticness': 0.98, 'liveness': 0.12},
    "Gorillaz - Feel Good Inc.": {'key': 6, 'mode': 1, 'tempo': 139, 'time_signature': 4, 'danceability': 0.82, 'energy': 0.7, 'speechiness': 0.18, 'instrumentalness': 0.0, 'valence': 0.77, 'acousticness': 0.01, 'liveness': 0.61},
    "Kevin Penkin - Become the God": {'key': 0, 'mode': 0, 'tempo': 180, 'time_signature': 4, 'danceability': 0.2, 'energy': 0.59, 'speechiness': 0.13, 'instrumentalness': 0.14, 'valence': 0.11, 'acousticness': 0.0, 'liveness': 0.34},
    "小畑貴裕 - 63194": {'key': 0, 'mode': 0, 'tempo': 154, 'time_signature': 4, 'danceability': 0.24, 'energy': 0.53, 'speechiness': 0.03, 'instrumentalness': 0.85, 'valence': 0.2, 'acousticness': 0.03, 'liveness': 0.08},
    "Koji Tamaki - 行かないで": {'key': 11, 'mode': 0, 'tempo': 81, 'time_signature': 4, 'danceability': 0.31, 'energy': 0.27, 'speechiness': 0.03, 'instrumentalness': 0.55, 'valence': 0.09, 'acousticness': 0.88, 'liveness': 0.09},
    "Jacky Cheung - 李香蘭": {'key': 9, 'mode': 0, 'tempo': 77, 'time_signature': 4, 'danceability': 0.35, 'energy': 0.2, 'speechiness': 0.03, 'instrumentalness': 0.0, 'valence': 0.08, 'acousticness': 0.72, 'liveness': 0.05},
    "Koji Tamaki - 初恋": {'key': 4, 'mode': 0, 'tempo': 125, 'time_signature': 4, 'danceability': 0.67, 'energy': 0.49, 'speechiness': 0.03, 'instrumentalness': 0.0, 'valence': 0.61, 'acousticness': 0.82, 'liveness': 0.12},
    "Samantha Lam - 初戀": {'key': 1, 'mode': 0, 'tempo': 130, 'time_signature': 4, 'danceability': 0.65, 'energy': 0.57, 'speechiness': 0.04, 'instrumentalness': 0.0, 'valence': 0.66, 'acousticness': 0.5, 'liveness': 0.07}
}


def search_song_tags(track_artist: str, track_title: str = ""):
    time.sleep(3)
    track_artist = track_artist.split(",")[0].strip()
    result = {
        "artist_tags": {
            "artist_name": track_artist,
            "styles": []
        },
        "track_tags": {
            "track_title": "",
            "album_name": "",
            "release_date": "",
            "key": "",
            "bpm": 0,
            "energy": 0.0,
            "happiness": 0.0,
            "danceability": 0.0,
            "acousticness": 0.0,
            "instrumentalness": 0.0
        }
    }
    track_title = track_title.strip()
    if not track_artist:
        result_str = str(json.dumps(result, ensure_ascii=False, indent=2))
        return result_str
    headers = {
        "User-Agent": f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{random.randrange(101, 138)}.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7",
        "DNT": "1",
        "Connection": "keep-alive"
    }
    headers['Origin'] = "https://tunebat.com"
    headers['Referer'] = "https://tunebat.com/"
    url = "https://api.tunebat.com/api/tracks/search"
    params = {
        "term": f"{track_title} - {track_artist}",
        "page": 1
    }
    try:
        scraper = cloudscraper.create_scraper()
        res = scraper.get(url=url, params=params, headers=headers, timeout=5)
        if res.status_code == 200:
            track_tags = res.json()['data']['items'][0]
            result['track_tags']['track_title'] = str(track_tags.get("n", ""))
            result['track_tags']['album_name'] = str(track_tags.get("an", ""))
            result['track_tags']['release_date'] = str(track_tags.get("rd", ""))
            result['track_tags']['key'] = str(track_tags.get("k", ""))
            result['track_tags']['bpm'] = int(track_tags.get("b", 0))
            result['track_tags']['energy'] = round(float(track_tags.get("e", 0.0)), 2)
            result['track_tags']['happiness'] = round(float(track_tags.get("h", 0.0)), 2)
            result['track_tags']['danceability'] = round(float(track_tags.get("da", 0.0)), 2)
            result['track_tags']['acousticness'] = round(float(track_tags.get("ac", 0.0)), 2)
            result['track_tags']['liveness'] = round(float(track_tags.get("li", 0.0)), 2)
            result['track_tags']['instrumentalness'] = round(float(track_tags.get("i", 0.0)), 2)
            result['track_tags']['speechiness'] = round(float(track_tags.get("s", 0.0)), 2)
    except:
        pass
    return result


def tunebat_to_embeat(tags: dict):
    embeat_input = {
        "key": 0,
        "mode": 0,
        "tempo": 0,
        "time_signature": 4,
        "danceability": 0.0,
        "energy": 0.0,
        "speechiness": 0.0,
        "instrumentalness": 0.0,
        "valence": 0.0,
        "acousticness": 0.0,
        "liveness": 0.0
    }
    track_tags = tags['track_tags']
    if track_tags.get("track_title") is None:
        raise ValueError("Tunebat error.")
    KEY_NAME_TO_PITCH_CLASS = {
        "C": 0,
        "C#": 1,
        "DB": 1,
        "D": 2,
        "D#": 3,
        "EB": 3,
        "E": 4,
        "F": 5,
        "F#": 6,
        "GB": 6,
        "G": 7,
        "G#": 8,
        "AB": 8,
        "A": 9,
        "A#": 10,
        "BB": 10,
        "B": 11,
    }
    MODE_NAME_TO_MODE_CLASS = {
        "minor": 0,
        "major": 1
    }
    key_name, mode_name = track_tags['key'].split(" ")
    key_name = key_name.replace("♯", "#").replace("♭", "b").upper().strip()
    mode_name = mode_name.lower().strip()
    embeat_input['key'] = KEY_NAME_TO_PITCH_CLASS[key_name]
    embeat_input['mode'] = MODE_NAME_TO_MODE_CLASS[mode_name]
    embeat_input['tempo'] = int(track_tags['bpm'])
    # embeat_input['time_signature'] = int(track_tags['time_signature'])
    embeat_input['danceability'] = float(track_tags['danceability'])
    embeat_input['energy'] = float(track_tags['energy'])
    embeat_input['speechiness'] = float(track_tags['speechiness'])
    embeat_input['instrumentalness'] = float(track_tags['instrumentalness'])
    embeat_input['valence'] = float(track_tags['happiness'])
    embeat_input['acousticness'] = float(track_tags['acousticness'])
    embeat_input['liveness'] = float(track_tags['liveness'])
    return embeat_input


def compare_two_songs(song_a_name: str, song_b_name: str):
    if song_a_name in song_dict:
        embeat_input_a = song_dict[song_a_name]
    else:
        track_artist_a, track_title_a = song_a_name.strip().split(" - ")
        song_a_tags = search_song_tags(track_artist=track_artist_a, track_title=track_title_a)
        embeat_input_a = tunebat_to_embeat(song_a_tags)
    if song_b_name in song_dict:
        embeat_input_b = song_dict[song_b_name]
    else:
        track_artist_b, track_title_b = song_b_name.strip().split(" - ")
        song_b_tags = search_song_tags(track_artist=track_artist_b, track_title=track_title_b)
        embeat_input_b = tunebat_to_embeat(song_b_tags)
    t1 = time.time()
    checkpoint_path = "train/checkpoints/model.pt"
    sim = infer(sample_a=embeat_input_a, sample_b=embeat_input_b, checkpoint_path=checkpoint_path)
    t2 = time.time()
    print(f"Checkpoint: {checkpoint_path}")
    print(song_a_name)
    print(f"Sample A: {embeat_input_a}")
    print(song_b_name)
    print(f"Sample B: {embeat_input_b}")
    print(f"Cosine similarity: {round(sim, 4)}")
    print(f"Used time: {round(((t2 - t1) * 1000), 3)}ms")


def eval_two_checkpoints(checkpoint_a: str, checkpoint_b: str):
    candidates = [
        {
            "song_a_name": "Abel Korzeniowski - Dance For Me Wallis",
            "song_b_name": "Abel Korzeniowski - Charms",
            "is_positive": True
        },
        {
            "song_a_name": "Eleni Karaindrou - Karaindrou: Waltz By The River",
            "song_b_name": "Eleni Karaindrou - Karaindrou: Dance Theme",
            "is_positive": True
        },
        {
            "song_a_name": "Eleni Karaindrou - Karaindrou: Dance Theme",
            "song_b_name": "Oskar Schuster - Gizeh",
            "is_positive": True
        },
        {
            "song_a_name": "梶浦 由記 - Decretum",
            "song_b_name": "梶浦 由記 - Sis puella magica!",
            "is_positive": True
        },
        {
            "song_a_name": "Satoshi Takebe - Summer of Farewells",
            "song_b_name": "曲锦楠 - 霞光",
            "is_positive": True
        },
        {
            "song_a_name": "Koji Tamaki - 行かないで",
            "song_b_name": "Jacky Cheung - 李香蘭",
            "is_positive": True
        },
        {
            "song_a_name": "Koji Tamaki - 初恋",
            "song_b_name": "Samantha Lam - 初戀",
            "is_positive": True
        },
        {
            "song_a_name": "The Beatles - While My Guitar Gently Weeps Remastered 2009",
            "song_b_name": "Regina Spektor - While My Guitar Gently Weeps",
            "is_positive": True
        },
        {
            "song_a_name": "周杰伦 - 晴天",
            "song_b_name": "Lia - 鳥の詩",
            "is_positive": False
        },
        {
            "song_a_name": "Oskar Schuster - Gizeh",
            "song_b_name": "Gorillaz - Feel Good Inc.",
            "is_positive": False
        },
        {
            "song_a_name": "Oskar Schuster - Gizeh",
            "song_b_name": "梶浦 由記 - sand dream",
            "is_positive": False
        },
        {
            "song_a_name": "梶浦 由記 - Decretum",
            "song_b_name": "Kevin Penkin - Become the God",
            "is_positive": False
        }
    ]
    print(f"Checkpoint A: {checkpoint_a}")
    print(f"Checkpoint B: {checkpoint_b}")
    checkpoint_a_win = 0
    checkpoint_b_win = 0
    for candidate in candidates:
        embeat_input_a = song_dict[candidate['song_a_name']]
        embeat_input_b = song_dict[candidate['song_b_name']]
        ckpt_a_sim = infer(sample_a=embeat_input_a, sample_b=embeat_input_b, checkpoint_path=checkpoint_a)
        ckpt_b_sim = infer(sample_a=embeat_input_a, sample_b=embeat_input_b, checkpoint_path=checkpoint_b)
        print(f"Song A: {candidate['song_a_name']} | Song B: {candidate['song_b_name']}")
        if not candidate['is_positive']:
            ckpt_a_sim = -ckpt_a_sim
            ckpt_b_sim = -ckpt_b_sim
            win_point = 1.0
        else:
            win_point = 1.0
        if ckpt_a_sim > ckpt_b_sim:
            checkpoint_a_win = checkpoint_a_win + win_point
            print(f"A Score: {round(ckpt_a_sim, 4)} | B Score: {round(ckpt_b_sim, 4)} | A win")
        elif ckpt_b_sim > ckpt_a_sim:
            checkpoint_b_win = checkpoint_b_win + win_point
            print(f"A Score: {round(ckpt_a_sim, 4)} | B Score: {round(ckpt_b_sim, 4)} | B win")
        else:
            checkpoint_a_win = checkpoint_a_win + win_point
            checkpoint_b_win = checkpoint_b_win + win_point
            print(f"A Score: {round(ckpt_a_sim, 4)} | B Score: {round(ckpt_b_sim, 4)} | Same")
    if checkpoint_a_win > checkpoint_b_win:
        print(f"Winner is checkpoint A: {checkpoint_a}\n")
        return 1
    elif checkpoint_b_win > checkpoint_a_win:
        print(f"Winner is checkpoint B: {checkpoint_b}\n")
        return 2
    else:
        print("Checkpoints A and B got the same score.\n")
        return 0


def eval_checkpoints(checkpoint_dir: str):
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    ckpt_files = os.listdir(checkpoint_dir)
    ckpt_files = [f for f in ckpt_files if f.endswith(".pt")]
    ckpt_files = natsorted(ckpt_files)
    if len(ckpt_files) <= 1:
        print(f"checkpoint_dir should have at least 2 checkpoint files, but got {len(ckpt_files)}.")
        return
    checkpoint_a = ""
    checkpoint_b = ""
    for i in range(len(ckpt_files)):
        if i == 0:
            checkpoint_a = f"{checkpoint_dir}/{ckpt_files[0]}"
            continue
        checkpoint_b = f"{checkpoint_dir}/{ckpt_files[i]}"
        result = eval_two_checkpoints(checkpoint_a=checkpoint_a, checkpoint_b=checkpoint_b)
        if result in [1, 0]:
            checkpoint_b = ""
        else:
            checkpoint_a = checkpoint_b
            checkpoint_b = ""
    print(f"Final winner: {checkpoint_a}")


if __name__ == "__main__":
    # result = search_song_tags(track_artist="Satoshi Takebe", track_title="Summer of Farewells")
    # print(result)
    # embeat_input = tunebat_to_embeat(result)
    # print(embeat_input)

    # song_a_name = "Eleni Karaindrou - Karaindrou: Waltz By The River"
    # song_b_name = "Eleni Karaindrou - Karaindrou: Dance Theme"
    # compare_two_songs(song_a_name, song_b_name)

    # checkpoint_a = "checkpoints/step_10.pt"
    # checkpoint_b = "checkpoints/step_70.pt"
    # eval_two_checkpoints(checkpoint_a=checkpoint_a, checkpoint_b=checkpoint_b)

    eval_checkpoints("checkpoints/")
