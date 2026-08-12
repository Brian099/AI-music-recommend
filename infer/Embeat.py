# -*- coding: utf-8 -*-
# Written by GD Studio
# Date: 2026-05-29

import json
import numpy as np
import os
import random
import requests
import sys
import time
import uuid
from collections import Counter
DISABLED_PACKAGES = []
try:
    from gensim.models import KeyedVectors
    _GENSIM_AVAILABLE = True
except Exception as e:
    _GENSIM_AVAILABLE = False
    DISABLED_PACKAGES.append("gensim")
from typing import Any, Union, Literal
from numpy.linalg import norm
from urllib.parse import urlparse
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qdrant_models
except ImportError:
    pass
try:
    import hdbscan
except Exception as e:
    DISABLED_PACKAGES.append("hdbscan")
    print(f"Failed to import optional package `hdbscan` and skip: {e}")
try:
    from dotenv import dotenv_values
except Exception as e:
    DISABLED_PACKAGES.append("dotenv")
    print(f"Failed to import optional package `python-dotenv` and skip: {e}")
try:
    from zhconv import convert
except Exception as e:
    DISABLED_PACKAGES.append("zhconv")
    print(f"Failed to import optional package `zhconv` and skip: {e}")
try:
    from EmbeatUtils import *
except Exception as e:
    DISABLED_PACKAGES.append("embeat_utils")
    print(f"Failed to import optional module `embeat_utils` and skip: {e}")


class EmbeatDatabase:
    def __init__(self,
                qdrant_url: str = "http://127.0.0.1:6333", qdrant_api_key: str = "", collection_name: str = "spotify_tracks", qdrant_timeout: int = 30,
                engenremap_path: str = "", artist_genre_idx_patch_path: str = "", related_artist_idx_path: str = "", track2vec_path: str = "",
                enable_name_search: bool = True, is_zhconv: bool = False, use_track_genre: bool = False, verbose_log: bool = True,
                same_artist_ratio_range: list = [0.15, 0.2], popular_ratio: float = 0.1, min_popularity: float = 0.1, min_related_track_score: float = 0.75,
                recall_similar_weights: list = [1.7, 1.0], recall_popular_weights: list = [1.0, 0.8], recall_same_artist_weights: list = [1.9, 1.0], recall_related_artist_weights: list = [1.8, 1.0], recall_related_track_weights: list = [2.0, 1.2]):
        self.file_dir = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.dirname(os.path.abspath(__file__))
        self.file_dir = str(self.file_dir).replace("\\", "/").rstrip("/")
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.collection_name = collection_name
        self.qdrant_timeout = qdrant_timeout
        self.enable_name_search = enable_name_search
        self.is_zhconv = is_zhconv
        self.use_track_genre = use_track_genre
        self.engenremap_path = engenremap_path
        self.artist_genre_idx_patch_path = artist_genre_idx_patch_path
        self.related_artist_idx_path = related_artist_idx_path
        self.track2vec_path = track2vec_path
        self.same_artist_ratio_range = same_artist_ratio_range
        self.popular_ratio = popular_ratio
        self.min_popularity = min_popularity
        self.min_related_track_score = min_related_track_score
        self.recall_similar_weights = recall_similar_weights
        self.recall_popular_weights = recall_popular_weights
        self.recall_same_artist_weights = recall_same_artist_weights
        self.recall_related_artist_weights = recall_related_artist_weights
        self.recall_related_track_weights = recall_related_track_weights
        self.verbose_log = verbose_log
        self.load_env()
        self.wait_qdrant_ready()
        self.build_qdrant_client()
        self.get_collection_version()
        self.read_json_files()
        self.read_track2vec()

    def load_env(self):
        if "dotenv" in DISABLED_PACKAGES:
            return
        env_file = f"{self.file_dir}/.env"
        if not os.path.isfile(env_file):
            return
        self.env_config = dotenv_values(env_file)
        for key, value in self.env_config.items():
            if key.startswith("EMBEAT_"):
                key = key.replace("EMBEAT_", "").strip().lower()
                setattr(self, key, value)
        self.enable_name_search = True if self.env_config.get("EMBEAT_ENABLE_NAME_SEARCH", "0").lower() in ["1", "true"] else False
        self.verbose_log = True if self.env_config.get("EMBEAT_VERBOSE_LOG", "0").lower() in ["1", "true"] else False
        self.is_zhconv = True if self.env_config.get("EMBEAT_ZHCONV", "0").lower() in ["1", "true"] else False
        self.use_track_genre = True if self.env_config.get("EMBEAT_USE_TRACK_GENRE", "0").lower() in ["1", "true"] else False
        self.is_output_extra = True if self.env_config.get("IS_OUTPUT_EXTRA", "0").lower() in ["1", "true"] else False
        self.qdrant_url = str(self.qdrant_url).strip().rstrip("/")
        if isinstance(self.same_artist_ratio_range, str):
            try:
                self.same_artist_ratio_range = [float(ratio) for ratio in self.same_artist_ratio_range.split(",")]
            except Exception as e:
                print(f"Failed to parse params `EMBEAT_SAME_ARTIST_RATIO_RANGE` from .env: {e}")
        if isinstance(self.recall_similar_weights, str):
            try:
                self.recall_similar_weights = [float(ratio) for ratio in self.recall_similar_weights.split(",")]
            except Exception as e:
                print(f"Failed to parse params `EMBEAT_RECALL_SIMILAR_WEIGHTS` from .env: {e}")
        if isinstance(self.recall_popular_weights, str):
            try:
                self.recall_popular_weights = [float(ratio) for ratio in self.recall_popular_weights.split(",")]
            except Exception as e:
                print(f"Failed to parse params `EMBEAT_RECALL_POPULAR_WEIGHTS` from .env: {e}")
        if isinstance(self.recall_same_artist_weights, str):
            try:
                self.recall_same_artist_weights = [float(ratio) for ratio in self.recall_same_artist_weights.split(",")]
            except Exception as e:
                print(f"Failed to parse params `EMBEAT_RECALL_SAME_ARTIST_WEIGHTS` from .env: {e}")
        if isinstance(self.recall_related_artist_weights, str):
            try:
                self.recall_related_artist_weights = [float(ratio) for ratio in self.recall_related_artist_weights.split(",")]
            except Exception as e:
                print(f"Failed to parse params `EMBEAT_RECALL_RELATED_ARTIST_WEIGHTS` from .env: {e}")
        if isinstance(self.recall_related_track_weights, str):
            try:
                self.recall_related_track_weights = [float(ratio) for ratio in self.recall_related_track_weights.split(",")]
            except Exception as e:
                print(f"Failed to parse params `EMBEAT_RECALL_RELATED_TRACK_WEIGHTS` from .env: {e}")
        try:
            self.qdrant_timeout = int(self.qdrant_timeout)
        except Exception as e:
            print(f"Failed to parse params `EMBEAT_QDRANT_TIMEOUT` from .env: {e}")
        try:
            self.popular_ratio = float(self.popular_ratio)
        except Exception as e:
            print(f"Failed to parse params `EMBEAT_POPULAR_RATIO` from .env: {e}")
        try:
            self.min_popularity = float(self.min_popularity)
        except Exception as e:
            print(f"Failed to parse params `EMBEAT_MIN_POPULARITY` from .env: {e}")
        try:
            self.min_related_track_score = float(self.min_related_track_score)
        except Exception as e:
            print(f"Failed to parse params `EMBEAT_MIN_RELATED_TRACK_SCORE` from .env: {e}")
        return

    # Check Qdrant state
    def wait_qdrant_ready(self):
        wait_seconds = max(0, self.qdrant_timeout)
        ready_url = str(self.qdrant_url).rstrip("/") + "/readyz"
        start_time = time.time()
        while True:
            try:
                response = requests.get(ready_url, timeout=3)
                if int(response.status_code) == 200:
                    return
            except Exception as e:
                print(f"Qdrant is not ready and retry: {e}")
            if (time.time() - start_time) >= wait_seconds:
                break
            time.sleep(3.0)
        raise ConnectionError(f"Failed to connect to Qdrant: {ready_url}")

    # Connect to Qdrant database
    def build_qdrant_client(self, is_timeout: bool = True):
        qdrant_api_key = str(self.qdrant_api_key).strip()
        qdrant_api_key = qdrant_api_key if qdrant_api_key else None
        parsed_url = urlparse(self.qdrant_url)
        if parsed_url.port is not None:
            port = int(parsed_url.port)
        elif parsed_url.scheme == "https":
            port = 443
        else:
            port = 6333
        if is_timeout and self.qdrant_timeout > 0:
            timeout = self.qdrant_timeout
        else:
            timeout = 86400
        self.client = QdrantClient(url=self.qdrant_url, api_key=qdrant_api_key, port=port, timeout=timeout)
        return

    # Build Qdrant index to speed up query
    def build_qdrant_index(self):
        self.build_qdrant_client(is_timeout=False)
        if self.verbose_log:
            print("-> Building Qdrant index for necessary fields to speed up query...")
        self.client.create_payload_index(collection_name=self.collection_name, field_name="artist_genre_idx", field_schema=qdrant_models.PayloadSchemaType.INTEGER, wait=True)
        self.client.create_payload_index(collection_name=self.collection_name, field_name="artist_idx", field_schema=qdrant_models.PayloadSchemaType.INTEGER, wait=True)
        self.client.create_payload_index(collection_name=self.collection_name, field_name="popularity", field_schema=qdrant_models.PayloadSchemaType.FLOAT, wait=True)
        if self.enable_name_search:
            if self.verbose_log:
                print("-> Building Qdrant index for name search... This might spend more memory and disk usage.")
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="track_name",
                field_schema=qdrant_models.TextIndexParams(
                    type="text",
                    tokenizer=qdrant_models.TokenizerType.WORD,
                    lowercase=True,
                    on_disk=True
                ),
                wait=True
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="artist_name",
                field_schema=qdrant_models.TextIndexParams(
                    type="text",
                    tokenizer=qdrant_models.TokenizerType.WORD,
                    lowercase=True,
                    on_disk=True
                ),
                wait=True
            )
        return

    # Get Qdrant collection version by collection fields
    def get_collection_version(self, use_schema: bool = False):
        self.collection_version = "v3"
        # If collection doesn't exist yet (fresh deployment), skip version detection
        if not self.client.collection_exists(self.collection_name):
            if self.verbose_log:
                print(f"-> Collection '{self.collection_name}' not found, will be created on first /scan. Defaulting to version v3.")
            return
        collection_fields = []
        if use_schema:
            collection_info = self.client.get_collection(collection_name=self.collection_name)
            collection_fields = list(collection_info.payload_schema.keys())
        else:
            records, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=1,
                with_payload=True,
                with_vectors=False
            )
            if records and records[0].payload:
                collection_fields = list(records[0].payload.keys())
        if "artist_idx" not in collection_fields:
            self.collection_version = "v1"
        elif "isrc" not in collection_fields:
            self.collection_version = "v2"
            return
        return


    # Read json file data and keep in memory
    def read_json_files(self):
        self.genre_index_dict = {}
        self.index_genre_dict = {}
        if not self.engenremap_path:
            if os.path.isfile(f"{self.file_dir}/engenremap.json"):
                self.engenremap_path = f"{self.file_dir}/engenremap.json"
        if not self.engenremap_path or not os.path.isfile(self.engenremap_path):
            self.engenremap_path = ""
            if "build_in_genre_index_dict" in globals() and build_in_genre_index_dict:
                if self.verbose_log:
                    print("Optional file `engenremap.json` is missing. System will use a build-in genre_index map.")
                self.genre_index_dict = build_in_genre_index_dict
            else:
                if self.verbose_log:
                    print("Optional file `engenremap.json` is missing. System will use the original artist_genre_idx from Qdrant database.")
        else:
            try:
                with open(self.engenremap_path, "r", encoding="utf-8") as f:
                    engenremap = json.loads(f.read())
                self.genre_index_dict = {item['genre']: item['index'] for item in engenremap}
            except Exception as e:
                print(f"Failed to read `engenremap.json` and skip: {e}")
        self.index_genre_dict = {v: k for k, v in self.genre_index_dict.items()}
        self.artist_genre_idx_patch = {}
        if not self.artist_genre_idx_patch_path:
            if os.path.isfile(f"{self.file_dir}/artist_genre_idx_patch.json"):
                self.artist_genre_idx_patch_path = f"{self.file_dir}/artist_genre_idx_patch.json"
        if not self.artist_genre_idx_patch_path or not os.path.isfile(self.artist_genre_idx_patch_path):
            self.artist_genre_idx_patch_path = ""
            if self.verbose_log:
                print("Optional file `artist_genre_idx_patch.json` is missing. System might have a bad predition on some edge cases.")
        else:
            try:
                with open(self.artist_genre_idx_patch_path, "r", encoding="utf-8") as f:
                    artist_genre_idx_patch = json.loads(f.read())
                self.artist_genre_idx_patch = {int(k): int(v) for k, v in artist_genre_idx_patch.items()}
            except Exception as e:
                print(f"Failed to read `artist_genre_idx_patch.json` and skip: {e}")
        self.related_artist_idx = {}
        if not self.related_artist_idx_path:
            if os.path.isfile(f"{self.file_dir}/related_artist_idx.json"):
                self.related_artist_idx_path = f"{self.file_dir}/related_artist_idx.json"
        if not self.related_artist_idx_path or not os.path.isfile(self.related_artist_idx_path):
            self.related_artist_idx_path = ""
            if self.verbose_log:
                print("Optional file `related_artist_idx.json` is missing. System might not have the best predition result.")
        else:
            try:
                with open(self.related_artist_idx_path, "r", encoding="utf-8") as f:
                    related_artist_idx = json.loads(f.read())
                self.related_artist_idx = {int(k): list(v) for k, v in related_artist_idx.items()}
            except Exception as e:
                print(f"Failed to read `related_artist_idx.json` and skip: {e}")
        return
    
    # Read track2vec key-vector file data and keep in memory
    def read_track2vec(self):
        self.wv = None
        if not self.track2vec_path or not os.path.isfile(self.track2vec_path):
            self.track2vec_path = f"{self.file_dir}/track2vec.wv"
        if not os.path.isfile(self.track2vec_path):
            self.track2vec_path = f"{self.file_dir}/track2vec.bin"
        if not os.path.isfile(self.track2vec_path):
            print("Optional file `track2vec.wv` or `track2vec.bin` is missing. System might not have the best predition result.")
            return
        try:
            if self.track2vec_path.split(".")[-1].lower() == "bin":
                self.wv = KeyedVectors.load_word2vec_format(self.track2vec_path, binary=True)
            else:
                self.wv = KeyedVectors.load(self.track2vec_path)
        except Exception as e:
            print(f"Failed to read `track2vec.wv` and skip: {e}")
        return

    # Get centroid vector of one aritst and pack result
    def find_query_record_by_artist(self, artist_idx: int = 0, artist_name: str = None, is_filter_noise: bool = True):
        result = None
        if artist_idx <= 0 and not artist_name:
            print("Must provide `artist_idx` or `artist_name` for `find_query_record_by_artist`.")
            return result
        must_conditions = []
        if artist_idx > 0:
            must_conditions.append(
                qdrant_models.FieldCondition(
                    key="artist_idx", match=qdrant_models.MatchValue(value=artist_idx)
                )
            )
        else:
            must_conditions.append(
                qdrant_models.FieldCondition(
                    key="artist_name", match=qdrant_models.MatchText(text=artist_name)
                )
            )
        artist_records = []
        next_page_offset = None
        while True:
            try:
                records, next_page_offset = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=qdrant_models.Filter(must=must_conditions),
                    limit=100,
                    with_payload=True, 
                    with_vectors=True,
                    offset=next_page_offset
                )
            except Exception as e:
                print(f"Failed to run Qdrant scroll: {e}")
                return result
            for record in records:
                if record.vector and record.payload is not None:
                    artist_records.append(record)
                    if artist_idx <= 0:
                        artist_idx = int(record.payload.get("artist_idx", 0))
            if next_page_offset is None:
                break
        if not artist_records:
            return result
        vectors_array = np.array([record.vector for record in artist_records])
        if is_filter_noise and "hdbscan" not in DISABLED_PACKAGES:
            min_cluster_size = 5
            if len(vectors_array) >= min_cluster_size:
                clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
                labels = clusterer.fit_predict(vectors_array)
                valid_indices = (labels != -1)
                filtered_vectors = vectors_array[valid_indices]
                if len(filtered_vectors) > 0:
                    vectors_array = filtered_vectors
        average_vector_array = np.mean(vectors_array, axis=0)
        average_vector_norm = norm(average_vector_array)
        if average_vector_norm == 0:
            result = artist_records[0]
            return result
        max_similarity = -float("inf")
        for record in artist_records:
            current_vector = np.array(record.vector)
            current_vector_norm = norm(current_vector)
            if current_vector_norm == 0:
                continue
            current_similarity = np.dot(average_vector_array, current_vector) / (average_vector_norm * current_vector_norm)
            if current_similarity > max_similarity:
                result = record
                max_similarity = current_similarity
        return result

    # Find seed track by track_id or track info
    def find_query_record_by_track(self, track_id: str = "", track_name: str = None, artist_name: str = None):
        result = None
        if not track_id and (not track_name or not artist_name):
            print("Must provide `track_id` or `track_name & artist_name` for `find_query_record_by_track`.")
            return result
        if self.collection_version not in ["v1"] and track_id:
            target_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(track_id)))
            try:
                records = self.client.retrieve(
                    collection_name=self.collection_name,
                    ids=[target_uuid],
                    with_payload=True,
                    with_vectors=True
                )
            except Exception as e:
                print(f"Failed to run Qdrant retrieve: {e}")
                return result
        else:
            must_conditions = []
            if track_id:
                must_conditions.append(
                    qdrant_models.FieldCondition(
                        key="track_id",
                        match=qdrant_models.MatchValue(value=str(track_id))
                    )
                )
            else:
                if not track_name or not artist_name:
                    return result
                must_conditions.append(
                    qdrant_models.FieldCondition(
                        key="track_name",
                        match=qdrant_models.MatchText(text=str(track_name))
                    )
                )
                must_conditions.append(
                    qdrant_models.FieldCondition(
                        key="artist_name",
                        match=qdrant_models.MatchText(text=str(artist_name))
                    )
                )
            try:
                records, _ = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=qdrant_models.Filter(must=must_conditions),
                    limit=1,
                    with_payload=True,
                    with_vectors=True
                )
            except Exception as e:
                print(f"Failed to run Qdrant scroll: {e}")
                return result
        if not records:
            return None
        result = records[0]
        return result

    # Find all artist genre indexs
    def get_artist_genre_idxs(self, artist_genres: str):
        artist_genre_idxs = []
        artist_genres = str(artist_genres).lower().strip()
        if not artist_genres:
            return None
        artist_genre_list = artist_genres.split(", ")
        artist_genre_list = [genre.strip() for genre in artist_genre_list if genre.strip()]
        if not artist_genre_list:
            return None
        for artist_genre in artist_genre_list:
            artist_genre_idx = self.genre_index_dict.get(artist_genre)
            if artist_genre_idx is not None:
                artist_genre_idxs.append(artist_genre_idx)
        if artist_genre_idxs:
            return artist_genre_idxs
        return None

    # Estimate genre_idx of query artist from related artists
    def get_artist_genre_idx(self, query_payload: dict):
        artist_genre_idx = 0
        artist_genre_idx = query_payload.get("artist_genre_idx") or 0
        artist_idx = query_payload.get("artist_idx") or 0
        query_artist_genres = query_payload.get("artist_genres") or ""
        related_artist_idxs = list(query_payload.get("related_artist_idxs") or [])
        if artist_idx <= 0:
            return artist_genre_idx
        if artist_genre_idx > 0 and query_artist_genres:
            return artist_genre_idx
        related_artist_idxs = [idx for idx in related_artist_idxs if idx != artist_genre_idx and idx > 0]
        if not related_artist_idxs:
            related_artist_idxs = self.related_artist_idx.get(artist_genre_idx) or []
        related_artist_genre_idxs = []
        if query_artist_genres:
            related_artist_genre_idxs.extend(self.get_artist_genre_idxs(query_artist_genres))
        for related_artist_idx in related_artist_idxs:
            must_condition = qdrant_models.FieldCondition(
                key="artist_idx",
                match=qdrant_models.MatchValue(value=related_artist_idx)
            )
            try:
                records, _ = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=qdrant_models.Filter(must=must_condition),
                    limit=1,
                    with_payload=True,
                    with_vectors=True
                )
            except Exception as e:
                print(f"Failed to run Qdrant scroll: {e}")
                return artist_genre_idx
            if records and records[0].payload:
                related_artist_genres = records[0].payload.get("artist_genres") or ""
                if related_artist_genres:
                    related_artist_genre_idxs.extend(self.get_artist_genre_idxs(related_artist_genres))
        related_artist_genre_idxs = [idx for idx in related_artist_genre_idxs if idx > 0]
        if not related_artist_genre_idxs:
            return artist_genre_idx
        counter = Counter(related_artist_genre_idxs)
        top_items = counter.most_common(2)
        if len(top_items) == 0:
            return artist_genre_idx
        elif len(top_items) == 1:
            artist_genre_idx = top_items[0][0]
        else:
            top1_count = top_items[0][1]
            top2_count = top_items[1][1]
            if top1_count >= 2 * top2_count:
                artist_genre_idx = top_items[0][0]
        return artist_genre_idx

    # Estimate genre_idx of query track from similar candidates
    def get_track_genre_idx(self, query_payload: dict, candidates: list, fallback_idx: int = -1):
        if not self.use_track_genre:
            return fallback_idx
        candidates_artist_genre_idxs = []
        for candidate in candidates:
            if candidate.payload.get("artist_name", "") == query_payload.get("artist_name", ""):
                continue
            if int(candidate.payload.get("artist_genre_idx") or 0) <= 0:
                continue
            if "," in candidate.payload.get("artist_genres", ""):
                continue
            candidates_artist_genre_idxs.append(candidate.payload.get("artist_genre_idx", 0))
        candidates_artist_genre_idxs = [candidate.payload.get("artist_genre_idx") for candidate in candidates if candidate.payload.get("artist_genre_idx") is not None and "," not in candidate.payload.get("artist_genres", "")]
        if len(set(candidates_artist_genre_idxs)) == 0:
            track_genre_idx = fallback_idx
        elif len(set(candidates_artist_genre_idxs)) == 1:
            track_genre_idx = candidates_artist_genre_idxs[0]
        else:
            genre_score_dict = {idx: 0.0 for idx in set(candidates_artist_genre_idxs)}
            for index, idx in enumerate(candidates_artist_genre_idxs):
                genre_score_dict[idx] = genre_score_dict[idx] + 1 / ((index + 1) ** 0.5)
            genre_score_list = list(genre_score_dict.items())
            genre_score_list = sorted(genre_score_list, key=lambda item: item[1], reverse=True)
            if genre_score_list and genre_score_list[0][1] > genre_score_list[1][1] * 2.0:
                track_genre_idx = genre_score_list[0][0]
            else:
                track_genre_idx = fallback_idx
        if track_genre_idx == 0:
            track_genre_idx = fallback_idx
        return track_genre_idx

    # Search nearest points by query vector
    def search_vector_similar_record(self, query_vector: list, candidate_limit: int, artist_genre_idx: Union[int, list, None] = None):
        result = []
        query_filter = None
        if artist_genre_idx is not None:
            if isinstance(artist_genre_idx, int):
                artist_genre_idx = [artist_genre_idx]
            query_list = []
            for idx in artist_genre_idx:
                if idx != 0 and idx not in query_list:
                    query_list.append(idx)
            if query_list:
                query_filter = qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="artist_genre_idx",
                            match=qdrant_models.MatchAny(any=query_list)
                        )
                    ]
                )
        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=candidate_limit,
                with_payload=True,
                with_vectors=False
            )
        except Exception as e:
            print(f"Failed to run Qdrant query_points: {e}")
            return result
        result = list(response.points)
        result = [r for r in result if r.payload.get("artist_genres", "") != ""]
        return result

    # Search high popularity points by artist genre
    def search_genre_popular_record(self, query_vector: list, candidate_limit: int, artist_genre_idx: Union[int, list, None] = None):
        result = []
        query_filter = None
        if artist_genre_idx is not None:
            if isinstance(artist_genre_idx, int):
                artist_genre_idx = [artist_genre_idx]
            query_list = []
            for idx in artist_genre_idx:
                if idx != 0 and idx not in query_list:
                    query_list.append(idx)
            if query_list:
                query_filter = qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="artist_genre_idx",
                            match=qdrant_models.MatchAny(any=query_list)
                        )
                    ]
                )
        try:
            records, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                order_by=qdrant_models.OrderBy(
                    key="popularity",
                    direction=qdrant_models.Direction.DESC
                ),
                limit=candidate_limit,
                with_payload=True,
                with_vectors=False
            )
        except Exception as e:
            print(f"Failed to run Qdrant scroll: {e}")
            return result
        candidate_ids = [record.id for record in records]
        if candidate_ids and query_vector:
            try:
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    query_filter=qdrant_models.Filter(
                        must=[
                            qdrant_models.HasIdCondition(has_id=candidate_ids)
                        ]
                    ),
                    limit=len(candidate_ids),
                    with_payload=True,
                    with_vectors=False
                )
            except Exception as e:
                print(f"Failed to run Qdrant query_points: {e}")
                return result
            result = list(response.points)
        else:
            result = records
        result = [r for r in result if r.payload.get("artist_genres", "") != ""]
        return result

    # Search same artist points by artist index
    def search_artist_popular_record(self, candidate_limit: int, artist_idx: int):
        result = []
        try:
            records, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="artist_idx",
                            match=qdrant_models.MatchValue(value=artist_idx)
                        )
                    ]
                ),
                order_by=qdrant_models.OrderBy(
                    key="popularity",
                    direction=qdrant_models.Direction.DESC
                ),
                limit=candidate_limit,
                with_payload=True,
                with_vectors=False
            )
        except Exception as e:
            print(f"Failed to run Qdrant scroll: {e}")
            return result
        result = records
        return result

    # Search related artist points by collaborative data and query vector
    def search_related_artist_record(self, query_vector: list, candidate_limit: int, related_artist_idxs: list):
        result = []
        if not related_artist_idxs:
            return result
        query_filter = None
        query_filter = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="artist_idx",
                    match=qdrant_models.MatchAny(any=related_artist_idxs)
                )
            ]
        )
        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=candidate_limit,
                with_payload=True,
                with_vectors=False
            )
        except Exception as e:
            print(f"Failed to run Qdrant query_points: {e}")
            return result
        result = list(response.points)
        result = sorted(result, key=lambda x: x.payload['popularity'], reverse=True)
        return result

    # Search related track points by collaborative data
    def search_related_track_record(self, query_payload: dict, candidate_limit: int):
        result = []
        if self.wv is None:
            return result
        query_track_id = query_payload.get("track_id", "")
        if not query_track_id:
            return result
        if query_track_id not in self.wv:
            return result
        wv_result = self.wv.most_similar(query_track_id, topn=candidate_limit)
        for track_id, score in wv_result:
            if self.min_related_track_score > 0.0 and score < self.min_related_track_score:
                continue
            record = self.find_query_record_by_track(track_id=track_id)
            if record is not None:
                result.append(record)
        return result

    # similar_candidates to final result
    def filter_similar_candidates(self, query_payload: dict, candidates: list, top_k: int = 20, prefer: Literal["default", "same_artist", "popularity"] = "default"):
        result = []
        if not candidates:
            return result
        result_track_names = []
        skip_unk_genre = True
        same_artist_ratio = self.same_artist_ratio_range[0]
        same_artist_counter = 0
        query_track_name = str(query_payload.get("track_name") or "").lower().split(" (")[0].split(" - ")[0].strip()
        query_artist_name = str(query_payload.get("artist_name") or "").lower().strip()
        query_artist_genre_idx = int(query_payload.get("artist_genre_idx") or 0)
        query_popularity = float(query_payload.get("popularity") or 0.0)
        min_popularity = min(query_popularity, self.min_popularity)
        track_genre_idx = self.get_track_genre_idx(query_payload=query_payload, candidates=candidates, fallback_idx=query_artist_genre_idx)
        prev_similarity = 1.0
        similarity_eps = 1e-5
        for candidate in candidates:
            payload = candidate.payload or {}
            payload_track_id = str(payload.get("track_id") or "").strip()
            payload_track_name = str(payload.get("track_name") or "").lower().split(" (")[0].split(" - ")[0].strip()
            payload_artist_name = str(payload.get("artist_name") or "").lower().strip()
            payload_artist_genre_idx = int(payload.get("artist_genre_idx") or 0)
            payload_popularity = float(payload.get("popularity") or 0.0)
            payload_similarity = float(candidate.score)
            if not payload_track_id or not payload_track_name:
                continue
            if payload_track_id == str(query_payload['track_id']):
                continue
            if payload_track_name == query_track_name and payload_artist_name == query_artist_name:
                continue
            if payload_track_name in result_track_names:
                continue
            if f"{query_track_name} " in payload_track_name:
                continue
            if "remix" in payload_track_name:
                continue
            if payload_popularity < min_popularity:
                continue
            if track_genre_idx > 0 and payload_artist_genre_idx != track_genre_idx:
                continue
            if skip_unk_genre and payload_artist_genre_idx == 0:
                continue
            if same_artist_counter >= max(1, int(top_k * same_artist_ratio)) and payload_artist_name == query_artist_name:
                continue
            if abs(payload_similarity - prev_similarity) < similarity_eps:
                if result and payload_popularity > float(result[-1].get("popularity") or 0.0):
                    result.pop(-1)
                else:
                    continue
            if payload_artist_name == query_artist_name:
                if prefer == "same_artist":
                    result.insert(same_artist_counter, payload)
                else:
                    result.append(payload)
                same_artist_counter = same_artist_counter + 1
            else:
                result.append(payload)
            result_track_names.append(payload_track_name)
            prev_similarity = payload_similarity
            if len(result) >= int(top_k):
                break
        if prefer == "popularity":
            result = sorted(result, key=lambda item: float(item['popularity']), reverse=True)
        return result

    # popular_candidates, same_artist_candidates, related_artist_candidates to final result
    def filter_others_candidates(self, query_payload: dict, candidates: list, top_k: int = 20, is_shuffle: bool = False):
        result = []
        if not candidates:
            return result
        result_track_names = []
        query_track_name = str(query_payload.get("track_name") or "").lower().split(" (")[0].split(" - ")[0].strip()
        query_artist_name = str(query_payload.get("artist_name") or "").lower().strip()
        query_album_name = str(query_payload.get("album_name") or "").lower().strip()
        query_popularity = float(query_payload.get("popularity") or 0.0)
        min_popularity = min(query_popularity, self.min_popularity)
        same_artist_counter = 0
        same_album_counter = 0
        if is_shuffle:
            random.shuffle(candidates)
        for candidate in candidates:
            payload = candidate.payload or {}
            payload_track_id = str(payload.get("track_id") or "").strip()
            payload_track_name = str(payload.get("track_name") or "").lower().split(" (")[0].split(" - ")[0].strip()
            payload_artist_name = str(payload.get("artist_name") or "").lower().strip()
            payload_album_name = str(payload.get("album_name") or "").lower().strip()
            payload_popularity = float(payload.get("popularity") or 0.0)
            if not payload_track_id or not payload_track_name:
                continue
            if payload_track_id == str(query_payload['track_id']):
                continue
            if payload_track_name == query_track_name and payload_artist_name == query_artist_name:
                continue
            if payload_track_name in result_track_names:
                continue
            if f"{query_track_name} " in payload_track_name:
                continue
            if "remix" in payload_track_name:
                continue
            if payload_popularity < min_popularity:
                continue
            if payload_album_name == query_album_name and same_album_counter < max(1, int(top_k * self.same_artist_ratio_range[0])):
                result.insert(same_album_counter, payload)
                same_album_counter = same_album_counter + 1
            elif payload_artist_name == query_artist_name:
                result.insert(same_artist_counter + same_album_counter, payload)
                same_artist_counter = same_artist_counter + 1
            else:
                result.append(payload)
            result_track_names.append(payload_track_name)
            if len(result) >= int(top_k):
                break
        return result

    # Remove duplicate items
    def filter_by_isrc(self, result: list):
        if self.collection_version in ["v1", "v2"]:
            return result
        if len(result) <= 1:
            return result
        tracker = {}
        for item in result:
            isrc = item.get("isrc", "")
            if not isrc:
                continue
            current_pop = item.get("popularity", 0.0)
            if isrc not in tracker:
                tracker[isrc] = item
            else:
                prev_pop = tracker[isrc].get("popularity", 0.0)
                if current_pop > prev_pop:
                    tracker[isrc] = item
        result = []
        for isrc in tracker:
            result.append(tracker[isrc])
        return result

    # Avoid same artist gathering
    def shuffle_result_block(self, result: list, column_name: str, max_block_len: int = 1, max_tries: int = 20):
        protect_multi_source = True
        is_shuffled = False
        for _ in range(max_tries):
            same_counter = 1
            prev_name = ""
            if is_shuffled:
                break
            for i in range(len(result)):
                current_name = result[i][column_name]
                if i == 0:
                    prev_name = current_name
                    continue
                sources = result[i].get("sources", []).copy()
                if "same_artist" in sources:
                    sources.remove("same_artist")
                if protect_multi_source and len(sources) > 1:
                    prev_name = current_name
                    continue
                if current_name == prev_name:
                    same_counter = same_counter + 1
                else:
                    prev_name = current_name
                    same_counter = 1
                if i + 1 < len(result) and same_counter > max_block_len:
                    poped_item = result.pop(i)
                    if i + 1 == len(result):
                        insert_position = len(result)
                    else:
                        insert_position = random.randrange(i + 1, len(result))
                    result.insert(insert_position, poped_item)
                    prev_name = result[i][column_name]
                    same_counter = 1
                    break
                is_shuffled = True
        return result

    # Merge 3-way recall result by given ratio
    def merge_result_by_ratio(self, query_payload: dict, similar_result: list, popular_result: list = [], same_artist_result: list = [], top_k: int = 20):
        result = []
        if not similar_result and not popular_result and not same_artist_result:
            return result
        if not popular_result and not same_artist_result:
            result = similar_result[:top_k]
            return result
        max_same_artist_ratio = random.uniform(self.same_artist_ratio_range[0], self.same_artist_ratio_range[-1])
        max_same_artist = max(1, int(top_k * max_same_artist_ratio))
        popular_len = max(1, min(int(top_k * self.popular_ratio), len(popular_result)))
        same_artist_len = max(1, min(int(top_k * self.same_artist_ratio_range[0]), len(same_artist_result)))
        candidates = popular_result[:popular_len] + same_artist_result[:same_artist_len]
        random.shuffle(candidates)
        result = similar_result.copy()
        result_ids = [item['track_id'] for item in result]
        query_artist_idx = query_payload.get("artist_idx", 0)
        query_album_name = query_payload.get("album_name", "")
        same_artist_count = sum(1 for item in result if item.get("artist_idx", 0) == query_artist_idx)
        for candidate in candidates:
            payload_track_id = candidate.get("track_id", "")
            if not payload_track_id:
                continue
            payload_artist_idx = candidate.get("artist_idx")
            if payload_track_id in result_ids:
                continue
            payload_track_name = candidate.get("track_name", "").strip()
            if not payload_track_name or payload_track_name in [item['track_name'] for item in result]:
                continue
            if payload_artist_idx == query_artist_idx:
                if same_artist_count >= max_same_artist:
                    continue
                same_artist_count = same_artist_count + 1
            if query_album_name and candidate.get("album_name", "") == query_album_name:
                insert_position = 0
            else:
                insert_position = random.randrange(min(1, len(result)), len(result))
            result.insert(insert_position, candidate)
            if payload_track_id not in result_ids:
                result_ids.append(payload_track_id)
            if len(result) > top_k:
                popped_item = result.pop(-1)
                result_ids.remove(popped_item['track_id'])
                if popped_item.get("artist_idx", 0) == query_artist_idx:
                    same_artist_count = same_artist_count - 1
        result = result[:top_k]
        return result

    # Merge 5-way recall result by item score
    def merge_result_by_score(self, query_payload: dict, similar_result: list, popular_result: list = [], same_artist_result: list = [], related_artist_result: list = [], related_track_result: list = [], top_k: int = 20):
        result = []
        trackid_result_dict = {}
        def remap(source_value, source_start, source_end, target_start, target_end):
            if source_start == source_end:
                return target_start
            ratio = (source_value - source_start) / (source_end - source_start)
            target_value = target_start + ratio * (target_end - target_start)
            min_val = min(target_start, target_end)
            max_val = max(target_start, target_end)
            target_value = max(min_val, min(target_value, max_val))
            target_value = round(target_value, 3)
            return target_value

        similar_result = self.filter_by_isrc(similar_result)
        popular_result = self.filter_by_isrc(popular_result)
        same_artist_result = self.filter_by_isrc(same_artist_result)
        related_artist_result = self.filter_by_isrc(related_artist_result)
        related_track_result = self.filter_by_isrc(related_track_result)
        if not popular_result and not same_artist_result and not related_artist_result and not related_track_result:
            result = similar_result[:top_k]
            for i in range(len(result)):
                result[i]['sources'] = ["similar"]
            return result
        popular_result = popular_result[:top_k]
        random.shuffle(popular_result)
        source_weight_dict = {
            "similar": self.recall_similar_weights,
            "popular": self.recall_popular_weights,
            "same_artist": self.recall_same_artist_weights,
            "related_artist": self.recall_related_artist_weights,
            "related_track": self.recall_related_track_weights
        }
        source_items_dict = {
            "similar": similar_result,
            "popular": popular_result,
            "same_artist": same_artist_result,
            "related_artist": related_artist_result,
            "related_track": related_track_result
        }
        for source, items in source_items_dict.items():
            for index, item in enumerate(items):
                track_id = item.get("track_id", "")
                artist_idx = item.get("artist_idx", 0)
                if not track_id:
                    continue
                if isinstance(source_weight_dict[source], list):
                    score = remap(source_value=index, source_start=0, source_end=len(items), target_start=source_weight_dict[source][0], target_end=source_weight_dict[source][-1])
                else:
                    score = float(source_weight_dict[source]) - (0.001 * index)
                if track_id in trackid_result_dict:
                    trackid_result_dict[track_id]['score'] = trackid_result_dict[track_id]['score'] + score
                    trackid_result_dict[track_id]['sources'].append(source)
                else:
                    trackid_result_dict[track_id] = {
                        "artist_idx": artist_idx,
                        "sources": [source],
                        "score": score
                    }
        trackid_sorted_keys = sorted(trackid_result_dict.keys(), key=lambda k: trackid_result_dict[k]['score'], reverse=True)
        trackid_result_dict = {trackid: trackid_result_dict[trackid] for trackid in trackid_sorted_keys}
        candidates = similar_result + popular_result + same_artist_result + related_artist_result + related_track_result
        max_same_artist_ratio = random.uniform(self.same_artist_ratio_range[0], self.same_artist_ratio_range[-1])
        max_same_artist = max(1, int(top_k * max_same_artist_ratio))
        max_payload_artist = max(1, int(top_k * self.same_artist_ratio_range[0]))
        query_artist_idx = query_payload.get("artist_idx", 0)
        query_track_name = query_payload.get("query_track_name", "")
        same_artist_count = 0
        popular_len = max(0, min(int(top_k * self.popular_ratio), len(popular_result)))
        popular_count = 0
        result_track_names = []
        for target_track_id in trackid_result_dict:
            if len(result) >= top_k:
                break
            for candidate in candidates:
                payload_track_id = candidate.get("track_id", "")
                payload_track_name = candidate.get("track_name", "").lower().split(" (")[0].split(" - ")[0].strip()
                payload_artist_idx = candidate.get("artist_idx", 0)
                if not payload_track_id:
                    continue
                if payload_track_id == target_track_id:
                    if payload_artist_idx == query_artist_idx:
                        if same_artist_count >= max_same_artist:
                            continue
                        same_artist_count = same_artist_count + 1
                    else:
                        payload_artist_count = sum([1 for item in result if item['artist_idx'] == payload_artist_idx])
                        if payload_artist_count >= max_payload_artist:
                            continue
                    if "popular" in trackid_result_dict[payload_track_id]['sources']:
                        popular_count = popular_count + 1
                    if payload_track_name in result_track_names or payload_track_name in query_track_name:
                        continue
                    result.append(candidate)
                    result_track_names.append(payload_track_name)
                    break
        if popular_count < popular_len:
            selected_candidates = []
            for target_track_id in trackid_result_dict:
                if popular_count >= popular_len:
                    break
                if target_track_id in [item['track_id'] for item in result]:
                    continue
                if "popular" not in trackid_result_dict[target_track_id]['sources']:
                    continue
                for candidate in popular_result:
                    if candidate.get("track_id", "") == target_track_id:
                        selected_candidates.append(candidate)
                        popular_count = popular_count + 1
                        break
            for _ in range(len(selected_candidates)):
                if result:
                    result.pop(-1)
            result.extend(selected_candidates)
        result = self.filter_by_isrc(result)
        result = result[:top_k]
        if len(result) < top_k:
            for candidate in candidates:
                if candidate not in result:
                    result.append(candidate)
            result = self.filter_by_isrc(result)
            result = result[:top_k]
        max_score = min(max([item['score'] for item in trackid_result_dict.values()]), 2.0)
        score_scaling_ratio = 100 / max_score if max_score > 0 else 0.0
        for i in range(len(result)):
            result[i]['sources'] = trackid_result_dict.get(result[i]['track_id'], {}).get("sources", [])
            score_int = int(min(trackid_result_dict.get(result[i]['track_id'], {}).get("score", 0.0), 2.0) * score_scaling_ratio)
            result[i]['score'] = round(score_int / 100, 2)
        result = self.shuffle_result_block(result=result, column_name="album_name", max_block_len=1, max_tries=top_k)
        result = self.shuffle_result_block(result=result, column_name="artist_name", max_block_len=2, max_tries=top_k)
        return result

    # 从不同声音特征中随机抽取一首歌曲作为新种子
    def get_random_track_diff_sound(self, track_id: str):
        """
        基于声音特征向量，从曲库中随机返回一首与原歌曲声学特征差异大的歌曲。
        不依赖风格标签，纯粹基于 embedding 向量距离。
        """
        import random
        import numpy as np
        
        # 1. 找到原歌曲及其向量
        query_record = self.find_query_record_by_track(track_id=track_id)
        if query_record is None:
            print("get_random_track_diff_sound: Track not found.")
            return None

        query_payload = query_record.payload or {}
        query_vector = query_record.vector or None
        
        # 如果记录中没有向量，重新查询获取
        if query_vector is None:
            try:
                rec = self.client.retrieve(
                    collection_name=self.collection_name,
                    ids=[query_record.id],
                    with_vectors=True,
                    with_payload=True
                )
                if rec and len(rec) > 0:
                    query_vector = rec[0].vector
                    query_payload = rec[0].payload or {}
            except Exception as e:
                print(f"get_random_track_diff_sound: Failed to retrieve vector: {e}")
                return None
        
        if query_vector is None:
            print("get_random_track_diff_sound: No vector available.")
            return None
        
        # 2. 使用反向向量（负向量）搜索距离最远的歌曲
        # 在余弦相似度下，-query_vector 会返回与原歌曲最不相似的歌曲
        neg_vector = [-v for v in query_vector]
        
        try:
            # 使用负向量查询，获取距离最远的 Top N 候选
            far_tracks = self.client.query_points(
                collection_name=self.collection_name,
                query=neg_vector,
                limit=100,
                with_payload=True,
                with_vectors=False
            )
            candidates = list(far_tracks.points)
            
            if self.verbose_log:
                print(f"get_random_track_diff_sound: Found {len(candidates)} far tracks via negative vector")
            
            # 从距离最远的前 100 首中随机选一首
            if candidates:
                selected = random.choice(candidates[:50])  # 从前 50 首最远的中随机选
                selected_payload = selected.payload or {}
                
                if self.verbose_log:
                    print(f"get_random_track_diff_sound: Selected '{selected_payload.get('track_name')}' (score={selected.score:.3f})")
                
                payload = selected_payload
                payload['track_id'] = str(payload.get('track_id') or '')
                payload['track_name'] = str(payload.get('track_name') or '')
                payload['artist_name'] = str(payload.get('artist_name') or '')
                payload['artist_genre_idx'] = int(payload.get('artist_genre_idx') or 0)
                return payload
                
        except Exception as e:
            print(f"get_random_track_diff_sound: Failed to query with negative vector: {e}")
        
        # 3. 回退策略：随机从曲库中选一首（排除原歌曲）
        if self.verbose_log:
            print("get_random_track_diff_sound: Fallback to random scroll")
        
        try:
            records, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=qdrant_models.Filter(
                    must_not=[
                        qdrant_models.FieldCondition(
                            key="track_id",
                            match=qdrant_models.MatchValue(value=str(query_payload.get('track_id', '')))
                        )
                    ]
                ),
                limit=200,
                with_payload=True,
                with_vectors=False
            )
            if records:
                selected = random.choice(list(records))
                selected_payload = selected.payload or {}
                payload = selected_payload
                payload['track_id'] = str(payload.get('track_id') or '')
                payload['track_name'] = str(payload.get('track_name') or '')
                payload['artist_name'] = str(payload.get('artist_name') or '')
                payload['artist_genre_idx'] = int(payload.get('artist_genre_idx') or 0)
                return payload
        except Exception as e:
            print(f"get_random_track_diff_sound: Fallback scroll failed: {e}")
        
        return None

    # Main method
    def search_entry(self, track_id: str = "", track_name: str = "", artist_name: str = "", artist_idx: int = 0, top_k: int = 20, add_query_track: bool = False, exclude_style: bool = False):
        result = []
        artist_idx = max(0, artist_idx)
        if (not track_id) and (not track_name and not artist_name) and (not artist_idx) and (not artist_name):
            print("Must provide one: 1. track_id; 2. track_name & artist_name; 3. artist_idx; 4. artist_name.")
            return result
        track_id = track_id.strip()
        track_name = track_name.strip()
        artist_name = artist_name.strip()
        if track_name and self.is_zhconv and "zhconv" not in DISABLED_PACKAGES:
            try:
                track_name = convert(track_name, locale="zh-hk")
                artist_name = convert(artist_name, locale="zh-hk")
            except Exception as e:
                print(f"Failed to convert `track_name` and `artist_name` to Traditional Chinese: {e}")
        t1 = time.time()
        artist_idx = int(artist_idx)
        query_record = None
        if track_id:
            query_record = self.find_query_record_by_track(track_id=track_id)
        elif track_name and artist_name:
            if not self.enable_name_search and self.verbose_log:
                print("WARNING: please set `enable_name_search` to True and rebuild the Qdrant index for faster query.")
            query_record = self.find_query_record_by_track(track_name=track_name, artist_name=artist_name)
        if query_record is None:
            if artist_idx:
                query_record = self.find_query_record_by_artist(artist_idx=artist_idx)
            elif artist_name:
                if not self.enable_name_search and self.verbose_log:
                    print("WARNING: please set `enable_name_search` to True and rebuild the Qdrant index for faster query.")
                query_record = self.find_query_record_by_artist(artist_name=artist_name)
        t2 = time.time()
        if query_record is None:
            print("search_entry: Track not found.")
            return result
        query_vector = query_record.vector
        if query_vector is None:
            print("search_entry: Vector of query track is missing.")
            return result
        top_k = max(1, top_k)
        query_payload = query_record.payload or {}
        query_artist_genres = str(query_payload.get("artist_genres") or "")
        query_artist_idx = int(query_payload.get("artist_idx") or 0)
        query_artist_name = str(query_payload.get("artist_name") or "")
        query_related_artist_idxs = list(query_payload.get("related_artist_idxs") or [])
        query_artist_genre_idx = self.get_artist_genre_idx(query_payload=query_payload)
        if query_artist_genre_idx == 0:
            query_artist_genre_idx = int(self.artist_genre_idx_patch.get(query_artist_idx, 0))
        if query_artist_genre_idx == 0 and "embeat_utils" not in DISABLED_PACKAGES:
            query_artist_genre_idx = search_everynoise(artist_name=query_artist_name)
        if "," in query_artist_genres or query_artist_genre_idx == 0:
            artist_genre_idxs = self.get_artist_genre_idxs(artist_genres=query_artist_genres)
        else:
            artist_genre_idxs = [query_artist_genre_idx]
        if self.verbose_log:
            if track_name or artist_name:
                print(f"Input track info: {track_name} - {artist_name}")
            print(f"Database URL: {self.qdrant_url}")
            print(f"Query track_id: {query_payload['track_id']}")
            print(f"Query track info: {query_payload['track_name']} - {query_payload['artist_name']}")
            print(f"Query artist genres: {[(self.index_genre_dict.get(idx) or '<unk>') for idx in artist_genre_idxs]}")
        if self.verbose_log:
            print(f"-> Find query record used time: {int((t2 - t1) * 1000)}ms")
        t1 = time.time()
        candidate_limit = max(1, min(int(top_k * 10), 512))
        exclude_genre_idx = query_artist_genre_idx if exclude_style and query_artist_genre_idx > 0 else 0
        candidates = self.search_vector_similar_record(query_vector=query_vector, candidate_limit=candidate_limit, artist_genre_idx=artist_genre_idxs)
        # 后置过滤：排除同风格候选（但保留声学相似度）
        if exclude_genre_idx > 0:
            candidates = [c for c in candidates if int(c.payload.get("artist_genre_idx", 0)) != exclude_genre_idx]
        result = self.filter_similar_candidates(query_payload=query_payload, candidates=candidates, top_k=top_k)
        t2 = time.time()
        if self.verbose_log:
            print(f"-> Similar recall used time: {int((t2 - t1) * 1000)}ms")
        t1 = time.time()
        popular_result = []
        if query_artist_genre_idx > 0:
            candidate_limit = max(1, min(int(top_k * 2), 128))
            track_genre_idx = self.get_track_genre_idx(query_payload=query_payload, candidates=candidates, fallback_idx=query_artist_genre_idx)
            popular_candidates = self.search_genre_popular_record(query_vector=query_vector, candidate_limit=candidate_limit, artist_genre_idx=track_genre_idx)
            # 后置过滤热门候选中的同风格
            if exclude_genre_idx > 0:
                popular_candidates = [c for c in popular_candidates if int(c.payload.get("artist_genre_idx", 0)) != exclude_genre_idx]
            popular_result = self.filter_others_candidates(query_payload=query_payload, candidates=popular_candidates, top_k=top_k)
        t2 = time.time()
        if self.verbose_log:
            print(f"-> Popular recall used time: {int((t2 - t1) * 1000)}ms")
        t1 = time.time()
        same_artist_result = []
        if query_artist_idx > 0:
            candidate_limit = max(1, min(int(top_k * 2), 128))
            same_artist_candidates = self.search_artist_popular_record(candidate_limit=candidate_limit, artist_idx=query_artist_idx)
            same_artist_result = self.filter_others_candidates(query_payload=query_payload, candidates=same_artist_candidates, top_k=top_k, is_shuffle=True)
        t2 = time.time()
        if self.verbose_log:
            print(f"-> Same artist recall used time: {int((t2 - t1) * 1000)}ms")
        t1 = time.time()
        related_artist_result = []
        if query_artist_idx > 0:
            candidate_limit = max(1, min(int(top_k * 2), 128))
            if not query_related_artist_idxs:
                query_related_artist_idxs = self.related_artist_idx.get(query_artist_idx) or []
            query_related_artist_idxs = [idx for idx in query_related_artist_idxs if idx != query_artist_idx and idx > 0]
            if len(query_related_artist_idxs) >= 10:
                query_related_artist_idxs = query_related_artist_idxs[:5]
            related_artist_candidates = self.search_related_artist_record(query_vector=query_vector, candidate_limit=candidate_limit, related_artist_idxs=query_related_artist_idxs)
            related_artist_result = self.filter_others_candidates(query_payload=query_payload, candidates=related_artist_candidates, top_k=top_k)
        t2 = time.time()
        if self.verbose_log:
            print(f"-> Related artist recall used time: {int((t2 - t1) * 1000)}ms")
        t1 = time.time()
        related_track_result = []
        if self.wv is not None:
            candidate_limit = max(1, min(int(top_k * 2), 128))
            related_track_candidates = self.search_related_track_record(query_payload=query_payload, candidate_limit=candidate_limit)
            related_track_result = self.filter_others_candidates(query_payload=query_payload, candidates=related_track_candidates, top_k=top_k)
        t2 = time.time()
        if self.verbose_log:
            print(f"-> Related track recall used time: {int((t2 - t1) * 1000)}ms")
        t1 = time.time()
        result = self.merge_result_by_score(query_payload=query_payload, similar_result=result, popular_result=popular_result, same_artist_result=same_artist_result, related_artist_result=related_artist_result, related_track_result=related_track_result, top_k=top_k)
        if add_query_track:
            if "score" not in query_payload:
                query_payload['score'] = 1.0
            result.insert(0, query_payload)
            result = result[:top_k]
        t2 = time.time()
        if self.verbose_log:
            print(f"-> Re-ranking used time: {int((t2 - t1) * 1000) + 1}ms")
        return result

    # Pack result to GDMusic API format
    def pack_to_search_result(self, embeat_result: list, is_random_order: bool = False):
        result = []
        if not embeat_result:
            return result
        for item in embeat_result:
            result.append({
                "id": str(item.get("track_id", "")),
                "name": str(item.get("track_name", "")),
                "artist": [str(item.get("artist_name", ""))],
                "album": str(item.get("album_name", "")),
                "pic_id": str(item.get("track_id", "")),
                "url_id": str(item.get("track_id", "")),
                "lyric_id": str(item.get("track_id", "")),
                "source": "spotify"
            })
            if self.is_output_extra:
                result[-1]['extra_data'] = {
                    "score": float(item.get("score", 0.0)),
                    "isrc": str(item.get("isrc", "")),
                    "popularity": float(item.get("popularity", 0.0)),
                    "artist_genres": str(item.get("artist_genres", ""))
                }
        if is_random_order:
            random.shuffle(result)
        return result

    # Print final result
    def print_result(self, result: list[dict[str, Any]]):
        result_genre_idxs = list(set([item['artist_genre_idx'] for item in result]))
        print(f"Result artist genres: {[(self.index_genre_dict.get(idx) or '<unk>') for idx in result_genre_idxs]}")
        print(f"======= Top {len(result)} items =======")
        print("index\ttrack_id\t\ttrack_name\tartist_name\talbum_name\tsources\t\tscore")
        for i, item in enumerate(result):
            print(f"{i + 1} \t{item['track_id']} \t{item['track_name']} \t{item['artist_name']} \t{item['album_name']} \t{item.get('sources', [])} \t{item.get('score', 0.0)}")
        print("")
        return


# Command line entry
def main():
    ed = EmbeatDatabase()
    if len(sys.argv) <= 1:
        sys.exit(0)
    command = sys.argv[1]
    if command in ["-t", "--track"]:
        if len(sys.argv) < 3:
            print("ERROR: track_id is required.")
            sys.exit(1)
        track_id = sys.argv[2]
        t1 = time.time()
        result = ed.search_entry(track_id=track_id)
        t2 = time.time()
        ed.print_result(result)
        print(f"Query used time: {t2 - t1:.3f}s")
    elif command in ["-a", "--artist"]:
        if len(sys.argv) < 3:
            print("ERROR: artist_name is required.")
            sys.exit(1)
        artist_name = sys.argv[2]
        t1 = time.time()
        result = ed.search_entry(artist_name=artist_name)
        t2 = time.time()
        ed.print_result(result)
        print(f"Query used time: {t2 - t1:.3f}s")
    elif command in ["-s", "--search"]:
        if len(sys.argv) < 3:
            print('ERROR: "<track_name> - <artist_name>" is required.')
            sys.exit(1)
        track_info = sys.argv[2].strip()
        if " - " not in track_info:
            print('ERROR: use this format for search `"<track_name> - <artist_name>"`.')
            sys.exit(1)
        track_name = track_info.split(" - ")[0]
        artist_name = track_info.split(" - ")[-1]
        t1 = time.time()
        result = ed.search_entry(track_name=track_name, artist_name=artist_name)
        t2 = time.time()
        ed.print_result(result)
        print(f"Query used time: {t2 - t1:.3f}s")
    elif command in ["-bi", "--build-index"]:
        t1 = time.time()
        ed.build_qdrant_index()
        t2 = time.time()
        time.sleep(2)
        ed.build_qdrant_client(is_timeout=True)
        print(f"Build index used time: {t2 - t1:.3f}s")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
