import random
from typing import Optional
import numpy as np
import torch
import os
import logging

import library.train_util as train_util

logger = logging.getLogger(__name__)

def addift_timesteps(min_t: int, max_t: int, num_segments: int, training_step: int, b_size: int = 1) -> torch.Tensor:
    """
    ADDifT用のtimestepsを生成する関数。
    min_tとmax_tの間をnum_segmentsで分割し、training_stepに応じてその中からランダムにtimestepsを生成する。
    """
    segment_size = (max_t - min_t) // num_segments
    current_segment = training_step % num_segments
    start = min_t + (current_segment * segment_size)
    end = min(start + segment_size, max_t)
    if training_step is None:
        current_segment = (current_segment + 1) % num_segments
    timesteps = torch.randint(start, end, (b_size,), device="cpu")  # updated to use instance variables
    return timesteps

class ADDifTBucketManager(train_util.BucketManager):
    def __init__(self, no_upscale, max_reso, min_size, max_size, reso_steps):
        super().__init__(no_upscale, max_reso, min_size, max_size, reso_steps)
        self.filename_base_to_reso = {}
        self.filename_base_to_resized_size = {}
        self.original_buckets: list[list[str | train_util.ImageInfo]] = []

    def select_bucket(self, image_width, image_height, image_info: Optional[train_util.ImageInfo] = None):
        "*_targetと*が同じresoに配置されるようにオーバーライドして挙動を変更している"
        reso, resized_size, ar_error = super().select_bucket(image_width, image_height, image_info)
        if image_info is not None:
            filename = self._get_filename(image_info).rstrip("_target")
            if filename in self.filename_base_to_reso:
                if reso != self.filename_base_to_reso[filename]:
                    logger.warning(f"{image_info.image_key}: bucket resolution mismatch ({reso} != {self.filename_base_to_reso[filename]}). Using {self.filename_base_to_reso[filename]}")
                    reso = self.filename_base_to_reso[filename]
                    resized_size = self.filename_base_to_resized_size[filename]
                    ar_error = (reso[0] / reso[1]) - (image_width / image_height)
            else:
                self.filename_base_to_reso[filename] = reso
                self.filename_base_to_resized_size[filename] = resized_size
        return reso, resized_size, ar_error

    @staticmethod
    def _get_filename(image_or_info: str | train_util.ImageInfo) -> str:
        if isinstance(image_or_info, str):
            return os.path.splitext(os.path.basename(image_or_info))[0]
        elif isinstance(image_or_info, train_util.ImageInfo):
            return os.path.splitext(os.path.basename(image_or_info.image_key))[0]
        raise ValueError(f"Unsupported type: {type(image_or_info)}")

    def shuffle(self):
        """
        それぞれのbucketがdata, target, data, target...となるように並び替える。
        strの場合それをファイルパスとして解釈し、ファイル名が_targetであればtargetとして扱う。そうでなければdataとして扱う。
        ImageInfoの場合はimage_or_info.image_keyがパスなので、同様に_targetであればtargetとして扱う。
        """
        for reso in self.resos:
            bucket_id = self.reso_to_id[reso]
            bucket = self.buckets[bucket_id].copy()
            random.shuffle(bucket)  # 同じ名前のものがランダムに並ぶようにする
            bucket.sort(key=lambda x: self._get_filename(x).removesuffix("_target"))
            datas: list[tuple[str, str|train_util.ImageInfo]] = []
            targets: list[tuple[str, str|train_util.ImageInfo]] = []
            for image_or_info in bucket:
                filename = self._get_filename(image_or_info)
                if filename.endswith("_target"):
                    targets.append((filename, image_or_info))
                else:
                    datas.append((filename, image_or_info))

            assert len(datas) == len(targets), f"Data and target count mismatch in bucket {reso}"
            pairs: list[tuple[str|train_util.ImageInfo, str|train_util.ImageInfo]] = []
            for (data_filename, data), (target_filename, target) in zip(datas, targets):
                assert target_filename.startswith(data_filename), f"Data and target filename mismatch in bucket {reso}"
                pairs.append((data, target))
            random.shuffle(pairs)
            self.buckets[bucket_id] = [item for pair in pairs for item in pair]

def split_batch_data_target(batch: dict) -> tuple[dict, dict]:
    """
    バッチをdataとtargetに分割する関数。
    input_ids_list, text_encoder_outputs_listはそれぞれの要素をdataとtargetに分割する。
    それ以外はそのままdataとtargetに分割する。
    """
    data = {}
    target = {}
    KEYS = ["custom_attributes", "loss_weights", "text_encoder_outputs_list", "input_ids_list", "alpha_masks", "images", "latents", "captions", "original_sizes_hw", "crop_top_lefts", "target_sizes_hw", "flippeds", "network_multipliers", "image_keys"]
    for key, value in batch.items():
        if key in KEYS and isinstance(value, (list, torch.Tensor, np.ndarray, tuple)):
            if key == "input_ids_list" or key =="text_encoder_outputs_list":
                data[key] = [c[0:-1:2] for c in value]
                target[key] = [c[1::2] for c in value]
            else:
                data[key] = value[0:-1:2]
                target[key] = value[1::2]
        else:
            data[key] = value
            target[key] = value
    return data, target
