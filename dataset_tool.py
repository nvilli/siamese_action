# 2020.09.10 use HMDB51 as baseline dataset to train model(alex_siamese & vgg16_siamese)
# 2020.09.12 update
# 2020.09.15 update dataset function

import os
import sys
import time
import random
import json
import cv2
import traceback
import numpy as np
import pickle as pickle
from PIL import Image
import torch.utils.data as data
from transform import *
from parameters import SysPara as para
from parameters import SysPara

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def seq_path(self):
        return self._data[0]

    @property
    def seq_len(self):
        return int(self._data[1])

    @property
    def seq_id_label(self):
        return int(self._data[2])

class Sia_Rpe_Dataset(data.Dataset):

    def __init__(self, video_path=None,
                dataset_type=None,
                meta_file_name=None,
                sample_method=None,
                seg_num_ext=None,
                mode=None,
                transform=None,
                modality=None,
                img_format=None):
        self.video_path = video_path
        self.dataset_type = dataset_type
        self.meta_file_name = meta_file_name
        self.sample_method = sample_method
        self.seg_num_ext = seg_num_ext
        self.mode = mode
        self.transform = transform
        self.modality = modality
        self.img_format = img_format

        self.item_list = None
        self._param_check()
        self._parse_meta_file()

    def __getitem__(self, index):

        video_item = self.item_list
        if video_item.seq_len < 1:
            video_item = self.item_list[index+1]
        img_path = os.path.join(self.video_path, video_item.seq_path)

        if self.sample_method == 'seg_random':
            _index = self._seg_random_sampler(video_item.seq_len, self.seg_num_ext, 0)

        if self.sample_method == 'seg_ratio':
            _index = self._seg_ratio_sampler(video_item.seq_len, self.seg_num_ext, 0, 0.5)

        data = []
        for i in range(self.seg_num_ext):
            data.extend(self._load_image(img_path, _index[i]))
        
        data = self.transform(data)

        return index, data, video_item, seq_id_label

    def __len__(self):
        return len(self.item_list)

    def get(self, index):
        return self.__getitem__(index)
    
    def length(self):
        return self.__len__()

    def count_video(self):
        return self.__len__()

    def _load_image(self, img_path, idx):

        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            return [Image.open(os.path.join(img_path, self.img_format.format(idx))).convert('RGB')]
        elif self.modality == 'FLow':
            x_img = Image.open(os.path.join(img_path, self.img_format.format('x', idx))).convert('L')
            y_img = Image.open(os.path.join(img_path, self.img_format.format('y', idx))).convert('L')
            return [x_img, y_img]

    
    def _paras_check(self):
        if self.video_path is None:
            raise ValueError("video path is None")
        if not (self.dataset_type in ['video', 'image']):
            raise ValueError("dataset type occurs error, dataset type should be [video, image]")
        if not (self.sample_method in ['seg_random', 'seg_ratio', 'seg_seg']):
            raise ValueError("just support sample method in [seg_random, seg_ratio, seg_seg]")
        if not (self.mode in ['train', 'val', 'test']):
            raise ValueError("just support mode in train, val, test")
        if self.seg_num_ext is None:
            raise ValueError("seg_num_ext is NOne")
        (_, ext) = os.path.splitext(self.meta_file_name)
        if not (ext in ['.pkl', '.csv'])
        raise ValueError("meta_file_name is not .pkl, .csv")

    def _seg_ratio_sampler(self, seg_len. seg_num. s. ratio):
        if seq_len < seg_num:
            index = []
            cycle_len = seg_num // seq_len
            for i in range(cycle_len):
                index.extend([s + i for i in range(seq_len)])
            tail_index = [s + i for i in range(seg_num - len(index))]
            index.extend(tail_index)
        if not (ratio >= 0.0 and ratio <= 1.0):
            raise ValueError("ratio is not >= 0 and <= 1", ratio)
        seg_len = seq_len // seg_num
        index = [s + i * seg_len + int((seg_len - 1) * ratio) for i in range(seg_num)]
        return index

    def _seg_seg_sampler(self, seq_len, seg_num_ext, seg_num_inner, func=None, sampler_type=None, ratio=0.5):
        if seq_len < seg_num_ext:
            raise ValueError("seq_len < seg_num", seq_len, seg_num_ext)
        seg_len_ext = seq_len //seg_num_ext

        if seg_len_ext < seg_num_inner:
            raise ValueError("seg_len_ext < seg_num_inner", seg_len_ext, seg_num_inner)
        if func is None:
            raise ValueError("no define single sampler, func is None")
        if not (sampler_type in ['rand', 'ratio']):
            raise ValueError('no define sampler_typr or sampler_type is error')
        index = []
        for i in range(seg_num_ext):
            s = i * seg_len_ext
            if sampler_type == 'rand':
                index += func(seg_len_ext, seg_num_inner, s)
            if sampler_type == 'ratio':
                index += func(seg_len_ext, seg_num_inner, s, ratio)

        return index

    def _dense_random_sampler(self, seq_len, frames, clip_len, s):
        if seq_len < frames:
            index = [i for i in range(seq_len)] + [seq_len - 1 for i in range(frames - seq_len)]
            return [s + i for i in index]
        if frames <= seq_len <= clip_len:
            index = [i for i in range(seq_len)]
            return [s + i for i in index][::(seq_len // frames)][:frames]
        s = random.randint(0, (seq_len - clip_len) - 1)
        index = [i for i in range(clip_len)]
        return [s + i for i in index][::(clip_len // frames)][:frames]
    
    def _dense_uniform_sampler(self, seq_len, frames, clip_len, s):
        if seq_len < frames:
            index = [i for i in range(seq_len)] + [seq_len - 1 for i in range(frames - seq_len)]
            return [s + i for i in index]
        if frames <= seq_len <= clip_len:
            index = [i for i in range(seq_len)]
            return [s + i for i in index][::(seq_len // frames)][:frames]
        s = ( seq_len - clip_len) // 2
        index = [i for i in range(clip_len)]
        return [s + i for i in index][::(clip_len // frames)][:frames]