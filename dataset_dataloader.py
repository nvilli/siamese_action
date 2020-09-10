# 2020.09.10 use HMDB51 as baseline dataset to train model(alex_siamese & vgg16_siamese)

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