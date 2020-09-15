# 2020.09.15 according to DLAC, this file is made to build a dataloader to accelearte Dataloader of Pytorch

import itertools
import os
import sys
import time
import random
import torch
import queue
import traceback
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from multiprocessing import Process, Queue, Lock

from parameters import SysPara as para
from utils import PipeInput


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__(), max_prefetch=1)


