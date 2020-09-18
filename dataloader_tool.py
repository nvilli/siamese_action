# 2020.09.15 according to DLAC, this file is made to build a dataloader to accelearte Dataloader of Pytorch
# DataLoaderX: using prefetch_generator to pre-fetch batch
# SuperDataLoader: make a cycled-iterator to get data mini-batch


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


# DataLoaderX
# acclerate data loader of pytorch
# use prefetch_generator to pre-fetch batch to accelarate pytorch data loader
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__(), max_prefetch=1)





# SuperDataLoader
# make a cycled-iterator to get data mini-batch

# make a cycled-iterator for SuperDataLoader
def _spl_cycle(iterable):
    while True:
        for x in iterable:
            yield x

# set a random seed for SuperDataLoader
def _spl_seed(seed_id=42):
    torch.manual_seed(seed_id)

# a multi-processing function for SuperDataLoader
def _spl_worker_loader(loader, lock, buffer_queue, worker_id=None):
    _spl_seed(worker_id)
    try:
        data_input = iter(_spl_cycle(loader))
    except Exception:
        exc_type, exc_value, exc_traceback_obj = sys.exc_info()
        traceback.print_tb(exc_traceback_obj)
    while True:
        try:
            index, data, label = next(data_input)
        except Exception:
            print('Buffer[{}] Error occured while getting data from loader!'.format(worker_id))
            exc_type, exc_value, exc_traceback_obj = sys.exc_info()
            traceback.print_tb(exc_traceback_obj)
            continue
        else:
            pass
        while True:
            try:
                buffer_queue.put((index, data, label), block-True, timeout=1)
            except queue.Full:
                continue
            except Exception:
                print("Buffer [{}] Error occured while putting data into buffer queue!".format(worker_id))
                exc_type, exc_value, exc_traceback_obj = sys.exc_info()
                break
            else:
                break

# main part of SuperDataLoader
class SuperDataLoader(object):
    # init SuperDataLoader itself
    def __init__(self, loader, num_workers_reading_buffer, num_workers=1):
        self.loader = loader
        self.num_workers_reading_buffer = num_workers_reading_buffer
        self.shutdown = False
        self.lock = Lock()
        self.num_workers = num_workers
        self._start()

    # inherit
    def _start(self):
        if self.num_workers_reading_buffer > 0:
            self._reading_process()

    def _reading_process(self):
        self.training_reading_buffer = Queue(maxsize=self.num_workers_reading_buffer)
        self.readers = []
        for i in range(self.num_workers_reading_buffer):
            w = Process(
                target=_spl_worker_loader,
                args=(
                    self.loader
                    self.lock
                    self.training_reading_buffer,
                    i,
                )
            )
            w.daemon = True
            w.start()
            self.readers.append(w)

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        return self

    def _shutdown_workers(self):
        if not self.shutdown:
            self.shutdown = True
            if self.num_workers_reading_buffer > 0:
                for w in self.readers:
                    w.terminate()

    def __del__(self):
        if self.num_workers > 0:
            self._shutdown_workers()

    def __next__(self):
        while True:
            try:
                batch = self.training_reading_buffer.get(block=True, timeout=1)
            except queue.Empty:
                continue
            except Exception:
                print(sys.exc_info())
                continue
            else:
                return batch