# this file is made for prepare dataset and dataloader
# 2020.09.19 update, add dataset and dataloader

import sys
import torch
import time
from torch.utils.data import DataLoader
from multiprocessing.managers import BaseManager
import torchvision.transforms as torch_transform

from dataset_tool import Sia_Rpe_Dataset
from dataloader_tool import DataLoaderX, SuperDataLoader
from parameters import SysPara as para
from utils import GenerateDPFlowAddr, AverageMeter
from transform import *

def train_provider_remote(transform=None):
    if not (para.DATASET.LOAD_METHOD in para.DATASET.LOAD_METHOD_LIST):
        print("Load Method %s don't exist, please check it"%(para.DATASET.LOAD_METHOD))
        sys.exit(0)
    if para.DATASET.LOAD_METHOD == 'Sia_Rpe_Dataset':
        BaseManager.register('Sia_Rpe_Dataset', Sia_Rpe_Dataset)
        manager = BaseManager()
        manager.start()
        inst = manager.Sia_Rpe_Dataset(video_path=para.DATASET.VIDEO_SEQ_PATH,
                                        dataset_type='video',
                                        meta_file_name=para.DATASET.TRAIN_META_PATH,
                                        sample_method=para.DATASET.TRAIN_SAMPLE_METHOD,
                                        seg_num_ext=para.TRAIN.SEG_NUM,
                                        mode='train',
                                        transform=transform,
                                        modality=para.TRAIN.MODALITY,
                                        img_format=para.DATASET.IMG_FORMAT)
        return inst
    else:
        print("invalid load method, [{}].support load method has: {}".format(para.DATASET.LOAD_METHOD, para.DATASET.LOAD_METHOD_LIST))
        sys.exit(0)

def train_base(transform=None):
    if not (para.DATASET.LOAD in para.DATASET.LOAD_METHOD_LIST):
        print("load method %s do not exit, please check it" % (para.DATASET.LOAD_METHOD))
        sys.exit(0)
    
    if para.DATASET.LOAD_METHOD == 'Sia_Rpe_Dataset':
        dataset = Sia_Rpe_Dataset(video_path=para.DATASET.VIDEO_SEQ_PATH,
                                    dataset_type='video',
                                    meta_file_name=para.DATASET.TRAIN_META_PATH,
                                    sample_method=para.DATASET.TRAIN_SAMPLE_METHOD,
                                    seg_num_ext=para.TRAIN.SEG_NUM,
                                    mode='train',
                                    transform=transform,
                                    modality=para.TRAIN_MODELITY,
                                    img_format=para.DATASET.IMG_FORMAT)
        return dataset
    else:
        print("invalid load method, [{}].support load method has: {}".format(para.DATASET.LOAD_METHOD, para.DATASET.LOAD_METHOD_LIST))

def val_base(transform=None):
    if not (para.DATASET.LOAD_METHOD in para.DATASET.LOAD_METHOD_LIST):
        print("=> Load Method %s don't exist!!!,Please checking it." % (para.DATASET.LOAD_METHOD))
        sys.exit(0)

    if para.DATASET.LOAD_METHOD == 'Sia_Rpe_Dataset':
        dataset = Sia_Rpe_DataSet(video_path=para.DATASET.VIDEO_SEQ_PATH,
                             dataset_type='video',
                             meta_file_name=para.DATASET.VAL_META_PATH,
                             sample_method=para.DATASET.VAL_SAMPLE_METHOD,
                             seg_num_ext=para.VAL.SEG_NUM,
                             mode='val',
                             transform=transform,
                             modality=para.VAL.MODALITY,
                             img_format=para.DATASET.IMG_FORMART)
        return dataset
    else:
        print("invalid load method, [{}].suppprt load method has:{}".format(para.DATASET.LOAD_METHOD, para.DATASET.LOAD_METHOD_LIST))


def train_loader_local(transform=None):
    dataset = train_base(transform=transform)
    loader = DataLoaderX(dataset,
                        batch_size=para.TRAIN.BATCH_SIZE,
                        shuffle=True,
                        num_workers=8,
                        pin_memory=False)
    if para.SUPER_LOADER:
        print("using superloader")
        loader = SuperDataLoader(loader, 4, num_workers=1)
    
    return loader, dataset.count_video()

def val_loader_local(transform=None):
    dataset = val_base(transform=transform)

    loader = DataLoaderX(dataset,
                        batch_size=para.VAL_BATCH_SIZE,
                        shuffle=False,
                        num_workers=8,
                        pin_memory=False)
    return loader, dataset.count_video()


if __name__=='__main__':
    crop_size = 224
    backbone = 'VGG16'
    input_mean = [0.485, 0.456, 0.406]
    input_std = [0.229, 0.224, 0.225]
    transform = torch_transform.Compose([
        GroupMultiScaleCrop(crop_size, [1, .875, .75, .66]),
        GroupRandomHorizontalFlip(is_flow=False),
        Stack(roll=(backbone == 'VGG16')),
        ToTorchFormatTensor(div=(backbone != 'VGG16')),
        GroupNormalize(input_mean, input_std)
    ])

    loader,_ = train_loader_local(transform=transform)

    print("loader length is %d"%(len(loader)))
    cnt = 0
    batch_time = AverageMeter()
    s = time.time()
    t0 = time.time()
    for epoch in range(1):
        iter_loader = iter(loader)
        index_lst = []
        for i in range(len(loader)):
            print(i)
            (index, data, label) = next(iter_loader)
            index_lst.extend(index.tolist())
            print(data.shape)
            batch_time.updata(time.time() - t0)
            print("epoch %d [%d] step time is %s" % (epoch, i, str(batch_time.val)))
            print("epoch %d [%d] step avg time is %s" % (epoch, i, str(batch_time.avg)))
            t0 = time.time()
        print("index_lst len:", len(index_lst))
        print("set(index_lst) len", len(set(index_lst)))
    e = time.time()
    print("one epoch %.6f"%((e-s)))