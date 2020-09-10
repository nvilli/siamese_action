# This is a configuration file to define model, train and test parameters
# 2020.09.10 add some parameters, and update

import os
import time
from easydict import EasyDict as edict

_C = edict()
SysPara = _C

_C.TIMESTAMP = time.strftime("%Y_%m_%d_%H:%M:%S", time.localtime())
_C.WORKSPACE_PATH = os.getcwd()
_C.GPUS = 2

_C.EXP_TUPE = 'T'
_C.MODEL_NAME = 'siamese_representation'
# backbone has three choice, vgg16, alex, and resnet
_C.BACKBONE = 'VGG16'
# pre-trained or not, if not, 'None'
_C.PRETRAIN_TYPE = 'imagenet'

_C.SNAPSHOT_ROOT = '/data'
_C.EXP_NAME = "{}_{}_{}".format(_C.MODEL_NAME, _C.BACKBONE, _C.PRETRAIN_TYPE)

_C.DATASET = edict()
_C.DATASET.ROOT_PATH = "/data"
_C.DATASET.NAME = "HMDB51"
_C.DATASET.CLASS_NUM = 174
_C.DATASET.IMG_FORMAT = 'img_{:05d}.jpg'
_C.DATASET.VIDEO_SEQ_PATH = os.path.join(_C.DATASET.ROOT_PATH, _C.DATASET.NAME, 'VideoSeq')
_C.DATASET.TRAIN_META_PATH = os.path.join(_C.DATASET.ROOT_PATH, _C.DATASET.NAME, 'MetafileSeq')
_C.DATASET.VAL_META_PATH = os.path.join(_C.DATASET, _C.DATASET.NAME, 'MetafileSeq')
_C.DATASET.TRAIN_SAMPLE_METHOD = 'seg_random'
_C.DATASET.VAL_SAMPLE_METHOD = 'seg_ratio'
_C.DATASET.LOAD_METHOD_LIST = ['TSNDataset', 'DPFlowDataset', 'HumanactionDataset', 'DPFlowUntrimmedDataset']

_C.TRAIN = edict()
_C.TRAIN.EPOCHS = 80
_C.TRAIN.BASE_BATCH_SIZE = 64
_C.TRAIN.BATCH_SIZE = _C.TRAIN.BASE_BATCH_SIZE//_C.GPUS if _C.USE_DISTRIBUTED else _C.TRAIN.BASE_BATCH_SIZE
_C.TRAIN.BASE_LR = 0.001
_C.TRAIN.LR_STEP_TYPE = 'step'
_C.TRAIN.LR_STEPS = [20, 40, 60]
_C.TRAIN.PERIOD_EPOCH = 10
_C.TRAIN.WARMUP_EPOCH = 10
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WEIGHT_DECAY = 5e-4
_C.TRAIN.CLIP_GRADIENT = 20
_C.TRAIN.PRINT_FREQ = 4
_C.TRAIN.EVALUATE_FREQ = 5
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.BEST_PREC = 0.0
_C.TRAIN.DROPOUT = 0.8
_C.TRAIN.SEG_NUM = 16
_C.TRAIN.CROP_NUM = 1
_C.TRAIN.MODALITY = 'RGB'
_C.TRAIN.PARTIAL_BN = True

_C.VAL = edict()
_C.VAL.BATCH_SIZE = 1
_C.VAL.PRINT_FREQ = 1
_C.VAL.SEG_NUM = 16
_C.VAL.CROP_NUM = 1
_C.VAL.MODALITY = 'RGB'

_C.IMG = edict()
_C.IMG.CROP_SIZE = 224
_C.IMG.SCALE_SIZE = _C.IMG.CROP_SIZE * 256 // 224
_C.IMG.MEAN = 0
_C.IMG.STD = 0

__C.PRETRAIN_MODEL_DICT = {
    'tsn':{
        'BNInception':{
            'imagenet':'bn_inception-52deb4733.pth',
            'resume':'',
            'finetune':''
        },
        'resnet50':{
            'imagenet':'resnet50-19c8e357.pth',
            'resume':'',
            'finetune': ''
        },
        'inceptionv3':{
            'imagenet':'inception_v3_google-1a9a5a14.pth',
            'resume':'',
            'finetune': ''
        }
    },
    'i3d':{
        'resnet50':{
            'imagenet':'resnet50-19c8e357.pth',
            'resume':'',
            'finetune': ''
        }
    },
    'slowfast':{
        'resnet50':{
            'imagenet':'resnet50-19c8e357.pth',
            'resume':'',
            'finetune': ''
        }
    },
    'tsm':{
        'resnet50':{
            'imagenet': 'resnet50-19c8e357.pth',
            'resume':'',
            'finetune': ''
        },
        'BNInception':{
            'imagenet': 'BNInception-9baff57459f5a1744.pth',
            'kinetics': 'BNInceptionKinetics-47f0695e.pth',
            'resume':'',
            'finetune': ''
        },
        'mobilenetv2':{
            'imagenet':'mobilenetv2_1.0-f2a8633.pth.tar',
            'resume':'',
            'finetune': ''
        }
    },
    'mfnet3d':{
        'mfnet2d':{
            'imagenet': 'MFNet2D_ImageNet1k-0000.pth',
            'resume':'',
            'finetune': ''
        }
    }
}