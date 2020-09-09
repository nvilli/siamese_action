import torch
import pickle
import torchvision
from torchvision import transforms
import torchvision.datasets as dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib as plt
from model import vgg16_sia, alex_siamese
import torchvision.models as models
import time
import numpy as np
import sys
import os

if __name__ == '__main__':

    