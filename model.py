# 2020.09.07
# build siamese net using VGG as backbone
# construct __init__(), foward()

# 2020.09.08
# build siamese net using AlexNet as backbone
# construct __init__(), foward()

import torch
import torch.nn as nn
import torch.nn.functional as F

class vgg16_sia(nn.Module):

    def __init__(self):
        super(vgg16_sia, self).__init()__

        # 16 weight layer
        # image(3 * 224 * 224)

        # first layer: conv3-64 -> conv3-64
        self.conv1_1 = nn.Conv2d(3, 64, 3) # dimension to3 * 222 * 22
        nn.init.xavier_normal_(self.conv1_1.weight) # init the weight of conv1_1
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=(1, 1))
        nn.init.xavier_normal_(self.conv1_2.weight) # init the weight of conv 1_2
        # maxpool
        self.maxpool1 = nn.MaxPool2d((2, 2), padding=(1, 1))

        # second layer: conv3-128 -> conv128
        self.conv2_1 = nn.Conv2d(64, 128, 3)
        nn.init.xavier_normal_(self.conv2_1.weight)
        self.conv2_2 = nn.Conv2d(128, 128, 3)
        nn.init.xavier_normal_(self.conv2_2.weight)
        # maxpool
        self.maxpool2 = nn.MaxPool2d((2, 2), padding=(1, 1))

        # third layer: conv3-256 -> conv3-256 -> conv3-256
        self.conv3_1 = nn.Conv2d(128, 256, 3)
        nn.init.xavier_normal_(self.conv3_1.weight)
        self.conv3_2 = nn.Conv2d(256, 256, 3)
        nn.init.xavier_normal_(self.conv3_2.weight)
        self.conv3_3 = nn.Conv2d(256, 256, 3)
        nn.init.xavier_normal_(self.conv3_3.weight)
        # maxpool
        self.maxpool3 = nn.MaxPool2d((2, 2), padding=(1, 1))

        # fourth layer: conv3-512 -> conv3-512 -> conv3-512
        self.conv4_1 = nn.Conv2d(256, 512, 3)
        nn.init.xavier_normal_(self.conv4_1.weight)
        self.conv4_2 = nn.Conv2d(512, 512, 3)
        nn.init.xavier_normal_(self.conv4_2.weight)
        self.conv4_3 = nn.Conv2d(512, 512, 3)
        nn.init.xavier_normal_(self.conv4_3.weight)
        # maxpool
        self.maxpool4 = nn.MaxPool2d((2, 2), padding=(1, 1))

        # fifth layer: conv3-512 -> conv3-512 -> conv3-512
        self.conv5_1 = nn.Conv2d(512, 512, 3)
        nn.init.xavier_normal_(self.conv5_1.weight)
        self.conv5_2 = nn.Conv2d(512, 512, 3)
        nn.init.xavier_normal_(self.conv5_2.weight)
        self.conv5_3 = nn.Conv2d(512, 512, 3)
        nn.init.xavier_normal_(self.conv5_3.weight)
        # maxpool -> FC-4096 -> FC-4096 -> FC-1000 -> soft-max
        # channel should be changed when use different dataset
        self.maxpool5 = nn.MaxPool2d((2, 2), padding=(1, 1))
        self.fc1_4096 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2_4096 = nn.Linear(4096, 4096)
        self.fc3_1 = nn.Linear(4096, 1)

    def foward_sigle_stack(self, x):
        # x.size()[0] is input size
        in_size = x.size()[0]

        #first layer
        out = self.conv1_1(x)
        out = nn.ReLU(out)
        out = self.conv1_2(out)
        out = nn.ReLU(out)
        out = self.maxpool1(out)

        # second layer
        out = self.conv2_1(out)
        out = nn.ReLU(out)
        out = self.conv2_2(out)
        out = nn.ReLU(out)
        out = self.maxpool2(out)

        # third layer
        out = self.conv3_1(out)
        out = nn.ReLU(out)
        out = self.conv3_2(out)
        out = nn.ReLU(out)
        out = self.conv3_3(out)
        out = nn.ReLU(out)
        out = self.maxpool3(out)

        # fourth layer
        out = self.conv4_1(out)
        out = nn.ReLU(out)
        out = self.conv4_2(out)
        out = nn.ReLU(out)
        out = self.conv4_3(out)
        out = nn.ReLU(out)
        out = self.maxpool4(out)

        # fifth layer
        out = self.conv5_1(out)
        out = nn.ReLU(out)
        out = self.conv5_2(out)
        out = nn.ReLU(out)
        out = self.conv5_3(out)
        out = nn.ReLU(out)
        out = self.maxpool5(out)

        # two full connecte layers
        out = out.view(in_size, -1)

        out = self.fc1_4096(out)
        out = nn.ReLU(out)
        out = self.fc2_4096(out)
        out = nn.ReLU(out)
        return out

    def foward(self, x1, x2):
        # calculate weight through single stack
        out1 = self.foward_sigle_stack(x1)
        out2 = self.foward_sigle_stack(x2)

        # calculate distance
        dis = torch.abs(x1 - x2)
        out = self.fc3_1(dis)

        return out

class alex_siamese(nn.Module):

    def __init__(self):
        super(alex_siamese, self).__init()__

        # input: 224 * 224 *3
        # first layer
        self.conv1_1 = nn.Conv2d(3, 96, 11, stride=4)
        nn.init.xavier_normal_(self.conv1_1.weight)
        self.maxpool1 = nn.MaxPool2d((3, 3), stride = 2)

        # second layer
        self.conv2_1 = nn.Conv2d(96, 256, 5, stride=1, padding=(2, 2))
        nn.init.xavier_normal_(self.conv2_1.weight)
        self.maxpool2 = nn.MaxPool2d((3, 3), stride=2)

        # third layer
        self.conv3_1 = nn.Conv2d(256, 384, 3, stride=1, padding=(1, 1))
        nn.init.xavier_normal_(self.conv3_1.weight)

        # fourth layer
        self.conv4_1 = nn.Conv2d(384, 384, 3, stride=1, padding=(1, 1))
        nn.init.xavier_normal_(self.conv4_1.weight)

        # fifth layer
        self.conv5_1 = nn.Conv2d(384, 256, 3, stride=1, padding=(1, 1))
        nn.init.xavier_normal_(self.conv5_1.weight)
        self.maxpool5 = nn.MaxPool2d((3, 3), stride=2)

        # sixth layer
        self.dense = nn.Sequential(
            nn.Linear(9216, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 50)
        )

    def foward_single_stack(self, x):
        in_size = x.size()[0]

        # first layer
        out = self.conv1_1(x)
        out = nn.ReLU(out)
        out = self.maxpool1(out)
        out = nn.ReLU(out)

        # second layer
        out = self.conv2_1(out)
        out = nn.ReLU(out)
        out = self.maxpool2(out)
        out = nn.ReLU(out)

        # third layer
        out = self.conv3_1(out)
        out = nn.ReLU(out)
        
        # fourth layer
        out = self.conv4_1(out)
        out = nn.ReLU(out)

        # fifth layer
        out = self.conv4_1(out)
        out = nn.ReLU(out)
        out = self.maxpool5(out)
        out = nn.ReLU(out)

        # sixth layer
        out = out.view(in_size, -1)
        out = self.dense(out)

        return out
    
    def foward(self, x1, x2):
        out1 = self.foward_single_stack(x1)
        out2 = self.foward_single_stack(x2)

        dis = torch.abs(out1 - out2)
        out = self.fc4(dis)

        return out