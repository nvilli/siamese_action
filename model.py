# 2020.09.07
# build siamese net using VGG as backbone
# construct __init__(), foward()

# 2020.09.08
# build siamese net using AlexNet as backbone
# construct __init__(), foward()

# 2020.08.10
# build siamese net using ResNet as backbone
# construct __init()__, foward()__

import torch
import torch.nn as nn
import torch.nn.functional as F


#siamese net, VGG16 as backbone
class vgg16_sia(nn.Module):

    def __init__(self):
        super(vgg16_sia, self).__init__()

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
        dis = torch.abs(out1 - out2)
        out = self.fc3_1(dis)

        return out


# siamese net, AlexNet as backbone
class alex_siamese(nn.Module):

    def __init__(self):
        super(alex_siamese, self).__init__()

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




# siamese net, ResNet as backbone
class ResidualBlock(nn.Module):

    def __init__(self, in_channel, out_channel, stride, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
         nn.Conv2d(in_channel, out_channel, 3, stride, 1, bias=False),
         nn.BatchNorm2d(out_channel),
         nn.ReLU(inplace=True),
         nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False),
         nn.BatchNorm2d(out_channel)   
        )
        self.right = shortcut

    
    def foward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)

class ResNet(nn.Module):

    def __init__(self, num_class=1000):
        super(ResNet, self).__init__()

        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        self.layer1 = self._make_layer(64, 128, 3)
        self.layer2 = self._make_layer(128, 256, 4, stride=2)
        self.layer3 = self._make_layer(256, 512, 6, stride=2)
        self.layer4 = self._make_layer(512, 512, 3, stride=2)
        
        self.fc_final = nn.Linear(in_features=1000, 1000)

        self.fc = nn.Linear(512, num_class)

    def _make_layer(self, in_channel, out_channel, block_num, stride=1):
        shortcut = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, stride, bias=False),
            nn.BatchNorm2d(out_channel)
        )

        layers = []
        layers.append(ResidualBlock(in_channel, out_channel, stride, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock(out_channel, out_channel))

        return nn.Sequential(*layers)

    def forward_single_stack(self, x):
        x = self.pre(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 7)
        x = x.view(x.size()[0], -1)
        return self.fc(x)

    def forward(self, x1, x2):
        out1 = forward_single_stack(x1)
        out2 = forward_single_stack(x2)

        dis = torch.abs(out1 - out2)
        out = self.fc_final(dis)

        return out