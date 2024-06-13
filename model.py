import torch.nn as nn
import torch
import torchvision.models as models
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out
class Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4 #第三层卷积层的卷积核个数是第一二层的4倍

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity  #捷径identity
        out = self.relu(out)

        return out
class ResNet(nn.Module):

    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        # if self.include_top:
        #     self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
        #     self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # if self.include_top:
        #     x = self.avgpool(x)
        return x
def resnet34( include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], include_top=include_top)
def resnet50(include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], include_top=include_top)
# class CRNN(nn.Module):
#
#     def __init__(self, img_channel, img_height, img_width, num_class,
#                  map_to_seq_hidden=64, rnn_hidden=256, leaky_relu=False):
#         super(CRNN, self).__init__()
#
#         self.cnn, (output_channel, output_height, output_width) = \
#             self._cnn_backbone(img_channel, img_height, img_width, leaky_relu)
#         self.output_channel = img_channel,
#         self.output_height = img_height // 16 - 1,
#         self. output_width = img_width // 4 - 1
#         # self.resnet = models.resnet50(pretrained=True)
#         # self.resnet.fc = nn.Identity()
#         #
#         # self.cnn = nn.Sequential(
#         #     self.resnet,
#         #     nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0),
#         #     nn.BatchNorm2d(512),
#         #     nn.ReLU(inplace=True)
#         # )
#         # self.cnn = nn.Sequential(
#         #     resnet34(),
#         #     nn.MaxPool2d(kernel_size=(1, 1)),
#         #     nn.ReLU(inplace=True)
#         # )
#
#         # self.cnn = resnet34()
#
#
#         self.map_to_seq = nn.Linear(512 * (img_height // 32), map_to_seq_hidden)
#         # self.map_to_seq = nn.Linear(2048, 256)
#
#         self.rnn1 = nn.LSTM(map_to_seq_hidden, rnn_hidden, bidirectional=True)
#
#         # 如果接双向lstm输出，则要 *2,固定用法
#         self.rnn2 = nn.LSTM(2 * rnn_hidden, rnn_hidden, bidirectional=True)
#
#         self.dense = nn.Linear(2 * rnn_hidden, num_class)
#
#     # CNN主干网络
#     def _cnn_backbone(self, img_channel, img_height, img_width, leaky_relu):
#         assert img_height % 16 == 0
#         assert img_width % 4 == 0
#
#         # 超参设置
#         channels = [img_channel, 64, 128, 256, 256, 512, 512, 512]
#         kernel_sizes = [3, 3, 3, 3, 3, 3, 2]
#         strides = [1, 1, 1, 1, 1, 1, 1]
#         paddings = [1, 1, 1, 1, 1, 1, 0]
#
#         cnn = nn.Sequential()#从前往后输出到输入
#
#         def conv_relu(i, batch_norm=False):
#             # shape of input: (batch, input_channel, height, width)
#             input_channel = channels[i]
#             output_channel = channels[i+1]
#             #添加模块
#             cnn.add_module(
#                 f'conv{i}',
#                 nn.Conv2d(input_channel, output_channel, kernel_sizes[i], strides[i], paddings[i])
#             )
#
#             if batch_norm:
#                 cnn.add_module(f'batchnorm{i}', nn.BatchNorm2d(output_channel))
#
#             relu = nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)
#             cnn.add_module(f'relu{i}', relu)
#
#         # size of image: (channel, height, width) = (img_channel, img_height, img_width)
#         conv_relu(0)
#         cnn.add_module('pooling0', nn.MaxPool2d(kernel_size=2, stride=2))
#         # (64, img_height // 2, img_width // 2)
#
#         conv_relu(1)
#         cnn.add_module('pooling1', nn.MaxPool2d(kernel_size=2, stride=2))
#         # (128, img_height // 4, img_width // 4)
#
#         conv_relu(2)
#         conv_relu(3)
#         cnn.add_module(
#             'pooling2',
#             nn.MaxPool2d(kernel_size=(2, 1))
#         )  # (256, img_height // 8, img_width // 4)
#
#         conv_relu(4, batch_norm=True)
#         conv_relu(5, batch_norm=True)
#         cnn.add_module(
#             'pooling3',
#             nn.MaxPool2d(kernel_size=(2, 1))
#         )  # (512, img_height // 16, img_width // 4)
#
#         conv_relu(6)  # (512, img_height // 16 - 1, img_width // 4 - 1)
#
#         output_channel, output_height, output_width = \
#             channels[-1], img_height // 16 - 1, img_width // 4 - 1
#         return cnn, (output_channel, output_height, output_width)

    # CNN+LSTM前向计算
class CRNN(nn.Module):

    def __init__(self, img_channel, img_height, img_width, num_class,
                 map_to_seq_hidden=64, rnn_hidden=256, leaky_relu=False):
        super(CRNN, self).__init__()

        self.cnn, (output_channel, output_height, output_width) = \
            self._cnn_backbone(img_channel, img_height, img_width, leaky_relu)
        # self.cnn1 = nn.Sequential(resnet34(),
        #                            nn.Conv2d(512, 512, 1, 1, 1),
        #                           nn.BatchNorm2d(512),
        #                           nn.MaxPool2d(kernel_size=(2, 1)),
        #                           nn.LeakyReLU(0.2, inplace=True),
        #                                        nn.Dropout(p=0.2))


        self.map_to_seq = nn.Linear(output_channel * output_height, map_to_seq_hidden)
        # self.map_to_seq = nn.Linear(512 * 1, map_to_seq_hidden)

        self.rnn1 = nn.LSTM(map_to_seq_hidden, rnn_hidden, bidirectional=True)

        # 如果接双向lstm输出，则要 *2,固定用法
        self.rnn2 = nn.LSTM(2 * rnn_hidden, rnn_hidden, bidirectional=True)

        self.dense = nn.Linear(2 * rnn_hidden, num_class)#输入神经元个数，输出神经元个数

    # CNN主干网络
    def _cnn_backbone(self, img_channel, img_height, img_width, leaky_relu):
        assert img_height % 16 == 0
        assert img_width % 4 == 0

        # 超参设置
        channels = [img_channel, 64, 128, 256, 256, 512, 512, 512]
        kernel_sizes = [3, 3, 3, 3, 3, 3, 2]
        strides = [1, 1, 1, 1, 1, 1, 1]
        paddings = [1, 1, 1, 1, 1, 1, 0]

        cnn = nn.Sequential()

        def conv_relu(i, batch_norm=False):
            # shape of input: (batch, input_channel, height, width)
            input_channel = channels[i]
            output_channel = channels[i + 1]

            cnn.add_module(
                f'conv{i}',
                nn.Conv2d(input_channel, output_channel, kernel_sizes[i], strides[i], paddings[i])
            )

            if batch_norm:
                cnn.add_module(f'batchnorm{i}', nn.BatchNorm2d(output_channel))

            relu = nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)
            cnn.add_module(f'relu{i}', relu)

        #size of image: (channel, height, width) = (img_channel, img_height, img_width)
        conv_relu(0)
        cnn.add_module('pooling0', nn.MaxPool2d(kernel_size=2, stride=2))
        # (64, img_height // 2, img_width // 2)

        conv_relu(1)
        cnn.add_module('pooling1', nn.MaxPool2d(kernel_size=2, stride=2))
        # (128, img_height // 4, img_width // 4)

        conv_relu(2, True)
        conv_relu(3)
        cnn.add_module(
            'pooling2',
            nn.MaxPool2d(kernel_size=(2, 1))
        )  # (256, img_height // 8, img_width // 4)

        conv_relu(4, batch_norm=True)
        conv_relu(5, batch_norm=True)
        cnn.add_module(
            'pooling3',
            nn.MaxPool2d(kernel_size=(2, 1))
        )  # (512, img_height // 16, img_width // 4)

        conv_relu(6, True)  # (512, img_height // 16 - 1, img_width // 4 - 1)

        output_channel, output_height, output_width = \
            channels[-1], img_height // 16 - 1, img_width // 4 - 1
        return cnn, (output_channel, output_height, output_width)
    def forward(self, images):
        # shape of images: (batch, channel, height, width)

        conv = self.cnn(images)
        # conv = self.cnn(conv)
        # print(conv.shape)
        # conv = self.cnn(conv)
        batch, channel, height, width = conv.size()

        conv = conv.view(batch, channel * height, width)
        conv = conv.permute(2, 0, 1)  # (width, batch, feature)

        # 卷积接全连接。全连接输入形状为(width, batch, channel*height)，
        # 输出形状为(width, batch, hidden_layer)，分别对应时序长度，batch，特征数，符合LSTM输入要求

        seq = self.map_to_seq(conv)
        # seq = self.cnn1(seq)
        recurrent, _ = self.rnn1(seq)
        recurrent, _ = self.rnn2(recurrent)

        output = self.dense(recurrent)
        return output  # shape: (seq_len, batch, num_class)

# input1 = torch.rand([64, 1, 3, 3])
# model = CRNN(1, 32, 100, 37)
# print(model)
# output = model(input1)