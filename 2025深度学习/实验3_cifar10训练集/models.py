import torch
import torch.nn as nn
import torch.nn.functional as F
# 原始CNN模型
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        
        # 第二个卷积层
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第三个卷积层
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        
        # 第四个卷积层
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全连接层
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.relu5 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 10)
        
    def forward(self, x):
        # 卷积层 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        # 卷积层 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool1(x)
        
        # 卷积层 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        # 卷积层 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool2(x)
        
        # 全连接层
        x = x.view(-1, 128 * 8 * 8)
        x = self.fc1(x)
        x = self.relu5(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# LeNet模型 (适配CIFAR-10的32x32输入)
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.classifier(x)
        return x

# AlexNet模型 (修改以适应CIFAR-10的32x32输入)
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 4 * 4)
        x = self.classifier(x)
        return x

# 定义ResNet模型
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(residual)
        out = self.relu(out)
        
        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # ResNet层
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        self.avg_pool = nn.AvgPool2d(4)
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

# VGG16模型 (修改以适应CIFAR-10的32x32输入)
class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            # 第1块 - 2个卷积层
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第2块 - 2个卷积层
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第3块 - 3个卷积层
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第4块 - 3个卷积层
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # 对于CIFAR-10，特征图尺寸在最后一个最大池化层后为2x2
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# 定义 Residual 类
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, 
                              padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, 
                                  kernel_size=1, stride=strides)
        else:
            self.conv3 = None
            
    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

# 定义 resnet_block 函数，移到类外部作为全局函数
def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, 
                               use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

# 完整的 ResNet 模型
class Resnet_fruit(nn.Module):
    def __init__(self, num_classes=10):
        super(Resnet_fruit, self).__init__()
        # 使用3通道输入（RGB彩色图像）
        self.b1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
        self.b3 = nn.Sequential(*resnet_block(64, 128, 2))
        self.b4 = nn.Sequential(*resnet_block(128, 256, 2))
        self.b5 = nn.Sequential(*resnet_block(256, 512, 2))
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# 添加SE注意力模块
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# 改进的残差块，增加了SE注意力机制和Bottleneck结构
class ResidualSEBottleneck(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1, use_se=True):
        super(ResidualSEBottleneck, self).__init__()
        
        # Bottleneck设计: 降维-卷积-升维
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # 添加SE注意力模块
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(out_channels)
            
        # 跳跃连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        # Dropout for regularization
        self.dropout = nn.Dropout2d(0.1)
            
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        # 应用SE注意力
        if self.use_se:
            out = self.se(out)
            
        # 跳跃连接
        out += self.shortcut(residual)
        out = F.relu(out)
        
        # 轻微dropout
        out = self.dropout(out)
        
        return out

# 构建改进的ResNet块
def resnet_bottleneck_block(in_channels, mid_channels, out_channels, num_blocks, stride=1):
    layers = []
    # 第一个块可能改变维度
    layers.append(ResidualSEBottleneck(in_channels, mid_channels, out_channels, stride))
    
    # 其余块保持维度不变
    for _ in range(1, num_blocks):
        layers.append(ResidualSEBottleneck(out_channels, mid_channels, out_channels))
        
    return nn.Sequential(*layers)

# 改进版的ResNet模型
class Resnet_fruit_v2(nn.Module):
    def __init__(self, num_classes=10):
        super(Resnet_fruit_v2, self).__init__()
        
        # 初始层 - 适合小尺寸输入图像的轻量级设计
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # 不使用maxpool以保留小图像的空间信息
        # 使用Bottleneck结构的残差块，减少参数量增加模型深度
        self.layer1 = resnet_bottleneck_block(64, 64, 256, 3)  # 保持空间尺寸
        self.layer2 = resnet_bottleneck_block(256, 128, 512, 4, stride=2)  # 减半空间尺寸
        self.layer3 = resnet_bottleneck_block(512, 256, 1024, 6, stride=2)  # 再减半
        self.layer4 = resnet_bottleneck_block(1024, 512, 2048, 3, stride=2)  # 再减半
        
        # 全局池化和分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类器包含Dropout减少过拟合
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x

# 模型管理类
class ModelManager:
    def __init__(self):
        self.models = {
            'custom_cnn': CustomCNN,
            'lenet': LeNet,
            'alexnet': AlexNet,
            'resnet18': ResNet18,
            'vgg16': VGG16,
            'resnet_fruit': Resnet_fruit,
            'resnet_fruit_v2': Resnet_fruit_v2  # 添加改进版模型
        }
    
    def get_model(self, model_name):
        """根据模型名称返回相应的模型实例"""
        if model_name not in self.models:
            raise ValueError(f"模型 '{model_name}' 不存在。可用模型: {list(self.models.keys())}")
        
        model = self.models[model_name]()
        return model
    
    def list_available_models(self):
        """列出所有可用的模型名称"""
        return list(self.models.keys())
    
    def add_model(self, model_name, model_class):
        """添加新的模型到模型管理器"""
        if model_name in self.models:
            print(f"警告: 模型 '{model_name}' 已存在，将被覆盖")
        self.models[model_name] = model_class

# 获取模型第一层卷积权重的函数
def get_first_conv_layer(model, model_name):
    """获取模型的第一个卷积层权重，用于可视化"""
    if model_name == 'custom_cnn':
        return model.conv1.weight.data
    elif model_name == 'lenet':
        return model.features[0].weight.data
    elif model_name == 'alexnet':
        return model.features[0].weight.data
    elif model_name == 'resnet18':
        return model.conv1.weight.data
    elif model_name == 'vgg16':
        return model.features[0].weight.data
    elif model_name == 'resnet_fruit':
        return model.b1[0].weight.data
    elif model_name == 'resnet_fruit_v2':
        return model.conv1.weight.data
    else:
        raise ValueError(f"不支持模型 '{model_name}' 的第一层卷积权重获取") 