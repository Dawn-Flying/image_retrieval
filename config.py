# -*- coding: utf-8 -*-
import argparse
import os
import torch
from torchvision import transforms

parser = argparse.ArgumentParser(description='Configuration file')
arg_lists = []

# 数据根目录
root_path = 'data'
# 数据集目录 data/caltech-101/101_ObjectCategories
data_path = os.path.join(root_path, 'caltech-101', '101_ObjectCategories')
# 训练集目录
train_path = os.path.join(root_path, 'train')
# 测试集目录
test_path = os.path.join(root_path, 'test')
# 模型存放目录
models_path = os.path.join(root_path, 'models')
# 存放图像特征数据 (key, value) key为图像路径，value为特征
features_file_path = os.path.join(models_path, 'features_train_50.npy')
model_file_path = os.path.join(models_path, 'caltech101_resnet50.pth')
# 运行环境
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 训练次数
num_epochs = 10
# 学习率
lr=0.001
momentum=0.9
batch_size=32

if not os.path.exists(root_path):
    os.makedirs(root_path)
if not os.path.exists(models_path):
    os.makedirs(models_path)

# 图像转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 添加配置
def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--data_path', type=str, default=data_path, help='数据集目录: data/caltech-101/101_ObjectCategories目录')
data_arg.add_argument('--model_file_path', type=str, default=model_file_path, help='model_file文件目录: data/models/caltech101_resnet50.pth')
data_arg.add_argument('--features_file_path', type=str, default=features_file_path, help='特征文件目录: data/models/features_train_50.npy')
data_arg.add_argument('--train_path', type=str, default=train_path, help='生成的训练数据目录: data/train')
data_arg.add_argument('--test_path', type=str, default=test_path, help='生成的测试数据目录: data/test')
data_arg.add_argument('--num_epochs', type=int, default=num_epochs, help='训练次数：100')
data_arg.add_argument('--lr', type=float, default=lr, help='学习率：0.001')
data_arg.add_argument('--momentum', type=float, default=momentum, help='momentum: 0.9')
data_arg.add_argument('--batch_size', type=int, default=batch_size, help='batch_size: 32')


def get_app_config():
    config, unparsed = parser.parse_known_args()
    return config


# 输出配置信息
def print_config(config):
    print('配置信息:')
    print('* 数据集目录 data_path:', config.data_path)
    print('* 模型目录 model_file_path:', config.model_file_path)
    print('* 特征目录 features_file_path:', config.features_file_path)
    print('* 训练集目录 train_path:', config.train_path)
    print('* 测试集目录 test_path:', config.test_path)
    print('* 运行环境 device:', device)
    print('* 图像转换函数 transform:', transform)
    print('* 训练次数 num_epochs:', config.num_epochs)
    print('* 学习率 lr:', config.lr)
    print('* momentum:', config.momentum)
    print('* batch_size:', config.batch_size)


if __name__ == '__main__':
    config = get_app_config()
    print_config(config)