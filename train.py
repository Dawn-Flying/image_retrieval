import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import shutil
import time
import torch.optim as optim
import config

from PIL import Image
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights

from pylab import mpl

# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
# 设置正常显示符号
mpl.rcParams["axes.unicode_minus"] = False

device = config.device
transform = config.transform
app_config = config.get_app_config()
data_path = app_config.data_path
train_path = app_config.train_path
test_path = app_config.test_path
features_file_path = app_config.features_file_path
model_file_path = app_config.model_file_path
num_epochs = app_config.num_epochs
lr = app_config.lr
momentum = app_config.momentum
batch_size = app_config.batch_size

# 划分训练和测试数据集
def split_data():
    # 是否划分train数据集
    split_train_data = True

    if os.path.exists(train_path):
        if len(os.listdir(train_path)) > 0:
            split_train_data = False

    if split_train_data:
        print('开始划分训练和测试数据集, 80%数据作为训练集...')
        # 创建 train 和 test 子集
        if not os.path.exists(train_path):
            os.makedirs(train_path)
        if not os.path.exists(test_path):
            os.makedirs(test_path)

        categories = os.listdir(data_path)
        for category in categories:
            image_names = os.listdir(os.path.join(data_path, category))
            num_images = len(image_names)
            num_train = int(num_images * 0.8)
            train_names = image_names[:num_train]
            test_names = image_names[num_train:]
            for train_name in train_names:
                src_path = os.path.join(data_path, category, train_name)
                dst_path = os.path.join(train_path, category, train_name)
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.copy(src_path, dst_path)
            for test_name in test_names:
                src_path = os.path.join(data_path, category, test_name)
                dst_path = os.path.join(test_path, category, test_name)
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.copy(src_path, dst_path)

        print(f'划分训练集和测试集完毕, 训练集目录: {train_path}, 测试集目录: {test_path}')


# 绘制损失函数
def show_loss(epoch_records, loss_records):
    plt.figure(figsize=(20, 8), dpi=80)
    plt.plot(epoch_records, loss_records, label="损失")
    plt.legend(loc="best")
    plt.savefig("./loss")
    plt.show()
    plt.close()


# 训练模型
def train():
    # 划分训练和测试数据集
    split_data()
    # 加载resnet50预训练模型
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    # 使用resnet50提确特征, 将最后的全连接层输出改为caltech-101数据集的分类数102即可。
    # 替换最后一层全连接层
    model.fc = nn.Linear(model.fc.in_features, 102)
    model.to(device)

    # 提取数据集中所有图像的特征向量
    train_dataset = ImageFolder(train_path, transform=transform)
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    dataset_len = len(train_dataset)

    if os.path.exists(model_file_path):
        # 加载特征向量文件
        model.load_state_dict(torch.load(model_file_path))
    else:
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr, momentum)

        loss_records = []
        epoch_records = []

        # 训练模型
        start_time = time.time()
        for epoch in tqdm(range(num_epochs)):
            running_loss = 0.0
            for i, data in enumerate(tqdm(trainloader)):
                images, labels = data
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            print('[%d] loss: %.3f' % (epoch + 1, running_loss))
            loss_records.append(running_loss)
            epoch_records.append(epoch+1)
        end_time = time.time()
        print('Finished Training. Time: {}'.format(end_time - start_time))

        # 保存模型
        torch.save(model.state_dict(), model_file_path)
        # 绘制损失函数
        show_loss(epoch_records, loss_records)

    # 最后移除全连接层
    model.fc = nn.Identity()
    model.eval()

    # 如果特征存在则直接返回
    if os.path.exists(features_file_path):
        # 加载特征向量文件
        with open(features_file_path, 'rb') as f:
            features = np.load(f, allow_pickle=True).item()

        return model, features

    # 记录图像的特征
    features = {}
    print('开始获取训练集图像特征...')

    for i in tqdm(range(dataset_len)):
        filenamePath = train_dataset.imgs[i][0]
        image_tensor = train_dataset[i][0]
        image_tensor = torch.unsqueeze(image_tensor, 0).to(device)
        with torch.no_grad():
            feature_tensor = model(image_tensor)
            feature_vector = torch.flatten(feature_tensor, start_dim=1)
            # 存放图片路径和对应的特征
            features[filenamePath] = feature_vector.cpu().numpy()

    print(dataset_len)
    print('获取训练集图像特征完毕')

    # 将特征向量保存到文件
    with open(features_file_path, 'wb') as f:
        np.save(f, features)
        print(f'特征数据已保存在{features_file_path}目录')

    # 返回模型和特征仓库数据集
    return model, features


# 测试一波
def query_demo(model, features):
    # 对查询图像进行特征提取并计算相似度
    query_image_path = os.path.join('data', 'image_0035.jpg')
    query_image = Image.open(query_image_path).convert('RGB')
    query_image_tensor = transform(query_image)
    query_image_tensor = torch.unsqueeze(query_image_tensor, 0).to(device)

    with torch.no_grad():
        query_feature_tensor = model(query_image_tensor)
        query_feature_vector = torch.flatten(query_feature_tensor, start_dim=1).cpu().numpy()
        query_feature_vector = np.squeeze(query_feature_vector)

    similarities = {}
    for filename, feature_vector in features.items():
        feature_vector = np.squeeze(feature_vector)
        similarity = np.dot(query_feature_vector, feature_vector) / (
                np.linalg.norm(query_feature_vector) * np.linalg.norm(feature_vector))
        similarities[filename] = similarity

    # 根据相似度排序并返回相似度最高的图像
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:5]

    plt.subplot(261)
    plt.imshow(query_image)

    i = 1
    for path, score in sorted_similarities:
        plt.subplot(261 + i)
        plt.imshow(Image.open(path).convert('RGB'))
        plt.title(f'排名: {i} 分值: {round(score.item(), 3)}')
        i = i + 1

    plt.show()


if __name__ == '__main__':
    model, features = train()
    query_demo(model, features)
