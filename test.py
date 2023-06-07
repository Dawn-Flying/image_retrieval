import torch
import torchvision.datasets as datasets
import numpy as np
import config

from tqdm import tqdm
from train import train
from sklearn.metrics import average_precision_score

# 测试集目录
app_config = config.get_app_config()
test_path = app_config.test_path
device = config.device


# MAP和Top-1评估指标
def evaluate(model, data_dir):
    transform = config.transform

    # 加载测试数据集
    test_data = datasets.ImageFolder(data_dir, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False, num_workers=4)

    # 提取特征向量并计算相似度
    features = []
    labels = []
    model.eval()
    with torch.no_grad():
        for images, image_labels in tqdm(test_loader):
            images = images.to(device)
            features_batch = model(images)
            features_batch = features_batch.cpu().numpy()
            features.extend(features_batch)
            labels.extend(image_labels.numpy())

    features = np.array(features)
    labels = np.array(labels)
    similarities = np.dot(features, features.T)
    # 使用余弦相似度进行归一化
    similarities = similarities / np.outer(np.linalg.norm(features, axis=1),
                                           np.linalg.norm(features, axis=1))

    # 计算精度指标
    top1_acc = 0
    map_acc = 0
    for i in tqdm(range(len(test_data))):
        query_idx = i
        query_label = labels[query_idx]
        query_similarities = similarities[query_idx]
        # 获取query_similarities从大到小排序的索引数组， 相似度最大的在最前面
        ranked_indices = np.argsort(-query_similarities)
        ranked_labels = labels[ranked_indices]

        # top1
        if ranked_labels[0] == query_label:
            top1_acc += 1

        # 计算排名越靠前的样本的精度, query_label每个样本的真实标签, -query_similarities表示每个样本的预测得分
        ap = average_precision_score(labels == query_label, -query_similarities)
        map_acc += ap

    top1_acc = top1_acc / len(test_data)
    map_acc = map_acc / len(test_data)

    print('Top-1 accuracy: {:.4f}'.format(top1_acc))
    print('Mean Average Precision: {:.4f}'.format(map_acc))


if __name__ == '__main__':
    # 获取模型
    model, _ = train()
    evaluate(model, test_path)
