import sys
import torch
import numpy as np

import config
from config import print_config
from train import train

from PIL import Image
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QFileDialog, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget

device = config.device
transform = config.transform

app_config = config.get_app_config()
print_config(app_config)
train_dir = app_config.train_path
features_file_path = app_config.features_file_path


# PyQt5可视化窗口
class ImageRetrievalWindow(QWidget):
    def __init__(self, model, features):
        super().__init__()
        self.model = model
        self.features = features
        # 布局
        self.initUI()

    # 初始化界面布局
    def initUI(self):
        # 设置窗口标题
        self.setWindowTitle("图像检索系统")

        # 创建上传布局
        upload_layout = QVBoxLayout()
        upload_layout.setAlignment(Qt.AlignTop)

        self.upload_button = QPushButton('上传图像')
        self.upload_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.upload_button.clicked.connect(self.upload_image)
        upload_layout.addWidget(self.upload_button)
        # 填充一个空块用来对齐
        upload_layout.addWidget(QLabel())

        self.query_image_label = QLabel()
        self.query_image_label.setFixedSize(224, 224)
        self.query_image_label.setStyleSheet('border: 1px solid gray')
        upload_layout.addWidget(self.query_image_label)

        # 创建查询布局1
        query_layout_part_one = QVBoxLayout()
        query_layout_part_one.setAlignment(Qt.AlignTop)

        # 创建查询布局2
        query_layout_part_two = QVBoxLayout()
        query_layout_part_two.setAlignment(Qt.AlignTop)
        # 填充一个空块用来对齐
        query_layout_part_two.addWidget(QLabel())

        self.query_button = QPushButton('查询')
        self.query_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.query_button.clicked.connect(self.query_image)
        query_layout_part_one.addWidget(self.query_button)

        self.retrieved_image_labels = []
        self.retrieved_image_titles = []

        for i in range(5):
            # 图像框
            retrieved_image_label = QLabel()
            retrieved_image_label.setFixedSize(224, 224)
            retrieved_image_label.setStyleSheet('border: 1px solid gray')
            self.retrieved_image_labels.append(retrieved_image_label)

            retrieved_image_title = QLabel()
            self.retrieved_image_titles.append(retrieved_image_title)

            if i >= 3:
                query_layout_part_two.addWidget(retrieved_image_title)
                query_layout_part_two.addWidget(retrieved_image_label)
            else :
                query_layout_part_one.addWidget(retrieved_image_title)
                query_layout_part_one.addWidget(retrieved_image_label)

        # 创建水平布局
        horizontal_layout = QHBoxLayout()
        horizontal_layout.addLayout(upload_layout)
        horizontal_layout.addLayout(query_layout_part_one)
        horizontal_layout.addLayout(query_layout_part_two)

        # 将水平布局添加到主布局中
        self.setLayout(horizontal_layout)

    def upload_image(self):
        # 打开文件对话框
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter('Images (*.png *.xpm *.jpg *.jpeg *.bmp)')
        if file_dialog.exec_():
            # 获取文件路径
            file_path = file_dialog.selectedFiles()[0]

            # 显示上传的图像
            query_image = QImage(file_path)
            self.query_image_label.setPixmap(QPixmap.fromImage(query_image).scaled(224, 224, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.query_image_label.setScaledContents(True)

            # 保存文件路径
            self.file_path = file_path

    def query_image(self):
        # 对查询图像进行特征提取并计算相似度
        query_image = Image.open(self.file_path).convert('RGB')
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

        i = 0
        # 显示相似图像
        for path, score in sorted_similarities:
            retrieved_image = self.image_to_qimage(path)
            self.retrieved_image_labels[i].setPixmap(QPixmap.fromImage(retrieved_image).scaled(224, 224, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.retrieved_image_labels[i].setScaledContents(True)
            self.retrieved_image_titles[i].setText(f"排名: {i+1} 得分: {round(score.item(), 3)}")
            i += 1

    def image_to_qimage(self, path):
        image = Image.open(path).convert('RGBA')
        qimage = QImage(image.tobytes(), image.width, image.height, QImage.Format_RGBA8888)
        return qimage


if __name__ == '__main__':
    # 训练数据
    model, features = train()
    # 启动可视化窗口
    app = QApplication(sys.argv)
    window = ImageRetrievalWindow(model=model, features=features)
    window.showMaximized = True
    window.show()
    sys.exit(app.exec_())

