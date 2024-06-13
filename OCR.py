# from PIL import Image
# import matplotlib.pyplot as plt
# import numpy as np
# import cv2
# import os
# import matplotlib.pyplot as plt
import random
#
# imgs_folder_path = 'D:\\匣钵计数\\带字钵'
# imgs_filepaths = [os.path.join(imgs_folder_path, f) for f in os.listdir(imgs_folder_path) if f.endswith('.jpg') or f.endswith('.png')]
#
# for img_filepath in range(0, len(imgs_filepaths)):
#     print(imgs_filepaths[img_filepath])
#     with Image.open(imgs_filepaths[img_filepath]) as img3:
#         img2 = np.uint8(img3)
#         sum_rows = img2.shape[0]
#         # the image length
#         sum_cols = img2.shape[1]
#         part1 = img2[int(sum_rows * 0.7):sum_rows, 0:int(sum_cols * 0.4)]
#         part2 = Image.fromarray(part1) #np数组转换为PIL图像对象
#         part2.save(imgs_filepaths[img_filepath])
#图片重命名
import os

# # 假设folder_path是你的图片文件夹路径
# folder_path = 'D:/匣钵计数/带字钵'
#
# # 获取文件夹中的所有文件名
# file_names = os.listdir(folder_path)
#
# # 设置新的文件名前缀
# new_name_prefix = '_JS0719'
#
# # 遍历文件夹中的每个文件
# for i, file_name in enumerate(file_names):
#     # 构建新的文件名
#     new_file_name = str(10034+i) + new_name_prefix + '.' + file_name.split('.')[-1]
#
#     # 构建旧的文件路径和新的文件路径
#     old_file_path = os.path.join(folder_path, file_name)
#     new_file_path = os.path.join(folder_path, new_file_name)
#
#     # 重命名文件
#     os.rename(old_file_path, new_file_path)
#文件名追加
# 假设folder_path是你的图片文件夹路径
folder_path = 'D:\Pycharm_Projects/crnn1/CRNN/data/images'

# 获取文件夹中的所有文件名
file_names = os.listdir(folder_path)

# 设置txt文件路径
train_file_path = 'D:\Pycharm_Projects/crnn1/CRNN/data/labels/train.txt'
test_file_path = 'D:\Pycharm_Projects/crnn1/CRNN/data/labels/test.txt'


# 打开txt文件，使用追加模式

    # 将每个文件名写入txt文件的新行

for file_name in file_names:
    num = random.randint(0, 9)

    if num < 8:
        with open(train_file_path, 'a') as txt_file:
            txt_file.write(file_name + '\n')
    else:
        with open(test_file_path, 'a') as test_file:
            test_file.write(file_name + '\n')