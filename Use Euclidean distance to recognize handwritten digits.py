import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# 读取文件,返回读取的训练集和测试集文件
def open_file():
    with open('data/train-images.idx3-ubyte', 'rb') as train_image:
        train_image_file = train_image.read()
    with open('data/t10k-images.idx3-ubyte', 'rb') as test_image:
        test_image_file = test_image.read()
    with open('data/train-labels.idx1-ubyte', 'rb') as train_label:
        train_label_file = train_label.read()
    with open('data/t10k-labels.idx1-ubyte', 'rb') as test_label:
        test_label_file = test_label.read()
    file_cache = [train_image_file, test_image_file, train_label_file, test_label_file]
    return file_cache


# 读入训练集并得到相应信息，返回训练集数据信息的cache
def get_train_info(file_cache):
    train_img_number = int(file_cache[0][4:8].hex(), 16)
    h_train_image = int(file_cache[0][8:12].hex(), 16)
    w_train_image = int(file_cache[0][12:16].hex(), 16)
    train_image_size = h_train_image * w_train_image
    train_label_number = int(file_cache[2][4:8].hex(), 16)
    train_cache = [train_img_number, train_image_size, train_label_number]
    return train_cache


# 读入测试集并得到相应信息，返回测试集数据信息的cache
def get_test_info(file_cache):
    test_img_number = int(file_cache[1][4:8].hex(), 16)
    h_test_image = int(file_cache[1][8:12].hex(), 16)
    w_test_image = int(file_cache[1][12:16].hex(), 16)
    test_image_size = h_test_image * w_test_image
    test_label_number = int(file_cache[3][4:8].hex(), 16)
    test_cache = [test_img_number, test_image_size, test_label_number]
    return test_cache


# 将训练集图片信息以一每一份图片切分，返回切分后训练集图像的列表(一维)
def sort_train_image(train_cache, file_cache):
    i = 16
    train_images = []
    for j in range(train_cache[0]):
        image = [item for item in file_cache[0][i:i + train_cache[1]]]
        i = i + train_cache[1]
        train_images.append(image)
    return train_images


# 将训练集标签信息以一每一份图片切分，返回切分后训练集标签的列表(一维)

def sort_train_label(train_cache, file_cache):
    i = 8
    train_labels = []
    for j in range(train_cache[2]):
        label = file_cache[2][i+j]
        train_labels.append(label)
    return train_labels


# 将测试集以一每一份图片切分，返回切分后测试集的列表(一维)
def sort_test_image(test_cache, file_cache):
    i = 16
    test_images = []
    for n in range(test_cache[0]):
        image = [item for item in file_cache[1][i:i + test_cache[1]]]
        i = i + test_cache[1]
        test_images.append(image)
    return test_images


# 将训练集标签信息以一每一份图片切分，返回切分后训练集标签的列表(一维)
def sort_test_label(test_cache, file_cache):
    i = 8
    test_labels = []
    for j in range(test_cache[2]):
        label = file_cache[3][i+j]
        test_labels.append(label)
    return test_labels


# 计算欧式距离
def get_dist(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    temp_dist = v1 - v2
    dist = math.sqrt(np.dot(temp_dist.T, temp_dist))
    print(dist)
    return dist


# 识别结果
def end_result(train_images, test_images, train_cache, test_cache):
    label_count = 0
    min_dist = 0
    for i in range(test_cache[0]):
        for j in range(train_cache[0]):
            dist = get_dist(test_images[i], train_images[j])
            print(dist)
            label_count += 1
            if dist < min_dist:
                min_dist = dist


file_cache = open_file()
train_cache = get_train_info(file_cache)
test_cache = get_test_info(file_cache)
train_images = sort_train_image(train_cache, file_cache)
train_labels = sort_train_label(train_cache, file_cache)
test_images = sort_test_image(test_cache, file_cache)
test_labels = sort_test_label(test_cache, file_cache)
end_result(train_images, test_images, train_cache, test_cache)








'''
res1 = 0  # 识别成功
    res2 = 0  # 拒绝识别
    for i in range(train_cache[0]):
        for j in range(test_cache[0]):
            if get_dist(train_images[j], test_images[i]) < 3000:
                res1 = res1 + 1
                break
            elif get_dist(train_images[j], test_images[i]) > 5000:
                res2 = res2 + 1
                break
    accuracy1 = (res1 / test_cache[0]) * 100  # 成功
    accuracy2 = (res2 / test_cache[0]) * 100  # 拒绝
    accuracy3 = (1 - accuracy1 - accuracy2) * 100  # 失败
    print("成功识别率：" + str(accuracy1) + "%")
    print("拒绝识别率：" + str(accuracy2) + "%")
    print("错误识别率：" + str(accuracy3) + "%")


def load_mnist(mnist_image_train_image_file, mnist_label_train_image_file):
    with open(mnist_image_train_image_file, 'rb') as f1:
        image_train_image_file = np.frombuffer(f1.read(), np.uint8, offset=16).reshape(-1, 28 * 28)
    with open(mnist_label_train_image_file, 'rb') as f2:
        label_train_image_file = np.frombuffer(f2.read(), np.uint8, offset=8)
    img = Image.fromarray(image_train_image_file[4].reshape(28, 28))  # First image in the training set.
    img.show()  # Show the image


if __name__ == '__main__':
    train_image_train_image_file = './train-images.idx3-ubyte'
    train_label_train_image_file = './train-labels.idx1-ubyte'

    load_mnist(train_image_train_image_file, train_label_train_image_file)
'''

