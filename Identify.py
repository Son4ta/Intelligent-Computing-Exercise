import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys


image_size = 28 * 28
image_wide = 28
threshold = 50
# 贝叶斯计数
correct_count = [0 for i in range(0, 10)]
# 贝叶斯条件概率
bayes_contingent = [np.ones(image_size) for i in range(0, 10)]
#
bayes_posterior = [np.ones(image_size) for i in range(0, 10)]
# 贝叶斯计数
bayes_count = [0 for i in range(0, 10)]
# 贝叶斯先验数组
bayes_prior = [0 for i in range(0, 10)]
# 贝叶斯分子
bayes_ans = [0 for i in range(0, 10)]
bayes_avg = np.zeros(image_size)


with open('data/train-images.idx3-ubyte', 'rb') as f:
    train_cache = f.read()
with open('data/train-labels.idx1-ubyte', 'rb') as f:
    train_label_cache = f.read()
with open('data/t10k-images.idx3-ubyte', 'rb') as f:
    test_cache = f.read()
with open('data/t10k-labels.idx1-ubyte', 'rb') as f:
    test_label_cache = f.read()


def test_load_by_cv2():
    # 测试函数

    # print(bayes_prior)
    bayes_init(8000)
    # cv_show_array(bayes_contingent[0])
    # print(bayes_contingent[0])
    scrutiny(bayes, 10000)
    # draw_plt(bayes_contingent[0])
    draw_plt(correct_count, 'bayes')
    # print(bayes(process_input(cv2.imread("image/4.jpg"))))
    # print(bayes_count)
    # print(bayes_prior[9])
    # euclidean(process_input(cv2.imread("image/6.jpg")))
    return 0


# 画图象 给一维数组
def draw_plt(array, title=''):
    plt.figure(figsize=(12, 8))
    plt.title(title)
    plt.plot(range(len(array)), array)
    plt.grid(True, linestyle='--', alpha=0.5)
    # plt.savefig('./plt_png/test1.2.png')
    plt.show()


# 计算贝叶斯矩阵 0.8420842084208421
def bayes(image):
    max_prob = 0
    max_index = 0
    for i in range(len(image)):
        # 二值化 只有0与1
        if image[i] > threshold:
            image[i] = 1
        else:
            image[i] = 0
    for i in range(10):
        prob = bayes_probability(image, bayes_contingent[i])
        bayes_ans[i] = prob * bayes_prior[i]
        # print(prob)
        if max_prob < prob:
            max_index = i
            max_prob = prob
    # print(max_index)

    # 贝叶斯全概率 没差 可能因为数据集是平均的
    # bayes_total = sum(bayes_ans)
    # for i in range(10):
    #     bayes_ans[i] /= bayes_total
    # return bayes_ans.index(max(bayes_ans)), 0

    if max_prob > 0:
        return max_index, max_prob
    else:
        return -1, max_prob


def process_input(image):
    # 处理小
    image = resize(image, 700)
    # 灰度
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 滤波
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    cv_show(gray)
    # 二值化
    binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)[1]
    # 对binary去噪，腐蚀与膨胀
    binary = cv2.erode(binary, None, iterations=2)
    binary = cv2.dilate(binary, None, iterations=3)
    cv_show(binary)
    # 边缘检测算子 并膨胀封闭
    edge = cv2.Canny(binary, 75, 200)
    edge = cv2.dilate(edge, None, iterations=1)
    cv_show(edge)
    # 边缘检测 取出最大边缘
    contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
    # 描绘最大边缘
    x, y, w, h = cv2.boundingRect(contours[0])
    cv2.rectangle(image, (x, y), (x + w, y + h), (153, 153, 0), 5)
    cv_show(image)
    # 切割
    binary = binary[y + 2:y + h - 2, x + 2:x + w - 2]  # 先用y确定高，再用x确定宽
    # 填充边缘 填充成方形
    border_h = int(h/5)
    border_w = int((h + border_h * 2 - w)/2)
    binary = cv2.copyMakeBorder(binary, border_h, border_h, border_w, border_w, cv2.BORDER_CONSTANT)
    cv_show(binary)
    # 重置大小
    binary = cv2.resize(binary, (image_wide, image_wide), cv2.INTER_AREA)
    # binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)[1]
    cv_show(binary)
    # 一维化行向量
    data = binary.copy().flatten()
    # 返回一维数组
    return data


# 验证集 参数是 识别模式 规模 验证集偏置
def scrutiny(method, size=100, offset=0):
    account = 0
    success = 0
    failure = 0
    refuse = 0
    count = [0 for i in range(0, 10)]
    i = 16 + offset*image_size
    # -1000是为了i不超限，毕竟几万张少一张没关系
    while i <= len(test_cache) - 1000:
        # 读取文件 HEX转BIN
        image = [item for item in test_cache[i:i + image_size]]
        # TODO 用注入形式调用
        result, value = method(image)
        # result, distance = euclidean(image)
        # result = bayes(image)
        answer = test_label(account + offset)
        count[answer] += 1
        if result == answer:
            correct_count[answer] += 1
            success += 1
        elif result == -1:
            refuse += 1
        else:
            failure += 1
        i += image_size
        account += 1
        if account > size:
            break
    print(success/(account - refuse))
    print(refuse)
    for i in range(len(correct_count)):
        correct_count[i] /= count[i]
    return success/(account - refuse)


# 识别主函数 传入一维向量 返回结果和距离
def euclidean(image):
    # 距离 缺省值为int最大
    distance = sys.maxsize
    # 最佳匹配图像
    best_matching = None
    # 最佳匹配图像的i值↓
    best_matching_count = None
    # 循环控制器
    # TODO: 这里的i记得改回去
    i = 16
    # -1000是为了i不超限，毕竟几万张少一张没关系
    while i <= len(train_cache) - 1000:
        # 读取文件 HEX转BIN
        image_template = [item for item in train_cache[i:i + image_size]]
        dis_temp = euclidean_distance(image, image_template)
        # 判断距离是否更优
        if dis_temp < distance:
            distance = dis_temp
            best_matching = image_template
            best_matching_count = i
            # 距离小于一定阈值直接退出 经验建议 1900-1600
            if distance < 1400:
                break
        # 文件图像步长为 28x28=image_size
        i += image_size
    result = train_label(int((best_matching_count - 16) / image_size))
    # cv_show_array(best_matching)
    print("匹配结果：" + str(result))
    print("欧氏距离：" + str(distance))
    # 拒绝识别
    if distance > 1900:
        return -1, -1
    return result, distance


# 计算贝叶斯矩阵
def bayes_init(size=60000):
    # 打开样本文件
    global bayes_avg
    # 循环控制器
    # TODO: 这里的i记得改回去
    i = 16
    account = 0
    # -1000是为了i不超限，毕竟几万张少一张没关系
    while i <= len(train_cache) - 1000:
        # 读取文件 HEX转BIN
        image_template = [item for item in train_cache[i:i + image_size]]
        for index in range(len(image_template)):
            # 二值化 只有0与1
            if image_template[index] > threshold:
                image_template[index] = 1
            else:
                image_template[index] = 0
        label = train_label(account)
        # if bayes_count[label] > 1775:
        #     i += image_size
        #     account += 1
        #     continue
        # 操作贝叶斯数组
        bayes_contingent[label] += image_template
        bayes_count[label] += 1
        # 文件图像步长为 28x28=image_size
        i += image_size
        account += 1
        if account > size:
            break
    # 贝叶斯条件概率计算
    for i in range(10):
        bayes_contingent[i] = bayes_contingent[i] / bayes_count[i]
    # 先验
    for i in range(10):
        bayes_prior[i] = bayes_count[i] / account
    # print(bayes_contingent)
    # # 贝叶斯全
    # for i in range(10):
    #     bayes_total += bayes_contingent[i] * (bayes_count[i] / size)
    # # 贝叶斯后验概率计算
    # for i in range(10):
    #     bayes_posterior[i] = bayes_contingent[i] / bayes_total
    for i in range(10):
        bayes_avg += bayes_contingent[i]
    bayes_avg /= size


# 展示函数
def cv_show(img, name="default"):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 展示函数plus 参数为一维向量list
def cv_show_array(img, name="default"):
    cv2.destroyAllWindows()
    img = img.reshape(28, 28)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 计算欧式距离，返回距离量 参数为两个一维向量list
def euclidean_distance(z, a):
    vector_z = np.array(z)
    vector_a = np.array(a)
    dif = vector_z - vector_a
    return math.sqrt(np.dot(dif.T, dif))


# 计算贝叶斯后验概率 z是待预测的图像 a是预处理产生的条件概率矩阵 a z都是行向量
def bayes_probability(z, a):
    vector_z = np.array(z)
    vector_a = np.array(a)
    ans = 1
    for i in range(len(vector_z)):
        if vector_z[i] == 1:
            ans *= vector_a[i]
        else:
            ans *= 1-vector_a[i]
    return ans


# 0~n-1
def train_label(i):
    # 第一张图片从第9个比特开始
    i += 8
    return int(train_label_cache[i])


# 0~n-1
def test_label(i):
    # 第一张图片从第9个比特开始
    i += 8
    return int(test_label_cache[i])


# 用于自动缩放图片
def resize(img, height=1000):
    width = int((height / img.shape[0]) * img.shape[1])
    return cv2.resize(img, (width, height), cv2.INTER_AREA)
