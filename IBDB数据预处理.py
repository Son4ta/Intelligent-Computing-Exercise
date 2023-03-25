import numpy as np
import matplotlib.pyplot as plt
from keras.api.keras.preprocessing import sequence
from keras.datasets import imdb


def draw_dual_plt(array1, array2):
    plt.plot(range(len(array1)), array1, color='r', linestyle='--', label='1')
    plt.plot(range(len(array2)), array2, color='g', linestyle='-.', label='2')

    # 显示图例
    plt.legend()  # 默认loc=Best

    # 添加网格信息
    plt.grid(True, linestyle='--', alpha=0.5)  # 默认是True，风格设置为虚线，alpha为透明度

    # 添加标题
    plt.xlabel('N0.')
    plt.ylabel('count')
    # plt.title('SB')
    plt.show()


max_features = 20000
maxlen = 250
embedding_size = 128
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
print(len(x_train[0]))
print(x_train[0])
print(y_train[0])
# 设定向量的最大长度，小于这个长度的补0，大于这个长度的直接截端
# x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
# x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
# print(len(x_train[0]))
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

x_pos_vector = [1 for i in range(max_features + 1)]
x_neg_vector = [1 for i in range(max_features + 1)]

pos_count = 0
neg_count = 0
# for index in range(len(x_train)):
#     size = len(x_train[index])
#     for i in x_train[index]:
#         x_vector[index][i] += 1
#     for j in range(max_features + 1):
#         if x_vector[index][j] == 0:
#             continue
#         x_vector[index][j] /= size

# 词向量统计
for index in range(len(x_train)):
    # 单词计数
    size = len(x_train[index])
    temp_vector = [0 for i in range(max_features + 1)]
    for i in x_train[index]:
        temp_vector[i] += 1
    for i in x_train[index]:
        temp_vector[i] /= size

    if y_train[index]:
        pos_count += 1
        for i in range(max_features + 1):
            x_pos_vector[i] += temp_vector[i]
    else:
        neg_count += 1
        for i in range(max_features + 1):
            x_neg_vector[i] += temp_vector[i]
    print(str(index) + "/25000")

# 写入文件
with open('pos.npz', 'w') as f:
    for line in x_pos_vector:
        f.write(str(line / pos_count) + '\n')

with open('neg.npz', 'w') as f:
    for line in x_neg_vector:
        f.write(str(line / neg_count) + '\n')

draw_dual_plt(x_pos_vector, x_neg_vector)
