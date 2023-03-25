import random
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import imdb

# 此量需要小于等于预处理 建议 2000左右 又快又准! 此量也是基因长度
# PS: 经过实验,预处理时的分母,即max_features其实无所谓
# PPS: 经过实验,预处理时maxlen也没用,250左右就行,闪避
max_features = 2000

# 测试集 在实验中发现它很慢于是放在这
(x_test, y_test) = imdb.load_data(num_words=max_features)[1]
# 词向量 在实验中发现它有点慢于是放在这
x_test_vector = [0 for i in range(max_features + 1)]

# 读取文件预处理好的贝叶斯概率
x_pos_vector = []
x_neg_vector = []

# 遗传算法 全局变量 种群 适应度矩阵 自然选择概率（轮盘赌） 迭代轮数 种群数量 保留优秀种率 变异率
# 种群 0 1代表选取
generation = []
adaptability = []
selection_probability = []
iteratorNum = 30
population = 10
cp = 0.2
mp = 0.1

# 自动变量
# 优秀种保留比
copy_num = int(population * cp)
# 交叉互换产生新一代数
cross_num = population - copy_num
mutate_num = int(max_features * mp)

with open('pos.npz', 'r') as f:
    Lines = f.readlines()
    for line in Lines:
        x_pos_vector.append(float(line.split()[0]))

with open('neg.npz', 'r') as f:
    Lines = f.readlines()
    for line in Lines:
        x_neg_vector.append(float(line.split()[0]))


def main():
    # draw_dual_plt(x_pos_vector, x_neg_vector)
    # print(scrutiny(gene_bayes, 2000, genes[0]))
    # create_generation()
    GA()
    return


# 审查测试函数 返回准确率  最近最佳记录:0.83084(25000样本) 2022年10月30日16点55分
# 已经遗传算法改造,要用就让method = gene_bayes
def scrutiny(method, test_size=25000, gene=None):
    # 计数
    count = 0
    # 正确计数
    correct_count = 0
    # 二值化 顺便设置测试规模
    for index in range(len(x_test)):
        count += 1
        if count > test_size:
            break
        for i in x_test[index]:
            x_test_vector[i] = 1

        # 调用method计算概率
        if y_test[index] == method(x_test_vector, gene):
            correct_count += 1

        # 置零数组准备预测下一个样本，恢复现场
        for i in range(max_features + 1):
            x_test_vector[i] = 0

        # print(str(count) + '/' + str(test_size))
    return correct_count / count


# 贝叶斯 返回判断 0 1
def bayes(test, gene=None):
    size = len(test)
    pos = 1
    neg = 1
    # 无基因调用
    for i in range(size):
        if test[i]:
            pos *= x_pos_vector[i] * test[i]
            neg *= x_neg_vector[i] * test[i]
        else:
            pos *= (1 - x_pos_vector[i])
            neg *= (1 - x_neg_vector[i])
    if pos > neg:
        return 1
    return 0


# 0.8345827086456772
# 转基因贝叶斯 返回判断 0 1
def gene_bayes(test, gene=None):
    pos = 1
    neg = 1
    # 只因测试
    # for i in range(200):
    #     gene[i] = 0
    for i in range(max_features):
        if not gene[i]:
            continue
        if test[i]:
            pos *= x_pos_vector[i] * test[i]
            neg *= x_neg_vector[i] * test[i]
        else:
            pos *= (1 - x_pos_vector[i])
            neg *= (1 - x_neg_vector[i])
    if pos > neg:
        return 1
    return 0


# 遗传算法
def GA():
    print("开始执行")
    # 初始种群
    create_generation()
    # 开始演化 iteratorNum代
    for i in range(iteratorNum):
        calculate_adaptability(1000)
        calculate_selection_probability()
        next_generation()


def next_generation():
    global generation
    generation = cross() + copy()
    mutate()


# 变异 默认选一条变异，变异率在上面调
def mutate():
    # 变异的个体下标
    index = random.randint(0, population-1)
    for i in range(mutate_num):
        # 变异位点
        point = random.randint(0, max_features-1)
        if generation[index][point]:
            generation[index][point] = 0
        else:
            generation[index][point] = 1


# 保留优秀种
def copy():
    # 排序并添加下标
    temp = [[adaptability[i], i] for i in range(population)]
    temp.sort(reverse=True)
    superior = []
    for i in range(copy_num):
        # 按照上面得到的下标取个体
        superior.append(generation[temp[i][1]])
    return superior


# 交叉
def cross():
    children = []
    for i in range(cross_num):
        index = random.randint(1, max_features-1)
        dad1 = generation[RWS()]
        dad2 = generation[RWS()]
        child = dad1 + dad2
        children.append(child)
    return children


# roulette wheel selection轮盘赌
def RWS():
    # 轮盘指针
    pointer = random.random()
    # 概率和
    count = 0
    for i in range(population):
        count += selection_probability[i]
        if count > pointer:
            return i


# 计算自然选择概率矩阵
def calculate_selection_probability():
    sum_adaptability = 0
    for item in adaptability:
        sum_adaptability += item
    for i in range(population):
        selection_probability.append(adaptability[i] / sum_adaptability)


# 适应度计算，参数为计算的测试集规模
def calculate_adaptability(test_size=2000):
    # 清空适应度矩阵
    adaptability.clear()
    size = len(generation)
    for i in range(size):
        # 调用scrutiny，准确率作为适应度
        adaptability.append(scrutiny(gene_bayes, test_size, generation[i]))
        print(str(i + 1) + "/" + str(size) + ":" + str(adaptability[i]))


def create_generation():
    global generation
    generation = [[random.randint(0, 1) for i in range(max_features)] for i in range(population)]
    # return generation


# 画图象工具↓
def draw_plt(array, title=''):
    plt.figure(figsize=(12, 8))
    plt.title(title)
    plt.plot(range(len(array)), array)
    plt.grid(True, linestyle='--', alpha=0.5)
    # plt.savefig('./plt_png/test1.2.png')
    plt.show()


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
    plt.title('SB')
    plt.show()


if __name__ == '__main__':
    main()
