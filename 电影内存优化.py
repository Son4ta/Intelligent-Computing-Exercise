import random
import numpy as np
import time
import matplotlib.pyplot as plt
from keras.datasets import imdb

# 史官
# 每一代最好的孩子
best = []

# 此量需要小于等于预处理 建议 2000左右 又快又准! 此量也是基因长度
# PS: 经过实验,预处理时的分母,即max_features其实无所谓
# PPS: 经过实验,预处理时maxlen也没用,250左右就行
max_features = 2200

# 测试集 在实验中发现它很慢于是放在这
(x_test, y_test) = imdb.load_data(num_words=max_features)[1]
# 词向量 在实验中发现它有点慢于是放在这
x_test_vector = [0 for i in range(max_features + 1)]

# 读取文件预处理好的贝叶斯概率
x_pos_vector = []
x_neg_vector = []

# 测试集大小
test_size = 250

# 遗传算法 全局变量 种群 适应度矩阵 自然选择概率（轮盘赌） 迭代轮数 种群数量 交叉互换点位数 保留优秀种率 变异率 变异基因占比
# 种群 0 1代表选取

adaptability = []
selection_probability = []
# 迭代轮数
iteratorNum = 50
# 种群数量
population = 100
# 交叉互换点位数
cross_points = 4
# 保留优秀种率
cp = 0.2
# 变异基因占比
_mp = 0.2
# 变异率 类似学习速率，小点好，目前0.001效果很好
_mr = 0.001
mp = _mp
mr = _mr

# 自动变量
# 优秀种保留比
copy_num = int(population * cp)
# 交叉互换产生新一代数
cross_num = population - copy_num
mutate_rate = int(max_features * mr)
mutate_num = int(population * mp)

with open('pos.npz', 'r') as f:
    Lines = f.readlines()
    for line in Lines:
        x_pos_vector.append(float(line.split()[0]))

with open('neg.npz', 'r') as f:
    Lines = f.readlines()
    for line in Lines:
        x_neg_vector.append(float(line.split()[0]))


def main():
    GA()
    draw_plt(best, generate_title())
    return


def generate_title():
    return ("no kill modify:"
            + " max_features:" + str(max_features) + " population:" + str(population)
            + " iteratorNum:" + str(iteratorNum) + " mp:" + str(_mp)
            + " mr:" + str(_mr) + " cp:" + str(cp) + " test_size:" + str(test_size)
            + " cross_points:" + str(cross_points) + " best:" + str(max(best)) + " %")


# 审查测试函数 返回准确率  最近最佳记录:0.83084(25000样本) 2022年10月30日16点55分
# 已经遗传算法改造,要用就让method = gene_bayes
def scrutiny(method, size=25000, gene=None):
    # 计数
    count = 0
    # 正确计数
    correct_count = 0
    # 二值化 顺便设置测试规模
    for index in range(len(x_test)):
        count += 1
        if count > size:
            break
        for i in x_test[index]:
            x_test_vector[i] = 1

        # 调用method计算概率
        if y_test[index] == method(x_test_vector, gene):
            correct_count += 1

        # 置零数组准备预测下一个样本，恢复现场
        for i in range(max_features + 1):
            x_test_vector[i] = 0

        # print(str(count) + '/' + str(size))
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
    generation = create_generation()
    # 开始演化 iteratorNum代
    for i in range(iteratorNum):
        calculate_adaptability(generation)
        calculate_selection_probability()
        generation = next_generation(generation)
        index = adaptability.index(max(adaptability))
        print("第" + str(i + 1) + "代已生成，最优个体:" + str(index + 1))
        best.append(max(adaptability))
        modify(index)
        print("当前变异率:" + str(mr) + " 变异比例:" + str(mp))


# 根据学习情况调整变异 如果优势种没变，则编号会为0
# 不应该增大mr,反而应该在学习速率变小的时候适当减小
def modify(index):
    global mp, mr
    if not index:
        mp += 0.05
        mr -= 0.0001
    else:
        mp = _mp
        mr = _mr
    if mp > 0.6:
        mp = 0.6
    if mr < 0.0005:
        mr = 0.0005


def next_generation(generation):
    return mutate(copy(generation) + cross(generation))


# 变异 默认选一条变异，变异率在上面调
def mutate(generation):
    for j in range(mutate_num):
        # 变异的个体下标
        index = random.randint(0, population - 1)
        # 不能变异最好的
        while index == 0:
            index = random.randint(0, population - 1)
        # 执行变异
        for i in range(mutate_rate):
            # 变异位点
            point = random.randint(0, max_features - 1)
            if generation[index][point]:
                generation[index][point] = 0
            else:
                generation[index][point] = 1
    return generation


# 保留优秀种
def copy(generation):
    # 排序并添加下标
    temp = [[adaptability[i], i] for i in range(population)]
    temp.sort(reverse=True)
    superior = []
    for i in range(copy_num):
        # 按照上面得到的下标取个体
        superior.append(generation[temp[i][1]])
    return superior


# 交叉
def cross(generation):
    children = []
    # for i in range(cross_points):
    for j in range(cross_num):
        point = random.randint(0, max_features - 1)
        dad = generation[RWS()]
        mom = generation[RWS()]
        child = np.concatenate((dad[:point], mom[point:]), axis=0)
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
    selection_probability.clear()
    # 优势很微弱，直接轮盘赌不合适
    sum_adaptability = 0
    for item in adaptability:
        sum_adaptability += item
    for i in range(population):
        selection_probability.append(adaptability[i] / sum_adaptability)
    # 所以直接干掉可能性在0.1以下的个体
    # 但是发现会同质化，爸爸和孩子和妈妈太像了，所以还是算了
    # sum_adaptability = 0
    # for i in range(population):
    #     if selection_probability[i] < 1 / population:
    #         selection_probability[i] = 0
    #         continue
    #     sum_adaptability += selection_probability[i]
    # for i in range(population):
    #     selection_probability[i] = selection_probability[i] / sum_adaptability

    print(selection_probability)


# 适应度计算，参数为计算的测试集规模
def calculate_adaptability(generation):
    # 清空适应度矩阵
    adaptability.clear()
    size = len(generation)
    for i in range(size):
        # 调用scrutiny，准确率作为适应度
        adaptability.append(scrutiny(gene_bayes, test_size, generation[i]))
        print(str(i + 1) + "/" + str(size) + ":" + str(adaptability[i]))


def create_generation():
    return np.random.randint(2, size=(population, max_features))
    # return [[random.randint(0, 1) for i in range(max_features)] for i in range(population)]


# 画图象工具↓
def draw_plt(array, title=''):
    plt.figure(figsize=(12, 8))
    plt.title(title)
    plt.plot(range(len(array)), array)
    plt.grid(True, linestyle='--', alpha=0.5)
    # plt.savefig('./plt_png/test1.2.png')
    plt.savefig("./result/" + time.strftime('%Y-%m-%d-%H-%M-%S') + ".png")
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
