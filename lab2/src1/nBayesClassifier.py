import numpy as np
import math
from collections import Counter
from process_data import load_and_process_data
from evaluation import get_micro_F1, get_macro_F1, get_acc


def norm_distribution_function(mean, s_d, x):
    temp = ((2 * math.pi) ** 0.5) * s_d
    temp = 1 / temp
    exp = math.exp(-0.5 * ((x - mean) ** 2) / (s_d ** 2))
    temp = temp * exp
    return temp


def mean_and_standard_deviation(data_subset):
    mean = np.average(data_subset)
    s_d = np.sqrt(np.var(data_subset))
    return (mean, s_d)


class NaiveBayes:
    '''参数初始化
    Pc: P(c) 每个类别c的概率分布
    Pxc: P(c|x) 每个特征的条件概率
    '''

    def __init__(self):
        self.Pc = {}
        self.Pxc = {}

    '''
    通过训练集计算先验概率分布p(c)和条件概率分布p(x|c)
    建议全部取log，避免相乘为0
    '''

    def fit(self, traindata, trainlabel, featuretype):
        '''
        需要你实现的部分
        '''
        '''
            1:number of category
            2:建立subset array with a certain category and a certain feature
            3:for discrete feature[0],我们应该统计pxc
        '''
        ran13 = list(range(1, 4))
        ran17 = list(range(1, 8))
        categ_l = {}  # label
        feature_l = {}  # (label, data[0])
        subset_dict = {}
        subset_flag = {}
        for i in ran13:
            categ_l[i] = 0
            for k in ran13:
                feature_l[(i, k)] = 0

        for line in range(traindata.shape[0]):
            categ_l[int(trainlabel[line])] += 1
            # 更新相应种类训练数据数量数
            feature_l[(int(trainlabel[line]), int(traindata[line][0]))] += 1
            # 为离散特征feature0和不同种类统计训练数据数量

            for j in ran17:
                if (int(trainlabel[line]), j) not in subset_flag:
                    subset_dict[(int(trainlabel[line]), j)] = np.array(float(traindata[line][j]))
                    subset_flag[(int(trainlabel[line]), j)] = 0
                else:
                    subset_dict[(int(trainlabel[line]), j)] = np.append(
                        subset_dict[(int(trainlabel[line]), j)],
                        float(traindata[line][j]))
    # todo: save
    # PC
        for i in ran13:
            self.Pc[i] = (categ_l[i] + 1) / (categ_l[1] + categ_l[2] + categ_l[3] + 3)
        # 计算Px[0]c
        for i in ran13:
            for j in range(8):
                if featuretype[j] == 0:
                    for k in ran13:
                        self.Pxc[(i, j, k)] = (feature_l[i, k] + 1) / (categ_l[i] + 3)
                elif featuretype[j] == 1:
                    self.Pxc[(i, j)] = mean_and_standard_deviation(subset_dict[(i, j)])

    '''
    通过一个数据子集去计算均值和标准差
    data_subset是一个数组（array）类型数据
    '''

    '''
    根据先验概率分布p(c)和条件概率分布p(x|c)对新样本进行预测
    返回预测结果,预测结果的数据类型应为np数组，shape=(test_num,1) test_num为测试数据的数目
    feature_type为0-1数组，表示特征的数据类型，0表示离散型，1表示连续型
    '''

    def predict(self, features, featuretype):
        '''
        需要你实现的部分
        '''
        pred = []
        test_num = features.shape[0]
        for k in range(test_num):
            max = 0
            c_predict = 0
            probabilities = []
            for c in range(1, 4):
                temp = self.Pc[c]
                temp *= self.Pxc[(c, 0, int(features[k][0]))]
                for i in range(1, 8):
                    (mean, s_d) = self.Pxc[(c, i)]
                    p = norm_distribution_function(mean, s_d, features[k][i])
                    temp *= p
                '''
                probabilities.append(temp)
                c_predict = np.argmax(probabilities)
                '''

                if temp > max:
                    max = temp
                    c_predict = c

            pred.append(c_predict)
        pred = np.array(pred).reshape(test_num, 1)
        return pred


def main():
    # 加载训练集和测试集
    train_data, train_label, test_data, test_label = load_and_process_data()
    feature_type = [0, 1, 1, 1, 1, 1, 1, 1]  # 表示特征的数据类型，0表示离散型，1表示连续型

    Nayes = NaiveBayes()
    Nayes.fit(train_data, train_label, feature_type)  # 在训练集上计算先验概率和条件概率

    pred = Nayes.predict(test_data, feature_type)  # 得到测试集上的预测结果

    # 计算准确率Acc及多分类的F1-score
    print("Acc: " + str(get_acc(test_label, pred)))
    print("macro-F1: " + str(get_macro_F1(test_label, pred)))
    print("micro-F1: " + str(get_micro_F1(test_label, pred)))


main()
