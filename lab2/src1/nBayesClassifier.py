import numpy as np
import math
from collections import Counter
from process_data import load_and_process_data
from evaluation import get_micro_F1, get_macro_F1, get_acc

ran13 = list(range(1, 4))
ran17 = list(range(1, 8))


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
            feature_l[(int(trainlabel[line]), int(traindata[line][0]))] += 1

            for j in ran17:
                if (int(trainlabel[line]), j) not in subset_flag:
                    subset_dict[(int(trainlabel[line]), j)] = np.array(float(traindata[line][j]))
                    subset_flag[(int(trainlabel[line]), j)] = 0
                else:
                    subset_dict[(int(trainlabel[line]), j)] = np.append(
                        subset_dict[(int(trainlabel[line]), j)],
                        float(traindata[line][j]))
        for i in ran13:
            self.Pc[i] = (categ_l[i] + 1) / (categ_l[1] + categ_l[2] + categ_l[3] + 3)
        for i in ran13:
            for j in range(8):
                if featuretype[j] == 0:
                    for k in ran13:
                        self.Pxc[(i, j, k)] = (feature_l[i, k] + 1) / (categ_l[i] + 3)
                elif featuretype[j] == 1:
                    self.Pxc[(i, j)] = (np.average(subset_dict[(i, j)]), np.sqrt(np.var(subset_dict[(i, j)])))
                else:
                    print("err featuretype")

    def predict(self, features, featuretype):
        '''
        需要你实现的部分
        '''
        pre_out = []
        for line in range(features.shape[0]):
            max = 0
            pre = 0
            for c in ran13:
                bayes_p = self.Pxc[(c, 0, int(features[line][0]))] * self.Pc[c]
                for i in ran17:
                    mean, sigma = self.Pxc[(c, i)]
                    bayes_p *= math.exp(-0.5 * ((features[line][i] - mean) / sigma) ** 2) / (
                            ((2 * math.pi) ** 0.5) * sigma)
                if bayes_p > max:
                    max = bayes_p
                    pre = c

            pre_out.append(pre)
        pred = np.array(pre_out).reshape(-1, 1)
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
