from process_data import load_and_process_data
from evaluation import get_macro_F1, get_micro_F1, get_acc
import numpy as np


# 实现线性回归的类
class LinearClassification:
    '''参数初始化
    lr: 梯度更新的学习率
    Lambda: L2范数的系数
    epochs: 更新迭代的次数
    '''

    def __init__(self, lr=0.05, Lambda=0.001, epochs=1000, w=0):
        self.lr = lr
        self.Lambda = Lambda
        self.epochs = epochs
        self.w = w

    '''
    根据训练数据train_features,train_labels计算梯度更新参数W
    '''

    def fit(self, train_features, train_labels):
        ''''
        需要你实现的部分
        '''
        attachment = np.ones(train_features.shape[0])
        print("train_features.shape:", train_features.shape)
        x = np.c_[attachment, train_features]
        print("x:", x)
        w = np.zeros(train_features.shape[1] + 1).reshape(-1, 1)

        for i in range(self.epochs):
            temp = np.dot(x, w)
            temp = temp - train_labels
            temp = np.dot(temp.reshape(temp.shape[1], temp.shape[0]), x)
            grad = 2 * temp + 2 * self.Lambda * w.reshape(1, -1)
            # print(grad)
            # 计算梯度
            w = w - self.lr * grad.reshape(-1, 1)
        # print(w)
        self.w = w

    '''
    根据训练好的参数对测试数据test_features进行预测，返回预测结果
    预测结果的数据类型应为np数组，shape=(test_num,1) test_num为测试数据的数目
    '''

    def predict(self, test_features):
        ''''
        需要你实现的部分
        '''
        test_num = test_features.shape[0]
        attachment = np.ones(test_features.shape[0])
        X = np.c_[attachment, test_features]
        pred = []
        for i in range(test_num):
            y_pred = np.dot(X[i], self.w)
            # print(y_pred.shape,X[i].shape,self.w.shape)
            if y_pred < 1.5:
                pred.append(1)
            elif y_pred > 2.5:
                pred.append(3)
            else:
                pred.append(2)
        pred = np.array(pred).reshape(test_num, 1)
        return pred


def main():
    # 加载训练集和测试集
    train_data, train_label, test_data, test_label = load_and_process_data()

    lR = LinearClassification(lr=0.000005)
    lR.fit(train_data, train_label)  # 训练模型
    pred = lR.predict(test_data)  # 得到测试集上的预测结果

    # 计算准确率Acc及多分类的F1-score
    print("Acc: " + str(get_acc(test_label, pred)))
    # todo: err F1
    '''
    Acc: 0.612410986775178
    TP: 106 17 112
    0.6217008797653959
    TP: 288 288 89
    0.6044071353620146
    TP: 208 76 180
    0.6190476190476191
    macro-F1: 0.6150518780583432
    TP: 602 381 381
    micro-F1: 0.612410986775178
    '''
    print("macro-F1: " + str(get_macro_F1(test_label, pred)))
    print("micro-F1: " + str(get_micro_F1(test_label, pred)))


main()
