import numpy as np
from random import randint
import math
import matplotlib.pyplot as plt


def sigmoid(v):
    for i in range(v.shape[0]):
        v[i][0] = 1 / (1 + math.exp(-v[i][0]))
    return v


class MLP_manual:
    def __init__(self, lr, epoch):
        self.lr = lr
        self.epoch = epoch

        self.y1 = np.zeros([5, 1])
        self.y2 = np.zeros([4, 1])
        self.y3 = np.zeros([4, 1])
        self.y4 = np.zeros([3, 1])

        self.w1 = np.zeros([4, 5])
        self.w2 = np.zeros([4, 4])
        self.w3 = np.zeros([3, 4])

        self.b1 = np.zeros([4, 1])
        self.b2 = np.zeros([4, 1])
        self.b3 = np.zeros([3, 1])

    def train_pass_layer(self, train_d):
        self.y1 = train_d
        self.y2 = sigmoid(np.dot(self.w1, self.y1) + self.b1)
        self.y3 = sigmoid(np.dot(self.w2, self.y2) + self.b2)
        self.y4 = sigmoid(np.dot(self.w3, self.y3) + self.b3)

    def hidden(self, w_h, y_h, y_l, w_l, b_l, yita_l, yita_h):
        for i in range(yita_l.shape[0]):
            for j in range(yita_h.shape[0]):
                yita_l[i][0] += yita_h[j][0] * w_h[j][i]
            yita_l[i][0] *= y_h[i][0] * (1 - y_h[i][0])
        return w_l - self.lr * np.dot(yita_l, y_l.reshape(1, -1)), b_l - self.lr * yita_l, yita_l

    def train(self, train_data, train_label):
        train_samples = train_data.shape[0]
        label = np.zeros([train_samples, 3], dtype=int)
        for i in range(train_samples):
            label[i][train_label[i][0] - 1] = 1

        losses = []

        for _ in range(self.epoch):
            for sample in range(train_samples):
                self.train_pass_layer(train_data[sample].reshape(-1, 1))

                yita1 = np.zeros([4, 1])
                yita2 = np.zeros([4, 1])
                yita3 = np.zeros([3, 1])

                for i in range(yita3.shape[0]):
                    yita3[i][0] = label[sample][i] * (self.y4[i][0] - 1)
                self.w3 -= self.lr * np.dot(yita3, self.y3.reshape(1, -1))
                self.b3 -= self.lr * yita3
                self.w2, self.b2, yita2 = self.hidden(self.w3, self.y3, self.y2, self.w2, self.b2, yita2, yita3)
                self.w1, self.b1, yita1 = self.hidden(self.w2, self.y2, self.y1, self.w1, self.b1, yita1, yita2)

            loss = 0
            for sample in range(train_samples):
                self.train_pass_layer(train_data[sample].reshape(-1, 1))
                for i in range(3):
                    if label[sample][i] == 1:
                        loss -= math.log(self.y4[i][0], math.e)
            losses.append(loss)
        return np.array(losses)


def main():
    train_label = []
    train_sample_size = 100
    lr = 0.05
    epoch = 500
    train_data = np.random.randn(train_sample_size, 5)
    for i in range(train_sample_size):
        train_label.append(randint(1, 3))

    mlp = MLP_manual(lr, epoch)
    losses = mlp.train(train_data, np.array(train_label).reshape(-1, 1))
    plt.plot(losses)
    plt.show()


main()
