import numpy as np
from utils.features import prepare_for_training


class LinearRegression:

    def __init__(self, data, labels, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
        """
        1.对数据进行预处理操作
        2.先得到所有的特征个数
        3.初始化参数矩阵
        """
        (data_processed, features_mean, features_deviation) = prepare_for_training(
            data,
            polynomial_degree,
            sinusoid_degree,
            normalize_data=True)

        self.data = data_processed
        self.labels = labels
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data

        self.theta = np.zeros((self.data.shape[1], 1))  # 定义参数

    def train(self, alpha, num_iterations=500):
        """
        训练模块，执行梯度下降
        """
        cost_history = self.gradient_descent(alpha, num_iterations)
        return self.theta, cost_history

    def gradient_descent(self, learning_rate, num_iterations):
        """
        实际迭代模块，会迭代num_iterations次
        """
        loss_history = []
        for _ in range(num_iterations):
            self.gradient_step(learning_rate)  # 梯度下降更新参数
            loss_history.append(self.caluate_loss(self.data, self.labels))  # 计算损失
        return loss_history

    def gradient_step(self, learning_rate):
        """
        梯度下降参数更新计算方法(矩阵运算)
        """
        num_examples = self.data.shape[0]
        prediction = self.caluate_predictions(self.data, self.theta)
        delta = prediction - self.labels
        theta = self.theta
        theta -= learning_rate * (1 / num_examples) * (np.dot(delta.T, self.data)).T
        self.theta = theta

    def caluate_loss(self, data, labels):
        """
        计算损失
        """
        num_examples = data.shape[0]
        delta = self.caluate_predictions(self.data, self.theta) - labels
        cost = (1 / 2) * np.dot(delta.T, delta) / num_examples
        return cost[0][0]

    def caluate_predictions(self, data, theta):
        """
        预测
        """
        predictions = np.dot(data, theta)
        return predictions

    def get_loss(self, data, labels):
        data_processed = prepare_for_training(data,
                                              self.polynomial_degree,
                                              self.sinusoid_degree,
                                              self.normalize_data
                                              )[0]

        return self.caluate_loss(data_processed, labels)

    def predict(self, data):
        """
        用训练的参数模型，与预测得到回归值结果
        """
        data_processed = prepare_for_training(data,
                                              self.polynomial_degree,
                                              self.sinusoid_degree,
                                              self.normalize_data
                                              )[0]

        predictions = self.caluate_predictions(data_processed, self.theta)

        return predictions
