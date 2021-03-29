import numpy as np
from scipy.optimize import minimize
from utils.features import prepare_for_training
from utils.hypothesis import sigmoid


class LogisticRegression:
    """
    通过sigmoid函数将多分类问题转换为多个二分类问题
    """
    def __init__(self, data, labels, polynomial_degree=0, sinusoid_degree=0, normalize_data=False):
        """
        1.对数据进行预处理操作
        2.先得到所有的特征个数
        3.初始化参数矩阵
        """
        (data_processed,
         features_mean,
         features_deviation) = prepare_for_training(data, polynomial_degree, sinusoid_degree, normalize_data=False)

        self.data = data_processed
        self.labels = labels
        self.unique_labels = np.unique(labels)
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data

        num_features = self.data.shape[1]
        num_unique_labels = self.unique_labels.shape[0]

        # 参数为二维矩阵，行为标签种类数目，列为参数
        self.theta = np.zeros((num_unique_labels, num_features))

    def train(self, max_iterations=1000):
        loss_histories = []
        num_features = self.data.shape[1]
        for label_index, unique_label in enumerate(self.unique_labels):
            current_initial_theta = np.copy(self.theta[label_index].reshape(num_features, 1))
            current_lables = (self.labels == unique_label).astype(float)
            (current_theta, loss_history) = self.gradient_descent(self.data,
                                                                  current_lables,
                                                                  current_initial_theta,
                                                                  max_iterations)
            self.theta[label_index] = current_theta.T
            loss_histories.append(loss_history)

        return self.theta, loss_histories

    def gradient_descent(self, data, labels, current_initial_theta, max_iterations):
        loss_history = []
        num_features = data.shape[1]
        result = minimize(
            # 优化目标（即损失函数）：
            lambda current_theta: self.calculate_loss(data, labels, current_theta.reshape(num_features, 1)),
            # 初始化的权重参数
            current_initial_theta,
            # 选择优化策略
            method='CG',
            # 梯度下降迭代计算公式
            jac=lambda current_theta: self.gradient_step(data, labels, current_theta.reshape(num_features, 1)),
            # 记录结果
            callback=lambda current_theta: loss_history.append(
                self.calculate_loss(data, labels, current_theta.reshape((num_features, 1)))),
            # 迭代次数
            options={'maxiter': max_iterations}
        )
        if not result.success:
            raise ArithmeticError('Can not minimize loss function' + result.message)
        optimized_theta = result.x.reshape(num_features, 1)
        return optimized_theta, loss_history

    def calculate_loss(self, data, labels, theta):
        num_examples = data.shape[0]
        predictions = self.calculate_predicitions(data, theta)
        y_is_current_label_cost = np.dot(labels[labels == 1].T, np.log(predictions[labels == 1]))
        y_is_not_current_label_cost = np.dot(1 - labels[labels == 0].T, np.log(1 - predictions[labels == 0]))
        loss = (-1 / num_examples) * (y_is_current_label_cost + y_is_not_current_label_cost)
        return loss

    def calculate_predicitions(self, data, theta):

        predictions = sigmoid(np.dot(data, theta))

        return predictions

    def gradient_step(self, data, labels, theta):
        """
        梯度下降参数更新计算方法(矩阵运算)
        """
        num_examples = labels.shape[0]
        predictions = self.calculate_predicitions(data, theta)
        label_diff = predictions - labels
        gradients = (1 / num_examples) * np.dot(data.T, label_diff)

        return gradients.T.flatten()

    def predict(self, data):
        num_examples = data.shape[0]
        data_processed = prepare_for_training(data, self.polynomial_degree, self.sinusoid_degree, self.normalize_data)[
            0]
        prob = self.calculate_predicitions(data_processed, self.theta.T)
        max_prob_index = np.argmax(prob, axis=1)
        class_prediction = np.empty(max_prob_index.shape, dtype=object)
        for index, label in enumerate(self.unique_labels):
            class_prediction[max_prob_index == index] = label
        return class_prediction.reshape((num_examples, 1))
