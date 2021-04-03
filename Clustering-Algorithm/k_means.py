import numpy as np


class KMeans:
    def __init__(self, data, k_value):
        self.data = data
        self.k_value = k_value

    def train(self, max_iterations):
        # 随机选择K个中心点
        center_points = self.center_point_init(self.data, self.k_value)
        # 开始训练
        examples_num = self.data.shape[0]
        example_closest_center_point = np.empty((examples_num, 1))
        for _ in range(max_iterations):
            # 得到当前每一个样本点到K个中心点的距离，找到最近的
            example_closest_center_point = self.center_point_find_closest(self.data, center_points)
            # 进行中心点位置更新
            center_points = self.center_point_compute(self.data, example_closest_center_point, self.k_value)
        return center_points, example_closest_center_point

    def center_point_init(self, data, k_value):
        examples_num = data.shape[0]
        random_ids = np.random.permutation(examples_num)
        center_point = data[random_ids[: k_value], :]
        return center_point

    def center_point_find_closest(self, data, center_points):
        example_num = data.shape[0]
        center_point_num = center_points.shape[0]
        example_closest_center_point = np.zeros((example_num, 1))
        for example_index in range(example_num):
            distance = np.zeros((center_point_num, 1))
            for centroid_index in range(center_point_num):
                distance_diff = data[example_index, :] - center_points[centroid_index, :]
                distance[centroid_index] = np.sum(distance_diff ** 2)
            example_closest_center_point[example_index] = np.argmin(distance)
        return example_closest_center_point

    def center_point_compute(self, data, example_closest_center_point, k_value):
        num_features = data.shape[1]
        center_points = np.zeros((k_value, num_features))
        for center_point_id in range(k_value):
            closest_ids = example_closest_center_point == center_point_id
            center_points[center_point_id] = np.mean(data[closest_ids.flatten(), :], axis=0)
        return center_points


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    data = pd.read_csv('../data/iris.csv')
    iris_types = ['SETOSA', 'VERSICOLOR', 'VIRGINICA']

    x_axis = 'petal_length'
    y_axis = 'petal_width'

    # 绘制散点图
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for iris_type in iris_types:
        plt.scatter(data[x_axis][data['class'] == iris_type], data[y_axis][data['class'] == iris_type], label=iris_type)
    plt.title('label known')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(data[x_axis][:], data[y_axis][:])
    plt.title('label unknown')
    plt.show()

    num_examples = data.shape[0]
    x_train = data[[x_axis, y_axis]].values.reshape(num_examples, 2)

    # 指定好训练所需的参数
    k_value = 3
    max_iteritions = 50

    k_means = KMeans(x_train, k_value)
    center_points, example_closest_center_point = k_means.train(max_iteritions)

    # 对比结果
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for iris_type in iris_types:
        plt.scatter(data[x_axis][data['class'] == iris_type], data[y_axis][data['class'] == iris_type], label=iris_type)
    plt.title('label known')
    plt.legend()

    plt.subplot(1, 2, 2)
    for centroid_id, centroid in enumerate(center_points):
        current_examples_index = (example_closest_center_point == centroid_id).flatten()
        plt.scatter(data[x_axis][current_examples_index], data[y_axis][current_examples_index], label=centroid_id)

    for centroid_id, centroid in enumerate(center_points):
        plt.scatter(centroid[0], centroid[1], c='black', marker='x')
    plt.legend()
    plt.title('label kmeans')
    plt.show()
