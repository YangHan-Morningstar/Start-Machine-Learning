import operator
import math


class DecisionTree(object):
    def __init__(self):
        super(DecisionTree, self).__init__()
        self.feature_labels = []

    def create_tree(self, data, labels):
        class_list = [example[-1] for example in data]
        if class_list.count(class_list[0]) == len(class_list):
            return class_list[0]
        if len(data[0]) == 1:
            return self.majority_calculate(class_list)

        # 选定根节点
        best_feature_index = self.choose_best_feature_to_split(data)
        best_feature_label = labels[best_feature_index]
        self.feature_labels.append(best_feature_label)
        my_tree = {best_feature_label: {}}

        del labels[best_feature_index]

        feature_value = [example[best_feature_index] for example in data]
        unique_feature_value = set(feature_value)
        for value in unique_feature_value:
            my_tree[best_feature_label][value] = self.create_tree(
                self.split_dataset(data, best_feature_index, value), labels)
        return my_tree

    def majority_calculate(self, class_list):
        """
        选取出现次数最多的类（众数）
        回归问题则选择均值
        """
        class_counter = {}
        for vote in class_list:
            if vote not in class_counter.keys():
                class_counter[vote] = 0
            class_counter[vote] += 1
        sorted_class_counter = sorted(class_counter.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_class_counter[0][0]

    def choose_best_feature_to_split(self, data):
        num_feature = len(data[0]) - 1
        base_entropy = self.calculate_entropy(data)

        best_info_gain = 0
        best_feature_index = -1

        for i in range(num_feature):
            current_feature_list = [example[i] for example in data]
            unique_feature_value = set(current_feature_list)
            new_entropy = 0
            for feature_value in unique_feature_value:
                sub_data = self.split_dataset(data, i, feature_value)
                prob = float(len(sub_data)) / len(data)
                new_entropy += prob * self.calculate_entropy(sub_data)

            info_gain = base_entropy - new_entropy
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature_index = i

        return best_feature_index

    def split_dataset(self, data, axis, value):
        """
        缩减数据集
        :param axis: 删除数据集中哪一列的特征
        :param value:
        :return:
        """
        result_dataset = []
        for example in data:
            if example[axis] == value:
                reduce_example = example[: axis]
                reduce_example.extend(example[axis + 1:])
                result_dataset.append(reduce_example)
        return result_dataset

    def calculate_entropy(self, data):
        """
        计算熵值
        :param data:
        :return: 熵值
        """
        num_example = len(data)
        label_counter = {}
        for example in data:
            current_label = example[-1]
            if current_label not in label_counter.keys():
                label_counter[current_label] = 0
            label_counter[current_label] += 1

        entropy_value = 0
        for label in label_counter:
            prob = float(label_counter[label]) / num_example
            entropy_value -= prob * math.log(prob)

        return entropy_value


if __name__ == "__main__":
    data = [[0, 0, 0, 0, 'no'],
            [0, 0, 0, 1, 'no'],
            [0, 1, 0, 1, 'yes'],
            [0, 1, 1, 0, 'yes'],
            [0, 0, 0, 0, 'no'],
            [1, 0, 0, 0, 'no'],
            [1, 0, 0, 1, 'no'],
            [1, 1, 1, 1, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [2, 0, 1, 2, 'yes'],
            [2, 0, 1, 1, 'yes'],
            [2, 1, 0, 1, 'yes'],
            [2, 1, 0, 2, 'yes'],
            [2, 0, 0, 0, 'no']]
    labels = ['F1-AGE', 'F2-WORK', 'F3-HOME', 'F4-LOAN']

    decision_tree = DecisionTree()

    my_tree = decision_tree.create_tree(data, labels)

    print(my_tree)
