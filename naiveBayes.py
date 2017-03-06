#!/usr/bin/python3.6
# _*_ coding: utf-8 _*_

"""
@Author: mfzhu
"""
import numpy as np
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def binary(data, threshold):
    """

    :param data: the data need to be binary
    :param threshold: threshold value for binaryzation
    :return:the data after binaryzation
    """
    data[data > threshold] = 255
    data[data <= threshold] = 0

    return data


def get_the_prior_probability(label):
    """

    :param label: label data
    :return: the dict contains the prior probability
    """
    prior_pro = Counter(label)
    sample_num = len(label)

    for key in prior_pro.keys():
        prior_pro[key] = (prior_pro[key] + 1) / (sample_num + 10)

    return prior_pro


def get_the_conditional_probability(data, label):
    """

    :param data: the binaryzation data
    :param label: label data
    :return: the conditional probability
    """
    per_label_data = {}
    condition_pro = {}

    for i in range(10):
        per_label_data[i] = data[np.where(label == i)]

    for key in per_label_data.keys():
        pro_array = []
        for j in range(784):
            pro_array.append(
                (np.count_nonzero(per_label_data[key][:, j]) + 1) / (per_label_data[key].shape[0] + 2))
        condition_pro[key] = pro_array

    return condition_pro


def sample_map(input_data, condition_pro, prior_pro):
    """

    :param input_data: singal sample data
    :param condition_pro: conditional probability
    :param prior_pro: prior probability
    :return: the tag of sample data according map
    """
    result = {}
    for key in prior_pro.keys():
        pro = prior_pro[key]
        for k in range(len(input_data)):
            if input_data[k] != 0:
                pro *= condition_pro[key][k]
            else:
                pro *= (1 - condition_pro[key][k])
        result[key] = pro
    return max(zip(result.values(), result.keys()))[1]


def data_set_map(data, condition_pro, prior_pro):
    """

    :param data: data set
    :param condition_pro:
    :param prior_pro:
    :return: a list contains the tags of input data set
    """
    result = []
    for j in range(data.shape[0]):
        result.append(sample_map(data[j, :], condition_pro, prior_pro))
    return result


if __name__ == '__main__':

    raw_path = r'F:\work and learn\ML\dataset\MNIST\train.csv'
    # raw data file path
    test_path = r'F:\work and learn\ML\dataset\MNIST\test.csv'
    # test data file path

    raw_data = np.loadtxt(raw_path, delimiter=',', skiprows=1)

    label = raw_data[:, 0]
    data = raw_data[:, 1:]
    # extract the label data and image data

    data = binary(data, 50)
    # binary the image data
    data_train, data_test, label_train, label_test = train_test_split(data, label, test_size=0.33,
                                                                      random_state=23333)
    # split the train data for training and testing

    prior_pro = get_the_prior_probability(label_train)
    condition_pro = get_the_conditional_probability(data_train, label_train)

    # using the train data and train label to calculate the prior probability
    # and conditional probability

    predict = data_set_map(data_test, condition_pro, prior_pro)
    train_result = accuracy_score(label_test, predict)
    print(train_result)

    # calculate the accuracy in test data(from train data)

    test_data = np.loadtxt(test_path, delimiter=',', skiprows=1)
    test_data = binary(test_data, 50)
    test_result = data_set_map(test_data, condition_pro, prior_pro)

    # get the prediction in test data

    index = [i for i in range(1, 28001)]
    res = pd.DataFrame(index, columns=['ImageId'])
    res['Label'] = test_result
    res.to_csv(r"C:\Users\PC\Desktop\result.csv", index=False)
    # generate the result in csv file











