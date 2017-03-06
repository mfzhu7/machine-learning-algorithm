##机器学习之朴素贝叶斯
@(算法小结)
####数学原理
> 我们假设输入变量空间是 $\cal X \in \cal R^n $的n维向量集合，输出空间为类标记为$\cal Y =\{c_1, c_2, .....c_k\}$。输入的向量是$x \in \cal X$，输出的变量是$y \in \cal Y$。则输入的数据集为：
$$  T =\{(x_1, y_2), (x_2, y_2)......(x_N, y_N)\} $$
>所以我们需要先学习到先验概率分布：
$$ \ P( Y = c_k) , k = 1,2,3,.....K\ $$
>和条件概率分布：
$$ \ P(X = x| Y = c_k) = P(X^{(1)} = x^{(1)}, X^{(2)} = x^{(2)}......X^{(n)} = x^{(n)} | Y = c_k), k = 1,2,3,.....K$$
> 在条件独立的情况下我们可以将条件概率分布重写为：
$$\ P(X = x| Y = c_k) = P(X^{(1)} = x^{(1)}| Y = c_k)P( X^{(2)} = x^{(2)}|  Y = c_k)......P(X^{(n)} = x^{(n)} | Y = c_k)$$
$$ =\prod\limits_{i=1}^n P(X^{(n)} = x^{(n)}| Y = c_k)$$

>根据贝叶斯定理我们有：
$$ \ P(X = x| Y = c_k) =  \frac{ P(X = x| Y = c_k) P(Y = c_k)}{\sum_{k} P(X = x | Y = c_k) P(Y = c_k)}$$

>将通过条件独立推得的公式带入即可得：
$$P(X = x | Y = c_k) = \frac{ P(Y = c_k)\prod\limits_{j}P(X^{(j)} = x^{(j)}| Y = c_k)}{\sum_{k}P(Y = c_k)\prod\limits_{j}P(X^{(j)} = x^{(j)}| Y = c_k)}$$

>于是朴素贝叶斯分类器可以表示为：
$$y = f(x) =arg  \mathop{max}\limits_{c_k}\frac{ P(Y = c_k)\prod\limits_{j}P(X^{(j)} = x^{(j)}| Y = c_k)}{\sum_{k}P(Y = c_k)\prod\limits_{j}P(X^{(j)} = x^{(j)}| Y = c_k)}$$



####代码的实现主要思路：
> 首先第一步是数据的预处理部分。MNIST数据集中不存在缺失值等情况，所以不需要清洗数据。主要的处理部分在于对于图像数据的二值化。本次试验使用的阈值为50。二值化处理在代码中实现为**binary**函数。

>第二步计算的是朴素贝叶斯中主要的中间数据**---先验概率**。对于读取的train.csv文件中的第一列就是所需要的标签数据。利用collection模块中的Counter函数可以得到对应不同标签的数量。因为考虑到不同的位置可能出现为0个概率，所以在此处使用了**拉格朗日平滑--在分母上加上标签的类别数**。在名为**get_the_prior_probability**函数中实现。

> 第三步计算的是朴素贝叶斯中的条件概率。首先我们利用的是numpy模块中的where函数找到不同的标签所对应的行数。通过行数来提取不同的标签对应的数据。即将所有标签为1的数据提取出，保存到字典中，key为标签，数据为对应为1的所有数据。同理将所有标签的数据提取出保存即可。接下来，通过遍历key，计算在每个位置上不为0的个数，经过拉格朗日平滑后即可以得到这个位置的条件概率，通过1 - 该概率即可以得到为0的概率。综上即可以得到每个标签下，每个位置对于的**条件概率**。将其保存为字典格式，在函数名为**get_the_conditional_probability**代码中实现。

>第四步实现的是计算单个数据的最大后验概率，返回其中概率最大的类别。通过遍历输入数据的每一个位置，判断其是否为0，来选择条件概率中的对应概率。需要注意的是输入的数据也需要经过**二值化**。这个模块通过函数名为**sample_map**代码实现。
>最后一步是计算多个样本的数据的最大后验概率。**data_set_map**通过调用**sample_map**函数来遍历数据集中的函数。
>几点说明。1、在训练模型的时候，利用了sklearn中的函数，将train数据进行随机划分，留有1/3作为测试数据。2、经过数次调整二值化的阈值，发现对于50~200内的阈值调整，得到的**正确率没有太大的改变**。提高正确率需要进一步对图像本身进行预处理。3、最后在kaggle上提交后的正确率在83.6%左右，计算时间短，但是正确率低。

代码如下：
```python
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

```

