import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets

# 加载鸢尾花数据集
iris = datasets.load_iris()

# 转换为DataFrame格式并设置列标签
df = pd.DataFrame(iris.data, columns=['sepal length', 'sepal width', 'petal length', 'petal width'])
# 为数据设置label，转化为二分类问题
df['label'] = iris.target
# 提取前100条数据
data = np.array(df.iloc[0:100, [0, 1, -1]])

# 得到x(特征向量)、y(分类标签)
x, y = data[:, :-1], data[:, -1]
# 将两类分类标签分别替换为1与-1，便于感知机处理
y = np.array([1 if i == 1 else -1 for i in y])


class Model:
    def __init__(self):
        # 初始化权重，特征向量长度为2，故在初始化中故将其分别赋予1的权重
        self.w = np.ones(len(data[0]) - 1)
        # 初始化偏置为0
        self.b = 0
        # 初始化学习率为0.1
        self.rate = 0.1

    # 定义sign函数,用于判断当前点是否分类正确
    def sign(self, x, w, b):
        y = np.dot(x, w) + b
        return y

    def fit(self, X_train, Y_train):
        Classfication_status = False  # 用于判断当前点是否分类正确，默认不正确
        while not Classfication_status:
            wrong_count = 0  # 错误分类点计数器
            for i in range(len(X_train)):
                x = X_train[i]
                y = Y_train[i]
                if y * self.sign(x, self.w, self.b) <= 0:
                    self.w = self.w + self.rate * np.dot(y, x)
                    self.b = self.b + self.rate * y
                    print(self.b, self.w)
                    wrong_count += 1
                if wrong_count == 0:  # 当损失函数为0时，分类结束
                    Classfication_status = True
        return self.w, self.b


# 实例化模型
perceptron = Model()
# 训练模型
w, b = perceptron.fit(x, y)

# x = np.linspace(4, 7, 10)
# y_ = -(w[0] * x + b) / w[1]
x_point = np.arange(4,7,0.5)
y_point = -(w[0] * x_point + b) / w[1]
plt.plot(x_point,y_point)
# 绘制散点图
plt.scatter(x[:50, 0], x[:50, 1], label='0')
plt.scatter(x[50:, 0], x[50:, 1], label='1')
# plt.plot(x, y_)
plt.xlabel("sepal length")
plt.ylabel("sepal width")
plt.legend()
plt.show()
