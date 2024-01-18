# k近邻|机器学习方法（李航）

# 什么是k近邻

你可以理解为人以群分，物以类聚。我们在生活中会逐渐与一群兴趣爱好和自己相似的人组成一个圈子，我们的朋友也会组成他们自己的圈子，依此类推。那么以我们为中心，以朋友与朋友的关系将所有人联系起来。相隔K层后，朋友的兴趣爱好还会和我们相似吗？而k近邻就是要找到相隔K层后，朋友的兴趣爱好依然和我们相似的，K层内所有的朋友。

书上的定义是：给定一个训练数据集，对新的输入实例，在训练数据集中找到与该实例最邻近的k个实例，这k个实例的多数属于某个类，就把该输入实例分为这个类。

从上面的定义可以得到k近邻的三个基本要素：

1. 在训练数据集中找到与该实例最邻近的k个实例
   1. 计算数据集到实例的距离 — 距离的度量
   2. 邻近的k个实例 — K值的选择
2. 这k个实例的多数属于某个类 — 分类决策规则决定



# k近邻模型

k近邻的算法中，当训练集、距离度量（如欧氏距离）、k值（近邻点的数量）及分类决策规则（如多数表决）确定后，对于任何一个新的输入实例，它所属的类可以被确定并且为唯一的类。

所以我们要做的是确定距离度量、k值、分类决策规则。



## 距离度量

k近邻模型的特征空间一般是n维实数向量空间$R^n$。使用的距离是欧氏距离，但也可以是其他距离，如更一般的$L_p$距离或Minkowski距离。

在n维实数向量空间$R^n$中，$x_i,x_j $在$R^n$上，计算$x_i,x_j $的$L_p$距离：
$$
L_p(x_i,x_j) = (\sum_{l=1}^{n}|x_i^{(l)}-x_j^{(l)}|^p)^{\frac{1}{p}}
$$


当p=2时，称为欧式距离：
$$
L_2(x_i,x_j) = (\sum_{l=1}^{n}|x_i^{(l)}-x_j^{(l)}|^2)^{\frac{1}{2}}
$$
当p=1时，称为曼哈顿距离：
$$
L_1(x_i,x_j) = \sum_{l=1}^{n}|x_i^{(l)}-x_j^{(l)}|
$$
当p=$\infty$时，它是各个坐标距离的最大值:
$$
L_\infty(x_i,x_j) = \max_{l}|x_i^{(l)}-x_j^{(l)}|
$$
下图代表在二维空间中p取不同值时，与原点的L距离为1的点围成的图形。

![3.1](F:\workCode\机器学习方法\K近邻法\images\3.1.jpg)

不同的距离度量所确定的最近邻点是不同的。**在k近邻算法中，通常采用欧氏距离。**



## k值的选择

k值的选择会对算法的结果产生重大的影响。

如果选择较小的k值，就相当于用较小的邻域中的训练实例进行预测，“学习”的近似误差会减小。但缺点是“学习”的估计误差会增大。预测结果会对近邻的实例点非常敏感，而远处的实例点则对此没有影响，如果邻近的实例点恰巧是噪声，预测就会出错。

k值的减小就意味着整体模型变得复杂，容易发生过拟合。

如果选择较大的k值，就相当于用较大邻域中的训练实例进行预测。其优点是可以减少学习的估计误差，但缺点是学习的近似误差会增大。

k值的增大就意味着整体的模型变得简单。



在应用中，k值一般取一个比较小的数值。**通常采用交叉验证法来选取最优的k值。**

*交叉验证法：书中第一章有说明*



**最优的k值的选择：通过对多个k值训练多个模型，然后通过交叉验证法找到经验风险最小的k值。**



## 分类决策规则（k近邻算法的损失函数）

**k近邻法中的分类决策规则往往是多数表决**，即由输入实例的k个邻近的训练实例中的多数类决定输入实例的类。

多数表决：如果分类的损失函数为0-1损失函数（即分类错误则为1，分类正确则为0，要极小化该损失函数，让分类错误的数量下降），分类函数为：
$$
f:R^n\rightarrow\{c_1,c_2,...,c_k\}
$$
误分类的概率:
$$
P(Y\ne f(X)) = 1-P(Y = f(X)) \\
\frac{1}{k}\sum_{x_i \in N_k(x)}I(y_i \ne c_j)=1-\frac{1}{k}\sum_{x_i \in N_k(x)}I(y_i = c_j)
$$
要使误分类率最小即经验风险最小，就要使 $\sum_{x_i \in N_k(x)}I(y_i = c_j)$ 最大，所以多数表决规则等价于经验风险最小化。



上面简单的k近邻法叫做线性扫描，当数据量大时，非常耗时。为了提高搜索效率，可以使用**kd树**



# kd树

kd树是不断将数据集切分为两个子空间，直到子空间不存在实例点，从而构成树形数据结构。通过检索树形数据结构，可以节省大量的数据搜索时间。



**构造kd树**方法如下：

1. 构造根结点：在n维的空间中，选择一个维度，在这个维度上实例的中维数作为切分点，将n维的空间划分为两个子空间。将落在超平面上的实例点作为根节点
2. 生成子节点：划分出来的两个子区域，小于切分点的为左区域，大于切分点的为右区域。分别选在另一个维度上按照1的方法确定两个区域的切分点，对两个子区域再次划分为两个部分。并将落在超平面上的实例点作为子节点，左区域的为左节点，有区域的为右节点。
3. 重复步骤2，直到特征空间中的子空间中不存在实例点。

**注意：如果样本点数量为偶数，数学上中位数应当是中间两个数的平均值，但在kd树中一个结点不能为空，因此可以选择中间两个数中的一个作为切分点。**



**搜索kd树**

![3.2](.\images\3.2.png)



# 线性扫描代码

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 加载鸢尾花数据集
iris = datasets.load_iris()
df = pd.DataFrame(iris.data)
df['label'] = iris.target
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'label']
df.head()

# 加载鸢尾花数据集
x = iris.data
y = iris.target

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)

class Model:
    def __init__(self, K=5):
        self.K = K

    # 计算新输入点与实例的距离
    def distance(self, data, x):
        # 将实例的特征砖矩阵
        x = np.mat(x)
        
        # 欧式距离
        d = np.sqrt(np.sum(np.square(data - x), axis = 1))
        
        num = len(data)
        arr = np.array(d).reshape(num)
        return arr
    
    # 获取距离最近的K个点的index
    def closestK(self, distance):
        # np.argsort 返回从小到大排列的元素的index， 比如np.argsort[1,3,2]得到的是[0,2,1]
        sort = np.argsort(distance)
        
        k_index = sort[:self.K]
        return k_index
    
    # 多数表决
    def classN(self, label, k_index):
        # 类别的个数
        num = len(np.unique(np.array(label)))
        record = np.zeros(num)

        for index in k_index:
            record[np.array(label)[index]] += 1
            
        class_n = list(record).index(max(record))        
        return class_n
        
    
    def fit(self, train_data, train_label):
        self.train_data = train_data
        self.train_label = train_label
        
    def predict(self, test_data):
        self.test_data = test_data
        start = time.time()

        # 得到测试集的记录的个数
        testNum = len(self.test_data)

        types = np.zeros(testNum)

        # 遍历每一个新输入实例
        for i in range(testNum):
            print('classifying %d' % i)
            dArr = self.distance(self.train_data, self.test_data[i])

            # 得到距离新输入实例距离最短的K个点的index
            sampleK_index = self.closestK(dArr)

            # 得到新输入实例的分类
            types[i] = self.classN(self.train_label, sampleK_index)

        end = time.time()
        print('Classifying time: ', end - start)
        return types

# 测试k=5时，的准确率
knn = Model(K=5)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

# 预测结果展示
labels = ["山鸢尾","虹膜锦葵","变色鸢尾"]
for i in range(len(y_pred)):
    label1 = labels[y_pred[i].astype(int)]
    label2 = labels[y_test[i].astype(int)]
    print("第%d次测试:\t预测值:%s\t\t真实值:%s"%((i+1),label1,label2))

# 计算准确率
accuracy = accuracy_score(y_test, y_pred) # accuracy_score()函数会比较真实标签值和预测标签值，并计算出准确分类的样本数占总样本数的比例，即准确率。
print("准确率:", accuracy)

# 测试k=1-50时，的准确率
precision = []
k1 = range(1,50)
for k in k1:
    knn = Model(K=k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    precision.append(accuracy)

plt.plot(k1,precision,label='line1',color='g',marker='.',markerfacecolor='pink',markersize=10)
plt.xlabel('K')
plt.ylabel('Precision')
plt.title('KNN')
plt.legend()
plt.show()
```

![3.3](.\images\3.3.png)