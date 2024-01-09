# 感知机模型(Perceptron)|机器学习方法（李航）

# 什么是感知机

试想一下，在一个盒子里面一个哈密瓜和一个西瓜。你只能用手摸判断哪个是哈密瓜，哪个是西瓜。我们用手摸两个瓜的大小，纹理等特征，然后我们的大脑会根据收集到数据和我们大脑中认知的哈密瓜和西瓜的特征进行对比，识别到哪个是哈密瓜，哪个是西瓜。

在上面例子中用手摸两个瓜的大小、纹理两个特征，是一个数据收集的过程，然后我们的到一个二维的数据集。我们大脑中认知的哈密瓜和西瓜的特征，就是一个训练好的模型。我们将收集到的数据集传入模型，就能知道哪个是哈密瓜，哪个是西瓜。

在了解感知机后，我们需要了解在数学上怎样实现感知机的。

首先我们要清楚感知机是一个二分类且线性可分的模型。在二维的平面中，可以画一条线将数据集分为两类；在三维的空间中，可以画一个平面将数据集分为两类；在n维的空间中，可以用n-1维的超平面将数据集分为两类。

以二维平面为例，怎么画出一条线可以将数据集分为两类，这就是感知机。下面就说怎么画出这样的一条线。



# 感知机模型

假设输入空间 $ \chi⊆ R^n $ ，输出空间为 $\gamma=\left \{ +1,-1\right \}$。其中每一个输入$x\subseteq \chi$表示对应于实例的特征向量，也就是对应于输入空间（特征空间）的一个点，$y\subseteq \gamma$输出表示实例的类别。从输入空间至输出空间的函数$f(x)$称为感知机。此处  $R^n$  ，代表着n维空间，如果是  $XOY$ 坐标轴平面，即n=2，以此类推。初学者对于后续概念可以根据二维平面进行理解。
$$
f(x)=sign(w\cdot x+b)
$$
其中$w$叫做权值向量，$b$叫作偏置。$w·x$表示w和x的内积，即$w1*x1+w2*x2…+wn*xn$。$Sign$是符号函数。
$$
sign(x)=\begin{cases}
1,&x \geq 0  \\
-1, & x < 0
\end{cases}
$$


下图可以更直观认知感知机

![2.1](.\images\2.1.jpg)

图的的点被一条直线分为两类，我们假设圆点是正类为1，叉叉是负类为-1，直线就是感知机。怎样画出一条线刚好能将全部点准确的分为两类呢？

书中提出了损失函数的概念。

损失函数代表的是模型对一个数据集的预测与该数据集的真实值之间的差距，显然，差距越大，说明模型预测越不准确，我们需要一个预测准确的模型，所以我们需要让这样的差距最小，即最小化损失函数。

对于感知机而言，损失函数为误分类点到超平面的总距离。可知损失函数越小，说明被误分类的点到超平面的距离较近，且被误分类点较少，如果损失函数为0，说明没有误分类点。

## 损失函数

感知机的损失函数为误分类点到超平面的总距离，首先看一个点到超平面的距离：
$$
\frac{1}{||w||}\begin{vmatrix}w*x_0+b\end{vmatrix}
$$
点到超平面的距离公式的推导：以下面二维的图为例

![2.1.1](.\images\2.1.1.jpg)

已知：直线$w_1*x_1+w_2*x_2+b=0$

由图得点A的$x_1$为0，代入直线$w_2*x_2+b=0$，得$x_2=-\frac{b}{w_2}$，所以点A为（0，$-\frac{b}{w_2}$）；

由图得点B的$x_2$为0，代入直线$w_1*x_1+b=0$，得$x_1=-\frac{b}{w_1}$，所以点B为（0，$-\frac{b}{w_1}$）；

所以OA为$-\frac{b}{w_2}$，OB为$-\frac{b}{w_1}$

由勾股定理得AB为$-b\frac{\sqrt{w_1^2+W_2^2}}{w_1w_2}$

根据三角形的面积得$\frac{1}{2}OA*OB=\frac{1}{2}AB*OC$
$$
-\frac{b}{w_1}*-\frac{b}{w_2}=-b\frac{\sqrt{w_1^2+W_2^2}}{w_1w_2}*OC
$$
所以$OC=-\frac{b}{\sqrt{w_1^2+W_2^2}}=-\frac{b}{||w||}$



上面知道了点到超平面的距离公式，但是我们需要的是误分类到超平面的距离，书上给出了公式：
$$
-y(w*x_i+b)>0
\tag{1}
$$
因为当$w*x+b\ge0$且被误分时，对应真实的$y=-1$，当$w*x<0$且被误分时，对应真实的$y=1$。这样就可以判断哪个点是误分类点。现在通过公式(1)，我们将所有误分类点提取出来，假设有M个误分类，下面计算M个误分类到超平面的距离：
$$
-y_1\frac{1}{||w||}\begin{vmatrix}w*x_1+b\end{vmatrix}-y_1\frac{1}{||w||}\begin{vmatrix}w*x_2+b\end{vmatrix}-...-y_1\frac{1}{||w||}\begin{vmatrix}w*x_M+b\end{vmatrix}=-\frac{1}{||w||}\sum_{x_i\in M}y(w*x_i+b)
$$
因为需要取得损失函数最小值，可以忽略常数||w||。可得其损失函数为：
$$
L(w,b)=-\sum_{x_i\in M}y(w*x_i+b)
$$
损失函数越大，说明对应的感知机模型越差，因此，我们需要一个损失函数最小的模型，接下来说明如何取得损失函数最小值。



## 随机梯度下降法

以图2.1为例，当w和b都为0的是后误分类点是最多的，随着w和b不断变大，误分类点先变少，再变多。就想回归函数一样，如下图。

![2.2](.\images\2.2.jpg)

随机梯度下降法，就是在上图上从最上的一个点开始，每次随机间隔一点，逐步的往下，直到找到最低的一个点。

在感知机中使用随机梯度下降法的步骤：

第一步，选取w和b的初始值分别为$w_0$，$b_0$

第二步，在训练集中选取数据$(x_i,y_i)$

第三步，如果$y_i(w*x_i)\le0$，则更新w和b
$$
w\leftarrow w+\eta y_ix_i \\
b\leftarrow b+\eta y_i
$$
第四步，转到步骤二，直至没有误分类点。



上代码

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# 加载鸢尾花数据集
iris = datasets.load_iris()
df = pd.DataFrame(iris.data)
df['label'] = iris.target
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'label']

# 提取前100条数据
data = np.array(df.iloc[0:100, [0, 1, -1]])
# 得到x(特征向量)、y(分类标签)
x, y = data[:, :-1], data[:, -1]

# 将两类分类标签分别替换为1与-1，便于感知机处理
y = np.array([1 if i == 1 else -1 for i in y])

class Model:
    def __init__(self):
        # 初始化权重，特征向量长度为2，故在初始化中故将其分别赋予1的权重
        self.w = np.zeros(len(data[0]) - 1)
        # 初始化偏置为0
        self.b = 0
        # 初始化学习率为0.1
        self.rate = 0.1

    # 定义sign函数,用于判断当前点是否分类正确
    def sign(self, x, w, b):
        y = np.dot(w, x) + b # y = w[0]*x[0] + w[1]*x[1] + b
        return y

    def fit(self, X_train, Y_train):
        while True:
            wrong_count = 0  # 错误分类点计数器
            for i in range(len(X_train)):
                x = X_train[i]
                y = Y_train[i]
                print(w, x, np.dot(w, x))
                if y * self.sign(x, self.w, self.b) <= 0:
                    self.w = self.w + self.rate * np.dot(x,y)
                    self.b = self.b + self.rate * y
                    wrong_count += 1
            if wrong_count == 0:  # 当损失函数为0时，分类结束
                break
        return self.w, self.b
    
# 实例化模型
perceptron = Model()
# 训练模型
w, b = perceptron.fit(x, y)

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
```

![2.3](.\images\2.3.png)
