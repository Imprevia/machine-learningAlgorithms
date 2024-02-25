# EM算法

书中的抛硬币游戏，当你知道抛硬币的过程和结果，就很容易统计出三个硬币正面出现的概率。但是当你只知道抛硬币结果，不知道过程（即你不知道抛硬币的顺序），要怎样才能统计出三个硬币正面出现的概率呢？

上面我们不知道抛硬币的过程为**隐变量或潜在变量**

我们知道的抛硬币结果为**观测变量**

如果数据集中含有隐变量的话，我们就无法简单地直接使用极大似然估计法或贝叶斯估计法来估计模型的参数，这时候，我们就需要使用EM算法了。

EM算法即expectation maximization algorithm，其中expectation是期望，maximization是极大化，EM算法的大致步骤也是如此，即首先求期望，接着求极大，所以EM算法也被称为期望极大算法。

# EM算法

EM算法的步骤：

**E步**：计算隐变量的概率

**M步**:计算模型参数的新估计值。

EM算法与初值的选择有关，选择不同的初值可能得到不同的参数估计值。



# EM算法的导出

我们面对一个含有隐变量的概率模型，目标是极大化观测数据（不完全数据）Y关于参数θ的对数似然函数，即极大化：
$$
L(\theta)=\log P(Y|\theta)=\log \sum_{Z}P(Y,Z|\theta)=\log\left( \sum_ZP(Y|Z,\theta)P(Z|\theta) \right)
$$
EM算法是通过迭代逐步近似极大化$L(\theta)$的，因此我们可以假设在第i次迭代后$\theta$ 的估计值是$\theta^{(i)}$，希望在$\theta^{(i)}$的情况下，新的估计值$θ$能让$L(\theta)$更大，这样，在每次迭代后对数似然函数都能增加，于是在有限次迭代后即可得到最大化的极大似然函数。

根据这一规律，我们可以认为$L(\theta)>L(\theta^{(i)})$，即$L(\theta)- L(\theta^{(i)})>0$，现在考虑不等式左边：
$$
L(\theta)- L(\theta^{(i)})=\log\left( \log\sum_ZP(Y|Z,\theta)|P(Z|\theta)- \log P(Y|\theta^{(i)}) \right)
$$
利用Jensen不等式，我们可以得到两者相减的下界，思路是只要下界增大，那么两者的差也会增大。
$$
L(\theta)- L(\theta^{(i)})=\log \left( \sum_ZP(Z|Y,\theta^{(i)})\frac{P(Y|Z, \theta)P(Z|\theta)}{P(Z|Y,\theta^{(i)})} \right)-\log P(Y|\theta^{(i)}) \\
\geq \sum_ZP(Z|Y, \theta^{(i)}) \log\frac{P(Y|Z, \theta)P(Z|\theta)}{P(Z|Y,\theta^{(i)})}-\log P(Y|\theta^{(i)}) \\ 
= \sum_ZP(Z|Y, \theta^{(i)}) \log\frac{P(Y|Z, \theta)P(Z|\theta)}{P(Z|Y,\theta^{(i)})P(Y|\theta^{(i)})}
$$
现在我们将L(θ(i))移到右边，并使整个右边为B(θ，θ(i))，得到：
$$
B(\theta, \theta^{(i)}) = L(\theta^{(i)}) +\sum_ZP(Z|Y, \theta^{(i)}) \log\frac{P(Y|Z, \theta)P(Z|\theta)}{P(Z|Y,\theta^{(i)})P(Y|\theta^{(i)})}
$$
于是：
$$
L(\theta) \geq B(\theta, \theta^{(i)})
$$
现在$B$代表$L(\theta^{(i)})$的下界，只要下界增大，$L(\theta)$也会增大，那么我们使$B$，得到的$\theta$这一轮迭代的最优$\theta$。
$$
\theta^{(i+1)} = \arg \max_\theta B(\theta, \theta^{(i)})
$$
于是问题转换为极大化$B$，可以得到$\theta^{(i)}$的表达式：
$$
\begin{aligned}
\theta^{(i+1)} & = \arg \max_\theta \left( L(\theta^{(i)}) +\sum_ZP(Z|Y, \theta^{(i)}) \log\frac{P(Y|Z, \theta)P(Z|\theta)}{P(Z|Y,\theta^{(i)})P(Y|\theta^{(i)})} \right) \\
& = \arg \max_\theta \left( \sum_ZP(Z|Y,\theta^{(i)}) \log (P(Y|Z,\theta)P(Z|\theta)) \right) \\ 
& = \arg \max_\theta \left( \sum_ZP(Z|Y,\theta^{(i)}) \log (P(Y, Z|\theta))  \right) \\

& = \arg \max_\theta Q(\theta, \theta^{(i)})
\end{aligned}
$$
其中第二个等式成立，因为$L(\theta^{(i)})$，$P(Z| Y,\theta^{(i)})P(Z|\theta^{(i)})$，为常数项，对最大化没有帮助，可以省去，并令 $\sum_ZP(Z|Y,\theta^{(i)}) \log (P(Y, Z|\theta)) = Q(\theta, \theta^{(i)}) $，于是，最大化下界的问题转变为最大化$Q$的问题。

在EM算法中，E即是确定$Q$的过程，而M就是最大化$Q$的过程。



## EM算法的步骤

首先，我们需要确定参数的初值θ(0)，不同的参数初值可能导致不同的模型，EM算法对初值的选择是敏感的，初值的选择可以参照前人的经验进行选择。



第二，进行EM中的E步，即确定Q：
$$
Q(\theta, \theta^{(i)}) = \sum_ZP(Z|Y,\theta^{(i)}) \log (P(Y, Z|\theta)) 
$$


第三，进行EM中的M步，即最大化Q，得到迭代的参数估计值：
$$
\theta^{(i+1)} = \arg \max_\theta Q(\theta, \theta^{(i)})
$$


使用新的参数估计值确定新的Q，并迭代第二和第三步，直到收敛。我们可以给出停止迭代的条件，一般是给出较小的正数$\varepsilon_1 $，$\varepsilon_2$使 ：
$$
||\theta^{(i+1)} - \theta^{(i)} || < \varepsilon_1
$$
或
$$
||Q(\theta^{(i+1)} ,\theta^{(i)}) - Q(\theta^{(i)} ,\theta^{(i)})|| < \varepsilon_2
$$


