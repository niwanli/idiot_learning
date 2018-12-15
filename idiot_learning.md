#  人工智障之傻瓜学习

[TOC]

## 1 线性回归

### 广告数据集

- 数据集名称：`Advertising.csv`

- 数据集内容：该数据集共4列200行，每一行表示一个特定的商品，前3列为输入特征，最后一列为输出特征。

  - 输入特征：	
    - TV：该商品用于电视上的广告费用 
    - Radio：在广播媒体上投资的广告费用
    - Newspaper：用于报纸媒体的广告费用 

  - 输出特征：
    - Sales：该商品的销量

### 初步分析

```python
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # 导入数据
    path = 'Advertising.csv'
    data = pd.read_csv(path)
    x = data[['TV', 'Radio', 'Newspaper']]
    y = data['Sales']
    
    # 绘图
    plt.figure(figsize=(9,12))
    plt.subplot(311)
    plt.plot(data['TV'], y, 'ro')
    plt.title('TV')
    plt.grid()
    plt.subplot(312)
    plt.plot(data['Radio'], y, 'g^')
    plt.title('Radio')
    plt.grid()
    plt.subplot(313)
    plt.plot(data['Newspaper'], y, 'b*')
    plt.title('Newspaper')
    plt.grid()
    plt.tight_layout()
    plt.show()
```

### 绘图结果

![](pictures/1.png)

> 观察发现Newspaper 散点图中Newspaper的线性关系并不明显。

### sklearn线性回归

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # 导入数据data
    path = 'Advertising.csv'
    data = pd.read_csv(path)
    
    # 提取特征列x和标签y
    feature_cols = ['TV', 'Radio']
    x = data[feature_cols]
    y = data['Sales'] # 等价于 y = data.Sales
    
    # 划分训练集（x_train，y_train）和测试集（x_test，y_test）
    # 默认分割为75%的训练集，25%的测试集
    from sklearn.cross_validation import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
    
    # sklearn线性回归LinearRegression()
    from sklearn.linear_model import LinearRegression
    linreg = LinearRegression()
    
    # 训练模型
    model = linreg.fit(x_train, y_train)
	
    # 模型预测
    y_pred = linreg.predict(x_test)
    
    # 查看训练得到的相关系数
    coeffs = dict(zip(feature_cols, linreg.coef_))
    print(coeffs)
    
    # 使用RMSE评估预测结果
    sum_mean = 0
    for i in range(len(y_pred)):
        sum_mean += (y_pred[i] - y_test.values[i]) ** 2
    print("RMSE:", np.sqrt(sum_mean / len(y_pred)))
    
    # 绘图
    plt.figure()  
    plt.plot(range(len(y_pred)),y_pred,'b',label="predict")  
    plt.plot(range(len(y_test)),y_test,'r',label="test")  
    plt.legend(loc="upper right") 
    plt.xlabel("the number of sales")  
    plt.ylabel('value of sales')  
    plt.show() 
```

### 输出结果

- 3个输入特征的相关系数：

  `{'Radio': 0.17915812245088836, 'Newspaper': 0.0034504647111804065, 'TV': 0.046564567874150288}`

- 均方根误差(Root Mean Squared Error)：

  `RMSE: 1.40465142303`

- 绘图结果

  ![](pictures/2.png)

  > - 根据3个输入特征的相关系数可以发现，Newspaper的系数很小 。
  > - 进一步观察，发现Newspaper 散点图中Newspaper的线性关系并不明显 。

**因此，尝试在训练模型的时候将输入特征Newspaper 移除，再看看线性回归预测结果的RMSE如何。**

```python
    # 修改提取的特征列x
    feature_cols = ['TV', 'Radio']
```

### 重新训练后的结果

- 3个输入特征的相关系数：

  `{'Radio': 0.18117959203112891, 'TV': 0.046602340710768547}`

- 均方根误差(Root Mean Squared Error)：

  `RMSE: 1.38790346994`

### 结论

在将Newspaper这个特征移除之后，得到RMSE变小了，说明Newspaper可能不适合作为预测销量的特征。

## 2 逻辑回归

### 鸢尾花数据集

- 数据集名称：`iris.data.txt`

- 数据集内容：该数据集包括3个鸢尾花类别，每个类别有50个样本。其中一个类别是与另外两类线性可分的，而另外两类不能线性可分。 每行代表一个鸢尾花样本，有4个输入特征和1个输出特征。

  - 输入特征：

    - sepal length
    - sepal width
    - petal length
    - petal width

  - 输出特征：

     - Iris Setosa

     - Iris Versicolour

     - Iris Virginica

### sklearn逻辑回归

​        

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# 将3种鸢尾花类别映射为{0, 1, 2}
def iris_type(s):
    it = {b'Iris-setosa': 0,
          b'Iris-versicolor': 1,
          b'Iris-virginica': 2}
    return it[s]
     
if __name__ == "__main__":
    # 导入数据data
    path = 'iris.data.txt'
    data = np.loadtxt(path, dtype=float, delimiter=',', converters={4:iris_type})
    x, y = np.split(data, [4,], axis=1)
     
    # 为了可视化，仅使用前两列特征
    x= x[:, :2]
     
    # 训练模型
    logreg = LogisticRegression()
    logreg.fit(x, y.ravel())
    
    # 画图
    M, N = 500, 500
    x1_min, x1_max = x[:,0].min(), x[:,0].max()
    x2_min, x2_max = x[:,1].min(), x[:,1].max()
    t1 = np.linspace(x1_min, x1_max, M)
    t2 = np.linspace(x2_min, x2_max, N)
    x1, x2 = np.meshgrid(t1, t2)
    x_test = np.stack((x1.flat, x2.flat), axis=1)
    
    # 模型预测
    y_predict = logreg.predict(x_test)
    y_predict = y_predict.reshape(x1.shape)
    
    # 绘图
    plt.pcolormesh(x1,x2,y_predict,cmap=plt.cm.prism)
    plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', cmap=plt.cm.prism)
    
    # 显示样本
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.grid()
    plt.show()
    
    # 预测结果
    y_predict = logreg.predict(x)
    y = y.reshape(-1)
    result = y_predict == y
    c = np.count_nonzero(result)
    print('正确分类的样本数：', c)
    print('准确率： %.2f%%' % (100 * float(c) / float(len(result))))
```

### 输出结果

- 正确分类的样本数： `115`

- 准确率： `76.67%`

- 绘图结果

  ![](pictures/3.png)

> 仅仅使用两个特征：花萼长度和宽度，在150个样本中，有115个分类正确，正确率为76.67% 。

**因此，尝试使用更多的特征（如4个全部使用） 来训练模型。**

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split

def iris_type(s):
    it = {b'Iris-setosa': 0,
          b'Iris-versicolor': 1,
          b'Iris-virginica': 2}
    return it[s]
    
if __name__ == "__main__":
    # 导入数据data
    path = 'iris.data.txt'
    data = np.loadtxt(path, dtype=float, delimiter=',', converters={4:iris_type})
    x, y = np.split(data, [4,], axis=1)
    
    # 使用逻辑回归训练模型
    logreg = LogisticRegression()
    logreg.fit(x, y.ravel())
    
    # 模型预测结果
    y_predict = logreg.predict(x)
    y = y.reshape(-1)
    result = y_predict == y
    c = np.count_nonzero(result)
    print('正确分类的样本数：', c)
    print('准确率： %.2f%%' % (100 * float(c) / float(len(result))))
```



### 重新训练后的结果

- 正确分类的样本数： `144`
- 准确率：` 96.00%`

### 结论

试验后会发现，当使用更多的特征进行训练后，模型在150个样本中，有144个分类正确，正确率为96%，分类效果提高明显。

## 3 决策树

决策树（Decision Tree）是一种简单但是广泛使用的分类器。通过训练数据构建决策树，可以高效地对未知的数据进行分类。

### 训练决策树

```python
import numpy as np
from sklearn import tree

def iris_type(s):
    it = {b'Iris-setosa': 0,
          b'Iris-versicolor': 1,
          b'Iris-virginica': 2}
    return it[s]
    
if __name__ == "__main__":
    # 导入数据data
    path = 'datasets\iris.data.txt'
    data = np.loadtxt(path, dtype=float, delimiter=',', converters={4:iris_type})
    x, y = np.split(data, [4,], axis=1)

    # 划分训练集（x_train，y_train）和测试集（x_test，y_test）
    # 默认分割为75%的训练集，25%的测试集
    from sklearn.cross_validation import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
    print('测试集样本数目：', len(x_test))
    
    # 初始化决策树
    clf = tree.DecisionTreeClassifier()
    
    # 训练决策树
    clf = clf.fit(x_train, y_train)
    
    # 预测样本
    y_predict = clf.predict(x_test)
    y_test = y_test.reshape(-1)
    result = y_predict == y_test
    c = np.count_nonzero(result)
    print('正确分类的样本数：', c)
    print('准确率： %.2f%%' % (100 * float(c) / float(len(result))))
```

### 输出结果

- 测试集样本数目： 38
- 正确分类的样本数： 37
- 准确率： 97.37%

> 由于训练测试样本太少，无法看出所训练模型的实际效果如何。

## 4 随机森林

随机森林`Random Forest`，指的是利用多棵树对样本进行训练并预测的一种分类器。它通过对数据集中的子样本进行训练，从而得到多棵决策树，以提高预测的准确性并控制在单棵决策树中极易出现的过拟合情况 。

## 5 支持向量机SVM

### 成人数据集

- 数据集名称：
  - 训练集：`adult.data.txt`
  - 测试集：`adult.test.txt`

- 数据集内容：该数据集记录的是一些成年人的基本信息，每条数据有`14`个输入特征和`1`个输出特征。该数据集的目的是：根据一个成人的`14`条基本信息，预测该人一年的薪资是否超过`50K`，`1`表示超过，`-1`表示不超过。

  - 输入特征：
    `age workclass fnlwgt(final weight) education education-num marital-status occupation relationship race sex captital-gain captital-loss hours-per-week native-country`

  - 输出特征：

    `>50K, <=50K`

**本数据集首先要做的处理是：将连续特征离散化，将有M个类别的离散特征转换为M个二进制特征。**

> 后面，我们使用[LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)对`SVM`模型进行训练、预测。 因此，需要将数据处理成`LIBSVM `所需的数据格式。
>
> - `adult.data.txt` --> `a9a.txt`
> - `adult.test.txt` -->  `a9a.t`

### LIBSVM

- `LIBSVM`参数选项：

```shell
  -s svm_type : set type of SVM (default 0)
	0 -- C-SVC
	1 -- nu-SVC
	2 -- one-class SVM
	3 -- epsilon-SVR
	4 -- nu-SVR
-t kernel_type : set type of kernel function (default 2)
	0 -- linear: u'*v
	1 -- polynomial: (gamma*u'*v + coef0)^degree
	2 -- radial basis function: exp(-gamma*|u-v|^2)
	3 -- sigmoid: tanh(gamma*u'*v + coef0)
-d degree : set degree in kernel function (default 3)
-g gamma : set gamma in kernel function (default 1/num_features)
-r coef0 : set coef0 in kernel function (default 0)
-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
-n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
-m cachesize : set cache memory size in MB (default 100)
-e epsilon : set tolerance of termination criterion (default 0.001)
-h shrinking: whether to use the shrinking heuristics, 0 or 1 (default 1)
-b probability_estimates: whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
-wi weight: set the parameter C of class i to weight*C, for C-SVC (default 1)
```

- 调用`LIBSVM`：

```python
import os
from svmutil import *

# 修改系统路径
os.chdir('tools\libsvm-3.23\python')

# 读取数据
train_y, train_x = svm_read_problem('../../../datasets/a9a.txt')
test_y, test_x = svm_read_problem('../../../datasets/a9a.t')

# 训练模型
m = svm_train(train_y, train_x, '-c 5')

# 测试模型
p_label, p_acc, p_val = svm_predict(test_y, test_x, m)
```

### 输出结果

- Accuracy = 84.9702% (13834/16281) (classification)

> 由结果可知，利用`LIBSVM`和`a9a`的训练集得到的`SVM`分类器模型在`a9a`测试集上的分类准确率约为84.97%。

## 6 聚类 

通过`MiniBatchKMeans`，`K-Means`，`SpectralClustering`，`DBSCAN`四种算法解决基本的聚类问题，使用`sklearn`提供的聚类模块和鸢尾花数据集，对聚类效果进行横向比较。

### 评价标准

- **Adjusted Rand Index（ARI）：**

用来计算两组标签之间的相似性，本实验中计算了算法聚类后得到的标签`algorithm.labels_`与数据集中真实类别标签`y`之间的相似性。取值范围：`-1~1`，值越大，相似性越高。

- **Homogeneity（同质性）：**

对于聚类结果中的每一个聚类，它只包含真实类别中的一个类的数据对象。取值范围：`0~1`，值越大，同质性越高。

- **Completeness（完整性）：**

对于真实类别中的一个类的全部数据对象，都被聚类到一个聚类中。取值范围：`0~1`，值越大，完整性越高。

### 比较不同算法性能1（iris数据集）

```python
from sklearn.cluster import MiniBatchKMeans, KMeans, SpectralClustering, DBSCAN
from sklearn.datasets import load_iris
from sklearn import metrics
import pandas as pd
import numpy as np
import time

# load dataset
iris = load_iris()
x = iris.data
y = iris.target

# create clustering estimators
three_means  = MiniBatchKMeans(n_clusters=3)
kmeans = KMeans(n_clusters=3)
spectral = SpectralClustering(n_clusters=3, eigen_solver='arpack')
dbscan = DBSCAN(eps=1.0)
cluster_algorithms = [three_means, kmeans, spectral, dbscan]

# evaluate the performances of 4 clustering algorithms
values = []
for algorithm in cluster_algorithms:
    t0 = time.time()
    # training
    algorithm.fit(x)
    t1 = time.time()
    delta_t = t1-t0
    # metrics
    ari = metrics.adjusted_rand_score(algorithm.labels_, y)
    homo = metrics.homogeneity_score(algorithm.labels_, y)
    compl = metrics.completeness_score(algorithm.labels_, y)
    scores = [ari, homo, compl, delta_t]
    values.append(scores)

# output measure values using DataFrame
cluster_names = ['MiniBatchKMeans', 'KMeans', 'SpectralClustering', 'DBSCAN']
metrics_names = ['ARI', 'Homo', 'Compl', 'Time']
df = pd.DataFrame(values, index=cluster_names, columns=metrics_names)
print(df)
```

### 输出结果1

|                    | ARI      | Homo     | Compl    | Time     |
| ------------------ | -------- | -------- | -------- | -------- |
| MiniBatchKMeans    | 0.730238 | 0.764986 | 0.751485 | 0.024001 |
| KMeans             | 0.730238 | 0.764986 | 0.751485 | 0.038002 |
| SpectralClustering | 0.743683 | 0.771792 | 0.760365 | 0.036002 |
| DBSCAN             | 0.568116 | 1.000000 | 0.579380 | 0.003000 |

> 分析可知，DBSCAN聚类速度最快，同质性指标值最高，达到了1.0，换言之，在DBSCAN算法中，聚类出来的每一个聚类都只包含真实类别中的一个类的数据对象，而完整性指标值最低，这是因为DBSCAN算法将低密度区域中的边缘数据对象当作噪声点抛弃，导致完整性不高。KMeans算法和SpectralClustering算法，聚类速度大致相同，SpectralClustering算法的评价指标略优于KMeans算法。MiniBatchKMeans与 KMeans聚类效果相同，但用时更短。

### 比较不同算法性能2（toy数据集）

```pythona
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

np.random.seed(0)

# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
n_samples = 1500
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                      noise=.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
no_structure = np.random.rand(n_samples, 2), None

colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
colors = np.hstack([colors] * 20)

clustering_names = [
    'MiniBatchKMeans', 'AffinityPropagation', 'MeanShift',
    'SpectralClustering', 'Ward', 'AgglomerativeClustering',
    'DBSCAN', 'Birch']

plt.figure(figsize=(len(clustering_names) * 2 + 3, 9.5))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                    hspace=.01)

plot_num = 1

datasets = [noisy_circles, noisy_moons, blobs, no_structure]
for i_dataset, dataset in enumerate(datasets):
    X, y = dataset
    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)

    # estimate bandwidth for mean shift
    bandwidth = cluster.estimate_bandwidth(X, quantile=0.3)

    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)

    # create clustering estimators
    ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    two_means = cluster.MiniBatchKMeans(n_clusters=2)
    ward = cluster.AgglomerativeClustering(n_clusters=2, linkage='ward',
                                           connectivity=connectivity)
    spectral = cluster.SpectralClustering(n_clusters=2,
                                          eigen_solver='arpack',
                                          affinity="nearest_neighbors")
    dbscan = cluster.DBSCAN(eps=.2)
    affinity_propagation = cluster.AffinityPropagation(damping=.9,
                                                       preference=-200)

    average_linkage = cluster.AgglomerativeClustering(
        linkage="average", affinity="cityblock", n_clusters=2,
        connectivity=connectivity)

    birch = cluster.Birch(n_clusters=2)
    clustering_algorithms = [
        two_means, affinity_propagation, ms, spectral, ward, average_linkage,
        dbscan, birch]

    for name, algorithm in zip(clustering_names, clustering_algorithms):
        # predict cluster memberships
        t0 = time.time()
        algorithm.fit(X)
        t1 = time.time()
        if hasattr(algorithm, 'labels_'):
            y_pred = algorithm.labels_.astype(np.int)
        else:
            y_pred = algorithm.predict(X)

        # plot
        plt.subplot(4, len(clustering_algorithms), plot_num)
        if i_dataset == 0:
            plt.title(name, size=18)
        plt.scatter(X[:, 0], X[:, 1], color=colors[y_pred].tolist(), s=10)

        if hasattr(algorithm, 'cluster_centers_'):
            centers = algorithm.cluster_centers_
            center_colors = colors[:len(centers)]
            plt.scatter(centers[:, 0], centers[:, 1], s=100, c=center_colors)
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.xticks(())
        plt.yticks(())
        plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                 transform=plt.gca().transAxes, size=15,
                 horizontalalignment='right')
        plot_num += 1

plt.show()
```

### 输出结果2

![](pictures\4.png)

##7 EM算法

### EM 算法步骤

- 输入：观测变量数据`Y`，隐变量数据`Z`，联合分布`P(Y,Z|θ)`，条件分布`P(Z|Y,θ)`.
- 输出：模型参数`θ`.

###EM算法求解高斯混合模型

```python
from __future__ import print_function
import numpy as np

def generateData(k,mu,sigma,dataNum):
    '''
    产生混合高斯模型的数据
    :param k: 比例系数
    :param mu: 均值
    :param sigma: 标准差
    :param dataNum:数据个数
    :return: 生成的数据
    '''
    # 初始化数据
    dataArray = np.zeros(dataNum,dtype=np.float32)
    # 逐个依据概率产生数据
    # 高斯分布个数
    n = len(k)
    for i in range(dataNum):
        # 产生[0,1]之间的随机数
        rand = np.random.random()
        Sum = 0
        index = 0
        while(index < n):
            Sum += k[index]
            if(rand < Sum):
                dataArray[i] = np.random.normal(mu[index],sigma[index])
                break
            else:
                index += 1
    return dataArray

def normPdf(x,mu,sigma):
    '''
    计算均值为mu，标准差为sigma的正态分布函数的密度函数值
    :param x: x值
    :param mu: 均值
    :param sigma: 标准差
    :return: x处的密度函数值
    '''
    return (1./np.sqrt(2*np.pi))*(np.exp(-(x-mu)**2/(2*sigma**2)))

def em(dataArray,k,mu,sigma,step = 10):
    '''
    em算法估计高斯混合模型
    :param dataNum: 已知数据个数
    :param k: 每个高斯分布的估计系数
    :param mu: 每个高斯分布的估计均值
    :param sigma: 每个高斯分布的估计标准差
    :param step:迭代次数
    :return: em 估计迭代结束估计的参数值[k,mu,sigma]
    '''
    # 高斯分布个数
    n = len(k)
    # 数据个数
    dataNum = dataArray.size
    # 初始化gama数组
    gamaArray = np.zeros((n,dataNum))
    for s in range(step):
        for i in range(n):
            for j in range(dataNum):
                Sum = sum([k[t]*normPdf(dataArray[j],mu[t],sigma[t]) for t in range(n)])
                gamaArray[i][j] = k[i]*normPdf(dataArray[j],mu[i],sigma[i])/float(Sum)
        # 更新 mu
        for i in range(n):
            mu[i] = np.sum(gamaArray[i]*dataArray)/np.sum(gamaArray[i])
        # 更新 sigma
        for i in range(n):
            sigma[i] = np.sqrt(np.sum(gamaArray[i]*(dataArray - mu[i])**2)/np.sum(gamaArray[i]))
        # 更新系数k
        for i in range(n):
            k[i] = np.sum(gamaArray[i])/dataNum

    return [k,mu,sigma]

if __name__ == '__main__':
    # 参数的准确值
    k = [0.3,0.4,0.3]
    mu = [2,4,3]
    sigma = [1,1,4]
    # 样本数
    dataNum = 5000
    # 产生数据
    dataArray = generateData(k,mu,sigma,dataNum)
    # 参数的初始值
    # 注意em算法对于参数的初始值是十分敏感的
    k0 = [0.3,0.3,0.4]
    mu0 = [1,2,2]
    sigma0 = [1,1,1]
    step = 6
    # 使用em算法估计参数
    k1,mu1,sigma1 = em(dataArray,k0,mu0,sigma0,step)
    # 输出参数的值
    print("参数实际值:")
    print("k:",k)
    print("mu:",mu)
    print("sigma:",sigma)
    print("参数估计值:")
    print("k1:",k1)
    print("mu1:",mu1)
    print("sigma1:",sigma1)
```

### 输出结果

```python
参数实际值:
k: [0.3, 0.4, 0.3]
mu: [2, 4, 3]
sigma: [1, 1, 4]
参数估计值:
k1: [0.54300716659941051, 0.19585407145739436, 0.26113876194319507]
mu1: [2.8156569715274342, 3.2809062129971607, 3.2809062129971629]
sigma1: [3.1837915155454017, 1.3907640263285175, 1.3907640263285221]
```