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