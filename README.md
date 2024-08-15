# ML_
Programming on ML

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics

names=['sepal-length','sepal-width','petal-length','petal-width','Class']
dataset=pd.read_csv("/content/8-dataset.csv",names=names)
dataset

sepal-length	sepal-width	petal-length	petal-width	Class
0	5.1	3.5	1.4	0.2	Iris-setosa
1	4.9	3.0	1.4	0.2	Iris-setosa
2	4.7	3.2	1.3	0.2	Iris-setosa
3	4.6	3.1	1.5	0.2	Iris-setosa
4	5.0	3.6	1.4	0.2	Iris-setosa
...	...	...	...	...	...
145	6.7	3.0	5.2	2.3	Iris-virginica
146	6.3	2.5	5.0	1.9	Iris-virginica
147	6.5	3.0	5.2	2.0	Iris-virginica
148	6.2	3.4	5.4	2.3	Iris-virginica
149	5.9	3.0	5.1	1.8	Iris-virginica
dataset.describe()

sepal-length	sepal-width	petal-length	petal-width
count	150.000000	150.000000	150.000000	150.000000
mean	5.843333	3.054000	3.758667	1.198667
std	0.828066	0.433594	1.764420	0.763161
min	4.300000	2.000000	1.000000	0.100000
25%	5.100000	2.800000	1.600000	0.300000
50%	5.800000	3.000000	4.350000	1.300000
75%	6.400000	3.300000	5.100000	1.800000
max	7.900000	4.400000	6.900000	2.500000
X=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]
print(X.head())
   sepal-length  sepal-width  petal-length  petal-width
0           5.1          3.5           1.4          0.2
1           4.9          3.0           1.4          0.2
2           4.7          3.2           1.3          0.2
3           4.6          3.1           1.5          0.2
4           5.0          3.6           1.4          0.2
X.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 150 entries, 0 to 149
Data columns (total 4 columns):
 #   Column        Non-Null Count  Dtype  
---  ------        --------------  -----  
 0   sepal-length  150 non-null    float64
 1   sepal-width   150 non-null    float64
 2   petal-length  150 non-null    float64
 3   petal-width   150 non-null    float64
dtypes: float64(4)
print(y.head())
]
print(y.head())
0    Iris-setosa
1    Iris-setosa
2    Iris-setosa
3    Iris-setosa
4    Iris-setosa
Name: Class, dtype: object
y.info()
<class 'pandas.core.series.Series'>
RangeIndex: 150 entries, 0 to 149
Series name: Class
Non-Null Count  Dtype 
--------------  ----- 
150 non-null    object
dtypes: object(1)
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.30)
Xtrain.info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 105 entries, 104 to 15
Data columns (total 4 columns):
 #   Column        Non-Null Count  Dtype  
---  ------        --------------  -----  
 0   sepal-length  105 non-null    float64
 1   sepal-width   105 non-null    float64
 2   petal-length  105 non-null    float64
 3   petal-width   105 non-null    float64
dtypes: float64(4)
memory usage: 4.1 KB
Xtest.info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 45 entries, 49 to 111
Data columns (total 4 columns):
 #   Column        Non-Null Count  Dtype  
---  ------        --------------  -----  
 0   sepal-length  45 non-null     float64
 1   sepal-width   45 non-null     float64
 2   petal-length  45 non-null     float64
 3   petal-width   45 non-null     float64
dtypes: float64(4)
memory usage: 1.8 KB
ytrain.info()
<class 'pandas.core.series.Series'>
Int64Index: 105 entries, 104 to 15
Series name: Class
Non-Null Count  Dtype 
--------------  ----- 
105 non-null    object
dtypes: object(1)
memory usage: 1.6+ KB
<class 'pandas.core.series.Series'>
Int64Index: 45 entries, 49 to 111
Series name: Class
Non-Null Count  Dtype 
--------------  ----- 
45 non-null     object
dtypes: object(1)
memory usage: 720.0+ bytes
