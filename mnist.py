from random import shuffle, seed
import time
seed(0)
import numpy as np
import util

#Data Set, Data load
path = 'train_mnist.csv'
norm_digits, digit_labels = util.load_mnist(path)

#Decison tree
from sklearn.tree import DecisionTreeClassifier
numbers1 = list(range(len(norm_digits)))
shuffle(numbers1)
clf1 = DecisionTreeClassifier()
shuffled_data1 = norm_digits[numbers1]
shuffled_labels1 = digit_labels[numbers1]
st1 = time.time()
acc1 = util.RunCV(clf1, shuffled_data1, shuffled_labels1, isAcc=True) #정확도
et1 = time.time() - st1 #실행 시간 측정
print(np.mean(acc1))
print(et1)

#Naive Bayesian
from sklearn.naive_bayes import GaussianNB
numbers2 = list(range(len(norm_digits)))
shuffle(numbers2)
clf2 = GaussianNB()
shuffled_data2 = norm_digits[numbers2]
shuffled_labels2 = digit_labels[numbers2]
st2 = time.time()
acc2 = util.RunCV(clf2, shuffled_data2, shuffled_labels2, isAcc=True)
et2 = time.time() - st2
print(np.mean(acc2))
print(et2)

#KNN
from sklearn.neighbors import KNeighborsClassifier
numbers3 = list(range(len(norm_digits)))
shuffle(numbers3)
clf3 = KNeighborsClassifier()
shuffled_data3 = norm_digits[numbers3]
shuffled_labels3 = digit_labels[numbers3]
st3 = time.time()
acc3 = util.RunCV(clf3, shuffled_data3, shuffled_labels3, isAcc=True)
et3 = time.time() - st3
print(np.mean(acc3))
print(et3)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
numbers4 = list(range(len(norm_digits)))
shuffle(numbers4)
clf4 = LogisticRegression()
shuffled_data4 = norm_digits[numbers4]
shuffled_labels4 = digit_labels[numbers4]
st4 = time.time()
acc4 = util.RunCV(clf4, shuffled_data4, shuffled_labels4, isAcc=True)
et4 = time.time() - st4
print(np.mean(acc4))
print(et4)

#Perceptron
from sklearn.linear_model import Perceptron
numbers5 = list(range(len(norm_digits)))
shuffle(numbers5)
clf5 = Perceptron(max_iter=500, n_jobs=3)
shuffled_data5 = norm_digits[numbers5]
shuffled_labels5 = digit_labels[numbers5]
st5 = time.time()
acc5 = util.RunCV(clf5, shuffled_data5, shuffled_labels5, isAcc=True)
et5 = time.time() - st5
print(np.mean(acc5))
print(et5)

#Multi-Layer Perceptron
from sklearn.neural_network import MLPClassifier
numbers6 = list(range(len(norm_digits)))
shuffle(numbers6)
clf6 = MLPClassifier(hidden_layer_sizes=20, max_iter=500)
shuffled_data6 = norm_digits[numbers6]
shuffled_labels6 = digit_labels[numbers6]
st6 = time.time()
acc6 = util.RunCV(clf6, shuffled_data6, shuffled_labels6, isAcc=True)
et6 = time.time() - st6
print(np.mean(acc6))
print(et6)

#RandomForest
from sklearn.ensemble import RandomForestClassifier
numbers7 = list(range(len(norm_digits)))
shuffle(numbers7)
clf7 = RandomForestClassifier()
shuffled_data7 = norm_digits[numbers7]
shuffled_labels7 = digit_labels[numbers7]
st7 = time.time()
acc7 = util.RunCV(clf7, shuffled_data7, shuffled_labels7, isAcc=True)
et7 = time.time() - st7
print(np.mean(acc7))
print(et7)

#Linear Discriminant Analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
numbers8 = list(range(len(norm_digits)))
shuffle(numbers8)
clf8 = LinearDiscriminantAnalysis()
shuffled_data8 = norm_digits[numbers8[:10000]]
shuffled_labels8 = digit_labels[numbers8[:10000]]
st8 = time.time()
acc8 = util.RunCV(clf8, shuffled_data8, shuffled_labels8, isAcc=True)
et8 = time.time() - st8
print(np.mean(acc8))
print(et8)

#Quad DA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
numbers9 = list(range(len(norm_digits)))
shuffle(numbers9)
clf9 = QuadraticDiscriminantAnalysis()
shuffled_data9 = norm_digits[numbers9]
shuffled_labels9 = digit_labels[numbers9]
st9 = time.time()
acc9 = util.RunCV(clf9, shuffled_data9, shuffled_labels9, isAcc=True)
et9 = time.time() - st9
print(np.mean(acc9))
print(et9)

#SVC
from sklearn.svm import SVC
numbers10 = list(range(len(norm_digits)))
shuffle(numbers10)
clf10 = SVC()
shuffled_data10 = norm_digits[numbers10[:10000]]
shuffled_labels10 = digit_labels[numbers10[:10000]]
st10 = time.time()
acc10 = util.RunCV(clf10, shuffled_data10, shuffled_labels10, isAcc=True)
et10 = time.time() - st10
print(np.mean(acc10))
print(et10)
