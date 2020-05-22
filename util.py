import numpy as np

def get10fold(data, turn):
    tot_length = len(data)
    each = int(tot_length/10)
    mask = np.array([True if each*turn <= i < each*(turn+1) else False
                     for i in list(range(tot_length))])
    return data[~mask], data[mask]

def RunCV(clf, data, labels, isAcc = True):
    from sklearn.metrics import precision_recall_fscore_support
    accuracies = []

    for i in range(10):
        data_tr, data_te = get10fold(data, i)
        labels_tr, labels_te = get10fold(labels,i)

        clf = clf.fit(data_tr, labels_tr) #모델 학습
        pred = clf.predict(data_te) #모델 예측
        correct = pred == labels_te #레이블과 비교했을 떄 얼마나 같은지

        if isAcc:
            acc = sum([1 if x == True else 0 for x in correct]) / len(correct) #True, False 비율 계산
            accuracies.append(acc)
        else:
            accuracies.append(precision_recall_fscore_support(pred, labels_te))
        accuracies.append(acc)

    return accuracies

def load_mnist(path):
    import numpy as np
    f = open(path)

    f.readline() #맨 위 헤더부분을 없앰
    digits = []
    digit_labels = []
    for line in f.readlines(): #전체 데이터를 읽어옴
        spplited = line.replace("\n", "").split(",")
        digit = np.array(spplited[1:], dtype=np.float32)
        label = int(spplited[0]) #강제 형변환
        digits.append(digit)
        digit_labels.append(label) #추가
    digits = np.array(digits)
    digit_labels = np.array(digit_labels) #하나의 행렬로 만듬
    norm_digits = digits/255 #정규화
    print(digits.shape) #(42000, 784)
    return norm_digits, digit_labels