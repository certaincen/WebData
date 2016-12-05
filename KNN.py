import numpy as np

def loadDataset():
    filename = 'webdata/head.1000'
    train_data = [[float(i) for i in line.split()] for line in open(filename)][0:800]
    return np.array(train_data)

def loadTestDataset():
    filename = 'webdata/head.1000'
    test_data = [[float(i) for i in line.split()] for line in open(filename)][800:1000]
    return np.array(test_data)

def KNN_classify(test_data,train_data,train_label,k):
    num=len(train_data)
    class1=0
    d=len(train_data[0])
    y=list()
    for i in range(num):
        tmp=abs(train_data[i]-test_data)
        y.append(sum(tmp))
    tmp=np.array(y)
    tmp=np.where(tmp<=np.sort(tmp)[k-1])[0]
    for i in tmp:
        if train_label[i]==0:
            class1=class1+1
    return int(class1<(k/2))



    return sum
def KNN_judge(train_set,test_set,k):
    d=len(train_set[0])
    train_data=train_set[:,0:d-1]
    train_label=train_set[:,d-1]
    test_data=test_set[:,0:d-1]
    test_label=test_set[:,d-1]
    a=0;
    total=len(test_set)
    for i in range(total):
        if test_label[i]==KNN_classify(test_data[i],train_data,train_label,k):
            a=a+1
    return float(a)/float(total)

if __name__ == '__main__':
    train_set = loadDataset()
    test_set = loadTestDataset()
    k = 6
    ac=KNN_judge(train_set, test_set,k)
    print(ac)








