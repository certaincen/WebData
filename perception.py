import numpy as np

def loadDataset():
    filename = 'webdata/head.1000'
    train_data = [[float(i) for i in line.split()] for line in open(filename)][200:1000]
    return np.array(train_data)

def loadTestDataset():
    filename = 'webdata/head.1000'
    test_data = [[float(i) for i in line.split()] for line in open(filename)][0:200]
    return np.array(test_data)

def train(train_data):
    w=np.zeros(train_data.shape[1])+1
    maxtime=1000;
    for t in range(maxtime):
        for i in range(train_data.shape[0]):
            ans=np.dot(w,train_data[i])
            if ans<=0:
                w=w-0.1*train_data[i]
    return w;

def test(w,test_set,test_label):
    a=0;
    for i in range(test_set.shape[0]):
        tmp=np.dot(w,test_set[i])
        if tmp>0 and test_label[i]==0:
            a=a+1;
            continue
        if tmp<0 and test_label[i]==1:
            a=a+1;
    return float(a)/float(test_set.shape[0])

def perception(train_set,test_set):
    d = len(train_set[0])
    train_label = train_set[:, d - 1]
    test_label = test_set[:, d - 1]
    total = len(test_set)
    for i in range(total):
        train_set[i][d-1]=1
        if train_label[i]==1:
            train_set[i]=-train_set[i]
    w=train(train_set)
    ac=test(w,test_set,test_label)
    return ac

if __name__ == '__main__':
    train_set = loadDataset()
    test_set = loadTestDataset()
    ac=perception(train_set, test_set)
    print(ac)