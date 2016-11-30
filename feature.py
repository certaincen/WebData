import sys
import re
import numpy as np
import scipy
import util
import math
import gc
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


def load_data(fname):
    fin = open(fname, 'r')
    label_list = []
    content_list = []
    while True:
        line = fin.readline()
        if not line:
            break
        items = line.strip().decode('utf-8').split('\t')
        label_list.append(items[0])
        content_list.append(items[1])
    fin.close()
    return label_list, content_list

def train_and_predict(clf, trainX, trainY, testX):
    print "train start"
    clf.fit(trainX, trainY)
    print "predict start"
    return clf.predict(testX)

def class_metrics(y_pred, y_true):
    target_names = ['unspam', 'spam']
    print(classification_report(y_true, y_pred, target_names=target_names))

def build_chinese_rate(sentence):
    hanzi_count = 0
    for ch in sentence:
        if util.chinese_filter(ch):
            hanzi_count += 1
    return 1 - float(hanzi_count)/len(sentence)

def build_x_num(sentence):
    pass


def build_feature(contentX):
    X = []
    for i in range(len(contentX)):
        sentence = ' '.join(contentX[i])
        tmp_list = []
        tmp_list.append(math.log(len(sentence),2))
        tmp_list.append(build_chinese_rate(sentence))
        X.append(tmp_list)
    X = np.array(X)
    X = preprocessing.normalize(X)
    return X

def save_result(index, fname, precision, f1, recall):
    fout = open(fname, 'a')
    fout.write("%d\t" %index)
    print precision
    class0 = "%f\t%f\t%f\t" %(precision[0], recall[0], f1[0])
    class1 = "%f\t%f\t%f\n" %(precision[1], recall[1], f1[1])
    print class0
    print class1
    fout.write(class0)
    fout.write(class1)
    fout.close()

def run(trainX, train_add_X, trainY, testX, test_add_X, testY, model_name, weight_dict, dec_name, dimension, index, fname):
    clf = None
    if model_name == 'K':
        clf = KNeighborsClassifier(class_weight = weight_dict)
    elif model_name == 'L':
        clf = LogisticRegression(class_weight = weight_dict)
        #clf = LogisticRegression()
    elif model_name == 'R':
        clf = RandomForestClassifier(class_weight = weight_dict)
    else:
        #clf = SVC()
        clf = LinearSVC(class_weight = weight_dict)
    print "iter dimension value %d" %dimension
    decompose = util.Decompose(dimension, dec_name)
    print "build feature start"
    decompose.fit(trainX)
    X_train = trainX
    X_test = testX
    X_train = decompose.transform(trainX)
    X_test = decompose.transform(testX)
    X_train = np.hstack((X_train, train_add_X))
    X_test = np.hstack((X_test, test_add_X))
    #X_train = scipy.sparse.hstack((X_train, train_add_X))
    #X_test = scipy.sparse.hstack((X_test, test_add_X))
    #X_train = train_add_X
    #X_train = train_add_X
    #X_test = test_add_X
    print 'Size of fea_train:' + repr(X_train.shape)
    print 'Size of fea_train:' + repr(X_train.shape)
    print "build feature end"
    predict_y = train_and_predict(clf, X_train, trainY, X_test)
    precision = precision_score(testY, predict_y, average=None)
    #print testY[:10]
    #print predict_y[:10]
    f1 = f1_score(testY, predict_y, average=None)
    recall = recall_score(testY, predict_y, average=None)
    save_result(index, fname, precision, f1, recall)
    class_metrics(predict_y, testY)
    del decompose
    del X_train
    del X_test
    del predict_y
    gc.collect()


def feature_collection():
    stop_word = util.load_stop_word()
    #print "start split cross value"
    #trainX, testX, trainY, testY = train_test_split(content_list, label_list, test_size=0.4, random_state=0)
    #trainX = cut_word(trainX)
    #testX = cut_word(testX)
    #util.save_word_seg(trainX, trainY, "word_seg.train")
    #util.save_word_seg(testX, testY, "word_seg.test")
    #sys.exit(-1)
    trainX, trainY = util.load_word_seg("word_seg.train")
    testX, testY = util.load_word_seg("word_seg.test")
    trainY = np.array(trainY, dtype=np.int)
    testY = np.array(testY, dtype=np.int)
    print "build ex feature"
    train_add_X = []
    test_add_X = []
    train_add_X = build_feature(trainX)
    test_add_X = build_feature(testX)
    print "build ex feature end"
    trainX, tv = util.build_tfidVector(stop_word, trainX, None)
    testX, tv = util.build_tfidVector(stop_word, testX, tv.vocabulary_)
    return trainX, train_add_X, trainY, testX, test_add_X, testY

def load_parameter(fname):
    fin = open(fname, 'r')
    content = fin.read().strip()
    fin.close()
    res_list  = []
    for line in content.split('\n'):
        para = line.split('\t')
        res_list.append(para)
    return res_list

def load_stop_index(fname):
    fin = open(fname, 'r')
    num = fin.read()
    fin.close()
    num = int(num)
    return num

if __name__=='__main__':
    index = 0
    try:
        index = load_stop_index(sys.argv[3])
    except:
        index = 0
    para_list = load_parameter(sys.argv[1])
    outfile = sys.argv[2]
    trainX, train_add_X, trainY, testX, test_add_X, testY = feature_collection()
    for i in range(index, len(para_list)):
        model_name = para_list[i][0]
        weight = int(para_list[i][1])
        weight_dict = {}
        weight_dict[0] = 1
        weight_dict[1] = weight
        dec_name = para_list[i][2]
        dimension = int(para_list[i][3])
        #run(trainX, train_add_X, trainY, testX, test_add_X, testY, model_name, weight_dict, dec_name, dimension, i, outfile)
        #sys.exit(-1)
        try:
            run(trainX, train_add_X, trainY, testX, test_add_X, testY, model_name, weight_dict, dec_name, dimension, i, outfile)
        except Exception, e:
            print e.message
            ofile = open(sys.argv[3], 'w')
            ofile.write(str(i))
            ofile.close()





