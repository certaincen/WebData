from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import jieba
import sys
import re
import pickle
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.externals import joblib


class Decompose:
    def __init__(self, dimension, dec='SVD'):
        if dec == 'SVD':
            self.decomposition = TruncatedSVD(n_components=dimension)
        elif dec == 'NMF':
            self.decomposition = NMF(n_components=dimension, init='random')
        elif dec == 'LDA':
            self.decomposition = LatentDirichletAllocation(n_topics=dimension)

    def fit(self, trainX):
        print "dec start"
        self.decomposition = self.decomposition.fit(trainX)
        print "dec end"

    def transform(self, testX):
        return self.decomposition.transform(testX)



def load_stop_word(fname="stopword.txt"):
    fin = open(fname, 'r')
    content = fin.read().decode("utf-8")
    fin.close()
    stop_word = content.split('\n')
    return stop_word

def transform_data(content_list):
    url_list = []
    data = []
    for cont in content_list:
        url = cont[1]
        body = cont[3].decode('utf-8')
        url_list.append(url)
        data.append(body)
    return url_list, data

def save_feature(feature_list, fname):
    fout = open(fname, 'w')
    for feature in feature_list:
        fout.write(feature.encode("utf-8")+ '\n')
    fout.close()

def chinese_tokenizer(line):
    seg_list = jieba.cut(line, cut_all=False)
    for word in seg_list:
        if re.match(u"^[\u4e00-\u9fa5]+$", unicode(word)):
            yield word

def chinese_filter(word):
    if re.match(u"^[\u4e00-\u9fa5]+$", unicode(word)):
        return True
    return False

def build_tfidVector(stop_word, data, vocabulary=None):
    #tv = TfidfVectorizer(sublinear_tf = False, stop_words = stop_word, \
    #                     tokenizer = chinese_tokenizer, vocabulary=vocabulary, token_pattern=u'^[\u4e00-\u9fa5]+$')
    tv = TfidfVectorizer(sublinear_tf = False, stop_words = stop_word, \
                        vocabulary=vocabulary, min_df=1, max_df=0.8)
    tfidf_data = tv.fit_transform(data)
    #analyze = tv.build_analyzer()
    save_feature(tv.get_feature_names(), 'debug.txt')
    return tfidf_data, tv

def cluster_data(train_data, cluster_num):
    kmeans = KMeans(n_clusters=cluster_num, random_state=0)
    return kmeans.fit_predict(train_data)

def fit(train_data, cluster_num):
    kmeans = KMeans(n_clusters=cluster_num, random_state=0)
    return kmeans.fit(train_data)

def predict(kmeans, test_data):
    return kmeans.predict(test_data)


def save_label(url_list, label_list, fname):
    fout = open(fname, 'w')
    for i in range(len(url_list)):
        url = url_list[i]
        label = label_list[i]
        fout.write("%s\t%d\n" %(url, label))
    fout.close()


def save_model(clf, outfile):
    joblib.dump(clf, outfile)

def load_model(fname):
    clf = joblib.load(fname)
    return clf 

def basic_tokenizer(line):
    seg_list = jieba.cut(line, cut_all=False)
    return seg_list

def save_word_seg(X, Y, fname):
    fout = open(fname, 'w')
    for i in range(len(X)):
        word_list = []
        #for word in util.chinese_tokenizer(X[i]):
        for word in basic_tokenizer(X[i]):
            word_list.append(word)
        word_str = '\t'.join(word_list)
        fout.write(word_str.encode('gb18030'))
        fout.write('\t' + str(Y[i]) + '\n')
    fout.close()

def load_word_seg(fname):
    fin = open(fname, 'r')
    content = fin.read().strip()
    fin.close()
    lines = content.split('\n')
    X_list = []
    y_list = []
    for line in lines:
        items = line.split('\t')
        y = items[-1]
        x = items[:-1]
        X_list.append('\t'.join(x).decode("gb18030"))
        y_list.append(y)
    return X_list, y_list

def save_vocabulary(tv, fname):
    with open(fname, 'wb') as handle:
        pickle.dump(tv, handle)

def load_vocabulary(fname):
    tv = pickle.load(open(fname, 'rb'))
    return tv
