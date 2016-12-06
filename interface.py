import util
import feature
import numpy as np

tv = util.load_vocabulary("vocabulary.pkl")
clk =util.load_model("model.plk")
dec = util.load_model("decom.plk")
while True:
    line = raw_input("input massage,input # quit\n")
    if line == '#':
        break
    word_list = feature.cut_word([line])
    X_test_add = feature.build_feature(word_list)
    X_test = tv.transform(word_list)
    X_test = dec.transform(X_test)
    X_test = np.hstack((X_test, X_test_add))
    Y = clk.predict(X_test)
    for y in Y:
        if y == 1:
            print "spame"
        else:
            print "unspam"



