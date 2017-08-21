
# -*- coding: utf-8 -*-
# constants, violation dictionary, and data used for normalization

import shared as s

cur_mode="debug"

if cur_mode=="debug":
    const_batch = 24; const_folds = 6; const_epochs=1
    pos_file="data/pos-few.csv";neg_file="data/neg-few.csv"
else:
    const_batch = 128; const_folds = 10; const_epochs=2
    pos_file="data/pos-200.csv";neg_file="data/neg-200.csv"

# start of the program, handles the parsing of arguments
def init_program():
    import argparse #this is needed only in init_program
    parser = argparse.ArgumentParser()

    #parser.add_argument('--epochs', action='store', required=True,type=int, help='specify the epochs to train the model')

    gp = parser.add_mutually_exclusive_group(required=True) 
    #Create a mutually exclusive group. argparse will make sure that only one of the arguments in the mutually exclusive group was present on the command line
    gp.add_argument('-b', '--binary', action='store_true',
            help='classify between suspicious or not suspicious')
    gp.add_argument('-m', '--multiclass', action='store_true',
            help='determine the category of suspicious actions')

    parser.add_argument('-s', '--save', action='store_true',
            help='to save the model instead of cross validation')

    global args; args = parser.parse_args()
    global csv; import csv
    global np; import numpy as np

# process data for the binary classification
def process_b(data):
    str_data = np.delete(data[1:, 1:], [7], axis=1)
    for i in range(str_data.shape[0]):
        str_data[i,5] = '0' if str_data[i,5] == 'normal' else '1' #所有的positive csv里 这一列都是 normal
    return str_data.astype('float32') # why？
    #return str_data

# process data for the multiclass classification
def process_m(data):
    #data[1:, 1:] 表示省略掉 0行 与0 列，第一行第一列开始取这样就去掉了 表头（0行）与采购凭证号（0列）
    #print(type(data));
    #s=data.shape
    #s.print_data("s the shape",s)
    str_data = np.delete(data[1:, 1:], [7], axis=1)  #去掉第7列， 即工厂， axis=0 的话则删掉第7行
    #s.print_data("str_data 1",str_data)
    #print("shape[0] is %d" %(str_data.shape[0]))
    line_num=str_data.shape[0];
    for i in range(line_num):
        varray=str_data[i, 5].split('；') # 有；号的话变array，例如 流程违规；成本偏高
        #print("linenum %d violation description is %s" %(i, varray[0]))
        vkey=varray[0]
        if(vkey in s.violations):
            str_data[i, 5] = s.violations[varray[0]]
        else:
            print("csv file issue, it contains a violation bit that's not existent in violation dict [%s], line num is about %d" %(vkey,i))
            s.print_data("violations dictionary",s.violations)
            exit()
    #s.print_data("str_data 2",str_data)
    ret=str_data.astype('float32')
    #print("length is %d" %(ret.shape[0]))
    return ret



# returns a list of indices, indicating the training/testing data
# for each iteration of the cross validation
def kfold(dim, k):
    #print("in function kfold dim is %d and k is %d" %(dim,k))
    remainder=dim % k
    if remainder!=0:
        exit("data lenth %d is not interger times of k %d" %(dim,k))
    #assert(dim % k == 0)
    indices = np.arange(dim); np.random.shuffle(indices)

    ret_train = []; ret_test = []
    for i in range(k):#kfold 即k次学习与测试 的 iteration 
        ret_train.append(list(np.concatenate((
                indices[0:i*dim//k], indices[(i+1)*dim//k:]))))
        ret_test.append(list(indices[i*dim//k:(i+1)*dim//k]))
                #indices[0:i*dim//10], indices[(i+1)*dim//10:]))))#hardcode 应该是错的
        #ret_test.append(list(indices[i*dim//10:(i+1)*dim//10]))#hardcode 应该是错的
    #s.print_data("ret_train",ret_train)
    #s.print_data("ret_test",ret_test)
    return zip(ret_train, ret_test)


# load data for the binary classification
def load_process_data_b():
    f1 = open(pos_file, 'r' ,encoding='utf-8')
    f2 = open(neg_file, 'r' ,encoding='utf-8')
    r1 = csv.reader(f1, delimiter=',', quotechar='"')
    r2 = csv.reader(f2, delimiter=',', quotechar='"')

    pos_data = process_b(np.array(list(r1)))
    neg_data = process_b(np.array(list(r2)))
    data = np.concatenate((pos_data, neg_data), axis=0)
    np.random.shuffle(data)

    #x_data = np.delete(data, [5], axis=1)#第5列是 violation 其实也就是train 中的y_data 因此删掉
    x_data = s.normalize(np.delete(data, [5], axis=1)) #第5列是 violation 其实也就是train 中的y_data，因此删掉
    #s.print_data("normalized data", x_data)
    y_data = data[:, 5]
    #len_data = data.shape[0] // 1280 * 1280
    #print("25016 trimmed is %d" %(25016 // 1280 * 1280))
    #len_data = data.shape[0]#hongwei 的改动，因为不明白为何需要 除以1280 取整后再 乘回来1280，数字少的时候结果就直接变0了，无法debug了
    len_data = data.shape[0] // const_folds * const_folds #20156 --> 20150

    f1.close(); f2.close()
    return x_data[:len_data], y_data[:len_data]


# load data for the multiclass classification
def load_process_data_m():
    f1 = open(neg_file, 'r', encoding='utf-8')
    r1 = csv.reader(f1, delimiter=',', quotechar='"')
    #s.print_data("r1", r1)
    data_list=np.array(list(r1))
    #s.print_data("data_list",data_list)
    data = process_m(data_list) 
    #s.print_data("str_data3",data)
    #s=data.shape
    #s.print_data("s the shape",s)
    #len_data = data.shape[0] // 1280 * 1280
    #len_data = data.shape[0]
    len_data = data.shape[0] // const_folds * const_folds #by hongwei: 20156 --> 20150  

    #print("=========================> len_data is %d" %(len_data))
    x_data = s.normalize(np.delete(data, [5], axis=1))
    #s.print_data("x_data after normalize",x_data)
    y_data = data[:, 5] #只取二维数组的第五列
    #s.print_data("data[:, 5]",y_data)
    #s.print_data("y_data after takiing data[:, 5]", y_data)
    f1.close()
    #s.print_data("x_data[:len_data]",x_data[:len_data])
    return x_data[:len_data], y_data[:len_data]


# one iteration of cross validation consists of training just one
# instance of our model, and immediately test it
def train_test(x_train, y_train, x_test, y_test,i_cur):
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(
            x_train.shape[1],)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    global summary_drawn
    if i_cur==0:
        model.summary() #打印出当前的model 

    model.compile(loss='categorical_crossentropy',
            optimizer=RMSprop(), #hongwei:为何不使用 SGD （stochastic gradient descent 随机梯度下降）或者其他算法，Adam ？
            metrics=['accuracy'])

    history = model.fit(x_train,y_train,
            batch_size=const_batch,
            epochs=const_epochs,
            verbose=1,
            validation_data=(x_test,y_test))
    meva=model.evaluate(x_test,y_test,verbose=0)
    #s.print_data("meva",meva)
    return meva

# train the model based on all the given data, and then save it
def train_save(x_train, y_train):
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(
            x_train.shape[1],)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()

    model.compile(loss='categorical_crossentropy',
            optimizer=RMSprop(),
            metrics=['accuracy'])

    history = model.fit(x_train, y_train,
            batch_size=const_batch,
            epochs=const_epochs,
            verbose=1)

    fname = 'model/binary.h5' if args.binary else 'model/multi.h5'
    model.save(fname)
    print("\n----------------------------------------\n")
    print('successfully saved the model at', fname)

# conduct the cross validation, take average and print results
def cross_validation(x_data, y_data):
    #print(x_data.shape[0] // 10 * 9, 'train samples')
    #print(x_data.shape[0] // 10, 'test samples')
    print(x_data.shape[0] // const_folds * (const_folds-1), 'train samples')
    print(x_data.shape[0] // const_folds, 'test samples')
    debugflag=False
    s.print_data("const_folds",const_folds)
    loss = []; accuracy = []
    kzipped=kfold(x_data.shape[0], const_folds)
    #s.print_data("kfold",kzipped)
    cur=0
    for train_idx, test_idx in kzipped:
        x_train, x_test = x_data[train_idx], x_data[test_idx]
        y_train, y_test = y_data[train_idx], y_data[test_idx]
        
        if debugflag:
            print("===================== train_test iteration [%d]" %(cur))
            s.print_data("train_idx",train_idx)
            s.print_data("x_train", x_train)
            s.print_data("y_train", y_train)
            s.print_data("test_idx",test_idx)
            s.print_data("x_test", x_test)
            s.print_data("y_test", y_test)
        
        score = train_test(x_train, y_train, x_test, y_test,cur)
        if debugflag:
            s.print_data("score", score)
        loss.append(score[0])
        accuracy.append(score[1])
        if debugflag:
            s.print_data("loss", loss)
            s.print_data("accuracy", accuracy)
            print("\n\n\n")
            exit("won't go to next iteration")
        s.print_data("score", score)
        
        cur=cur+1

    print("\n----------------------------------------\n")
    print("run #\ttest loss\ttest accuracy") #画表头

    total_loss = total_accuracy = 0.0
    for i in range(const_folds):
        print("%d\t%f\t%f" % (i, loss[i], accuracy[i]))
        total_loss += loss[i]; total_accuracy += accuracy[i]

    print("\naverage loss:", total_loss/const_folds)
    print("average accuracy:", total_accuracy/const_folds)

# main routine for ML model training
def main():
    import time
    ts_start=time.time()
    init_program()
    x_data = None; y_data = None; global num_classes
    #s.print_data("ini x_data",x_data)
    if args.binary: #hongwei 只训练 nn 如何判断 0、1
        #print("passed in argument is binary");
        num_classes = 2
        x_data, y_data = load_process_data_b()
    else: #multicast 表示需要判断不好的情况下到底属于那种 violations/舞弊 
        #print("passed in argument is multiclass");
        num_classes = len(s.violations)
        x_data, y_data = load_process_data_m()
    #exit("debugging, won't train...")
    #print("data loaded... now will execute to_categorical function...");
    #s.print_data("y_data before",y_data)
    global to_categorical; from keras.utils import to_categorical
    y_data = to_categorical(y_data, num_classes)#其实就是把数字转成了二维数组，长度为 Y_data 的行数，列数为违规的种类数
    #s.print_data("y_data after (to_categorical)",y_data)
    global Sequential; from keras.models import Sequential
    global Dense,Dropout; from keras.layers import Dense, Dropout
    global RMSprop; from keras.optimizers import RMSprop
    global bk; from keras import backend as bk
    
    if args.save: 
        print("argument passed in -s. Now will execute train_save function...");
        train_save(x_data, y_data)
    else: 
        print("argument no -s. Therefore now will do cross_validation...");
        cross_validation(x_data, y_data)
    bk.clear_session()
    ts_end=time.time()
    seconds_used=ts_end-ts_start
    print("main will end...total seconds: %d" %(seconds_used));

# standard python 3 routine to invoke the main() function
if __name__ == '__main__': main()
