# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 16:03:43 2017

@author: wuho
"""
violations = {
    '采购策略问题': 0,
    '串标':1,
    '虚假业务':2,
    '收受回扣':3,
    '流程违规':4,
    '成本偏高':5,
}

#全部使用`(value - min) / (max - min)`的方法进行标准化，因此需要将`minval`和`denom`数组保存在文件开头。这种线性标准化的方法对数据做了一定预设，因此也许不是最佳方案。事实上，最佳的标准化方案有待商榷，也是这个课题未来发展中亟待解决的一个问题。
minval = [1.0, 0, 3401001, -100000000, 101, 1000, -100000000]
denom = [1000052162559.0, 88717952, 1100000, 200000000, 300, 2780, 200000000]

def print_data(str, data):
    from pprint import pprint
    if hasattr(data, "__len__"):
        print("### %s [%s], length of %d, detail:" %(str, type(data), len(data)))
    else:
        print("### %s [%s], detail:" %(str, type(data)))  
    pprint(data)


# normalize floating point data, based on a pre-set assumption of
# maximum and minimum values
def normalize(data):
    import numpy as np
    ret = np.empty(data.shape)#生成一个 和data 一样shape 的空数组，不知为何数组的值不全是0
    #print_data("data",data)
    arr_max = data.max(axis=0)
    #print (arr_max)
    arr_min = data.min(axis=0)
    #print (arr_min)
    arr_max_minus_min=arr_max-arr_min
    for i in range(len(arr_max_minus_min)):
        if arr_max_minus_min[i]==0:
            arr_max_minus_min[i]=arr_max[i]
    #print (arr_max_minus_min)
    line_num=data.shape[0]
    column_num=data.shape[1]
    #print("======================begin")
    debugflag=False
    for i in range(column_num): #一个 iteration 处理 1列数据 ,
        #从0 起分别是 "supplier_ID","净价","物料组","订单净值","采购组","采购组织","采购订单数量"
        #print("val_min )
        arr_min_duplicated = np.full((line_num,), arr_min[i]) #其实就是生成一个一维的 ndarray， 长度为line_num 每个的值都是 minval[i]
        if debugflag:
            print("==========begin of %d========= " %(i))
            print_data("min_arr",arr_min_duplicated)
            print_data("data[:,i]",data[:,i]) #即为第i列，同理 data[j,:]为第j行
            #print_data("data[i,:]",data[i,:])
        ret[:,i] = (data[:,i] - arr_min_duplicated) / arr_max_minus_min[i] #第i列 换成此 array, min_arr 和 denom 是预定义的，其中 denom 是本列的 max-min 值
        #注意，上行的分母却可以不必是一个ndarray
        if debugflag:
            print_data("ret[:,i]",ret[:,i])
            print("\n\n\n")
    #print("======================xxxxxxxxxxxxxxxxxx")
    #print_data("2",ret)
    return ret
