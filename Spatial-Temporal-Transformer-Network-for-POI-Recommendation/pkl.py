# # -*- coding: UTF8 -*-
#
# # cPickle是python2系列用的，3系列已经不用了，直接用pickle就好了
# import pickle
#
# # 重点是rb和r的区别，rb是打开2进制文件，文本文件用r
# f = open('foursquare_cut_one_day.pkl','rb')
# data = pickle.load(f)
# print(data)
#
import sys
sys.getdefaultencoding()
import pickle
import numpy as np
np.set_printoptions(threshold=1000000000000000)
path = "data/NYC_data.pkl"
file = open(path,'rb')
inf = pickle.load(file)        #读取pkl文件的内容
#print(inf)
#fr.close()
inf=str(inf)
obj_path = 'data/NYC_pkl.txt'
ft = open(obj_path, 'w')
ft.write(inf)