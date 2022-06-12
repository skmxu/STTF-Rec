import numpy as np

##设置全部数据，不输出省略号
import sys

np.set_printoptions(threshold=sys.maxsize)

boxes = np.load('./data/NYC_POI.npy')
print(boxes)
np.savetxt('./data/NYC_POI.txt', boxes, fmt='%s', newline='\n')
print('---------------------boxes--------------------------')
