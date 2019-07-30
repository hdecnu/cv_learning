"""
kmeans算法,初始点随机筛选
"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets

#随机产生k个初始点
def random_center(data,k):
    row = data.shape[0]
    m = np.random.choice(range(0,row),k,replace = False)
    center=data[m,:]
    return center  

#把每个点分到对应的类别
def class_index(data,center):
    k = center.shape[0]
    distance = np.zeros((data.shape[0],k))
    for i in range(k):
        distance[:,i] = np.sum((data - center[i,:])**2,axis=1) #每个点都计算与k个中心点的距离
    index = np.argmin(distance,axis = 1)
    return index

#更新中心点
def update_center(data,index,center):
    k = center.shape[0]
    col = data.shape[1]
    update_center = np.zeros((k,col))
    for i in range(k):
        update_center[i,:] = np.mean(data[index == i,:],axis=0)
    return update_center

#主函数迭代，当前后中心点相同时停止迭代
def main():
    iris=datasets.load_iris()#引入iris鸢尾花数据,iris数据包含4个特征变量
    data=iris.data
    k = 4
    center = random_center(data,k)
    for i in range(100):
        index = class_index(data,center)
        center_new = update_center(data,index,center)
        if (center_new == center).all():
            break
        else:
            center = center_new
    plt.scatter(data[:,0],data[:,1],c = index)
    plt.scatter(center[:,0],center[:,1],c = range(k),linewidths = 8)
    plt.show()        

if __name__ == '__main__':
    main()