"""
kmeans++算法产生初始点
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets

#用kmeans++产生初始点
#选一个初始点簇，计算其他点与初始点簇的距离，最小的作为dn，把dn相加df，随机生成[0,1)的r,选r<df对应的点加入初始点簇
def random_center(data,k):
    row = data.shape[0]
    index = np.random.choice(range(0,row),1)
    center = data[index,:]
    for i in range(k-1):
        data_new = np.delete(data,index,0)  #去除中心点簇
        n = center.shape[0] #中心点个数
        distance = np.zeros((data_new.shape[0],n))
        for j in range(n):
            distance[:,j] = np.sqrt(np.sum((data_new - center[j,:])**2,axis=1)) #计算每个点与中心点簇的距离
        distance = np.min(distance,axis = 1)
        distance = distance/np.sum(distance,axis=0)
        distance = np.cumsum(distance,axis=0) 
        r = np.random.rand(1)
        index = np.argwhere(distance >= r)[0]
        center = np.append(center,data_new[index,:],axis = 0)
        data = data_new
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