
#先用随机森林，svm试一下，转化到同一个维度
#不行就用多层神经网络，需要学会不平衡数据的训练模型
#再不行就用卷积神经网络

#数据下载网址
import wfdb
import numpy as np


e = ['100','101','102','103','104','105','106','107','108','109','111','112','113','114','115','116','117','118','119','121','122','123','124','200','201','202','203','205','207','208','209','210','212','213','214','215','217','219','220','221','222','223','228','230','231','232','233','234']
features = ['MLII', 'V1', 'V2', 'V4', 'V5'] 

for 
signals, fields = wfdb.rdsamp('100', channels = [0])
ann = wfdb.rdann('100', 'atr')

good = ['N']
ids = np.in1d(ann.symbol, good)
imp_beats = ann.sample[ids]
beats = ann.sample

Normal = []
for i in imp_beats:
    beats = list(beats)
    j = beats.index(i)
    if(j!=0 and j!=(len(beats)-1)):
        x = beats[j-1]
        y = beats[j+1]
        diff1 = abs(x - beats[j])//2
        diff2 = abs(y - beats[j])//2
        Normal.append(signals[beats[j]-diff1:beats[j]+diff2, 0])



'''这个的对新增数据切割用的
data = np.array(csv_data)
signals = []
count = 1
peaks =  biosppy.signals.ecg.christov_segmenter(signal=data, sampling_rate = 200)[0]
for i in (peaks[1:-1]):
    diff1 = abs(peaks[count - 1] - i)
    diff2 = abs(peaks[count + 1]- i)
    x = peaks[count - 1] + diff1//2
    y = peaks[count + 1] - diff2//2
    signal = data[x:y]
    signals.append(signal)
    count += 1
'''