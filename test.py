#coding=utf-8

import os,sys
import numpy as np
from numpy.lib.stride_tricks import as_strided
import scipy
from scipy.io import wavfile
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import StratifiedShuffleSplit
#自己定义的文件
from gmmhmm import gmmhmm

reload(sys)
sys.setdefaultencoding("utf-8")

fpaths = []
labels = []
spoken = []

def getCurDir():
	path=sys.path[0]
	if os.path.isdir(path):
		return path
	elif os.path.isfile(path):
		return os.path.dirname(path)

#audio文件夹下有7个子文件夹，每个子文件夹下包含一个单词的多个音频
audioDir = getCurDir() + '/audio/'
for f in os.listdir(audioDir):
	for w in os.listdir(audioDir + f):
		fpaths.append(audioDir + f + '/' + w)#每个音频文件的路径
		labels.append(f) #每个音频文件的标签
		if f not in spoken:
			spoken.append(f)
print('Words spoken:', spoken)


data = np.zeros((len(fpaths), 32000))#7行32000列的矩阵，这里32000只要够大就行了
print("fpaths length: ", len(fpaths))
maxsize = -1
for n,file in enumerate(fpaths):
	_, d = wavfile.read(file) #读取音频文件，d为一个array。（音频文件采样率为8000，dtype=Int16.）
	data[n, :d.shape[0]] = d #将音频文件的内容放到对应的行
	if d.shape[0] > maxsize:
		maxsize = d.shape[0]
print("max file length: ", maxsize)
data = data[:, :maxsize]#一行对应一个audio文件数据，列数为最长的audio长度

#Each sample file is one row in data, and has one entry in labels
print('Number of files total:', data.shape[0])
all_labels = np.zeros(data.shape[0])
#将每一个文件对应的单词（如'apple')映射为数字如'1'
for n, label in enumerate(set(labels)):#去除重复的元素，set中保存为文件夹的名字
	all_labels[np.array([i for i, _ in enumerate(labels) if _ == label])] = n
print('Labels and label indices', all_labels)

#提取特征部分
#这里没有使用MFCC等复杂的特征，只是使用简单特征做一个演示
#使用短时傅里叶变换寻找波峰
#该函数修改自：http://stackoverflow.com/questions/2459295/stft-and-istft-in-python
def stft(x, fftsize=64, overlap_pct=.5):
	hop = int(fftsize * (1 - overlap_pct))
	w = scipy.hanning(fftsize + 1)[:-1]	
	raw = np.array([np.fft.rfft(w * x[i:i + fftsize]) for i in range(0, len(x) - fftsize, hop)])
	return raw[:, :(fftsize // 2)]

##################################################展示一个音频的例子#############################
def showExample():
	plt.plot(data[0, :], color = 'steelblue')#一个apple 音频的波形图
	plt.title('%s audio example'%labels[0])
	plt.xlim(0, 3500)
	plt.xlabel('Time (samples)')
	plt.ylabel('Amplitude (signed 16 bit)')
	plt.figure()
	#plt.show()
	#plt.close() 取消注释这一句出来的是什么？。。。。
	
	#PSD
	log_freq = 20 * np.log(np.abs(stft(data[0, :])) + 1)# + 1 to avoid log of 0
	print(log_freq.shape)#(216, 32)
	plt.imshow(log_freq, cmap='gray', interpolation=None)
	plt.xlabel('Freq (bin)')
	plt.ylabel('Time (overlapped frames)')
	plt.ylim(log_freq.shape[1])
	plt.title('PSD of %s example'%labels[0])
	plt.show()

#波峰检测，来源于：http://kkjkok.blogspot.com/2013/12/dsp-snippets_9.html
def peakfind(x, n_peaks, l_size=3, r_size=3, c_size=3, f=np.mean):
	win_size = l_size + r_size + c_size
	shape = x.shape[:-1] + (x.shape[-1] - win_size + 1, win_size)
	strides = x.strides + (x.strides[-1],)
	xs = as_strided(x, shape=shape, strides=strides)
	def is_peak(x):
		centered = (np.argmax(x) == l_size + int(c_size/2))
		l = x[:l_size]
		c = x[l_size:l_size + c_size]
		r = x[-r_size:]
		passes = np.max(c) > np.max([f(l), f(r)])
		if centered and passes:
			return np.max(c)
		else:
			return -1
	r = np.apply_along_axis(is_peak, 1, xs)
	top = np.argsort(r, None)[::-1]
	heights = r[top[:n_peaks]]
	#Add l_size and half - 1 of center size to get to actual peak location
	top[top > -1] = top[top > -1] + l_size + int(c_size / 2.)
	return heights, top[:n_peaks]

#波峰检测的例子
def detectPeakExample():
	plot_data = np.abs(stft(data[20, :]))[15, :]
	values, locs = peakfind(plot_data, n_peaks=6)
	fp = locs[values > -1]
	fv = values[values > -1]
	plt.plot(plot_data, color='steelblue')
	plt.plot(fp, fv, 'x', color='darkred')
	plt.title('Peak location example')
	plt.xlabel('Frequency (bins)')
	plt.ylabel('Amplitude')
	plt.show()


#使用高频率峰仅仅对于单用户语音起作用，对于多人语音来说需要使用更好的特征，如MFCC，或者DNN提取的特征等
all_obs = []
for i in range(data.shape[0]):
	d = np.abs(stft(data[i, :]))#216 * 32
	n_dim = 6
	obs = np.zeros((n_dim, d.shape[0]))#6 * 216
	for r in range(d.shape[0]):#0...5
		_, t = peakfind(d[r, :], n_peaks=n_dim)
		obs[:, r] = t.copy()
	if i % 10 == 0:
		print("Processed obs %s" % i)
	all_obs.append(obs)#obs.shape: 6 * 216
#all_obs此时为105个array，array的shape为 6 * 216	
all_obs = np.atleast_3d(all_obs)#3维数组，105 * 6 * 216
print(all_obs.shape)
sss = StratifiedShuffleSplit(all_labels, test_size=0.1, random_state=0)
print("sss", sss)
for n,i in enumerate(all_obs):
	all_obs[n] /= all_obs[n].sum(axis=0)

for train_index, test_index in sss:
	X_train, X_test = all_obs[train_index, ...], all_obs[test_index, ...]#训练的特征
	y_train, y_test = all_labels[train_index], all_labels[test_index]#分类标签
print('Size of training matrix:', X_train.shape)
print('Size of testing matrix:', X_test.shape)

ys = set(all_labels)
ms = [gmmhmm(6) for y in ys]
_ = [m.fit(X_train[y_train == y, :, :]) for m, y in zip(ms, ys)]
ps = [m.transform(X_test) for m in ms]
res = np.vstack(ps)
predicted_labels = np.argmax(res, axis=0)
missed = (predicted_labels != y_test)
print('Test accuracy: %.2f percent' % (100 * (1 - np.mean(missed))))
