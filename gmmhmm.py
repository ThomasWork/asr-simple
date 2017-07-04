#coding='UTF-8'
import scipy.stats as st
import numpy as np

class gmmhmm:
	#This class converted with modifications from https://code.google.com/p/hmm-speech-recognition/source/browse/Word.m
	def __init__(self, n_states):
		self.n_states = n_states#状态数量
		self.random_state = np.random.RandomState(0)#定义一个产生随机数的变量
		
		#Normalize random initial state
		#归一化随机初始化的状态
		#一个列向量，表示每个状态的第一次出现的概率
		self.prior = self._normalize(self.random_state.rand(self.n_states, 1))
		#状态转移矩阵，随机赋值
		self.A = self._stochasticize(self.random_state.rand(self.n_states, self.n_states))
		self.mu = None
		self.covs = None
		self.n_dims = None
		   
	def _forward(self, B):
		log_likelihood = 0.
		T = B.shape[1]
		alpha = np.zeros(B.shape)
		for t in range(T):
			if t == 0:
				alpha[:, t] = B[:, t] * self.prior.ravel()
			else:
				alpha[:, t] = B[:, t] * np.dot(self.A.T, alpha[:, t - 1])
		 
			alpha_sum = np.sum(alpha[:, t])
			alpha[:, t] /= alpha_sum
			log_likelihood = log_likelihood + np.log(alpha_sum)
		return log_likelihood, alpha
	
	def _backward(self, B):
		T = B.shape[1]
		beta = np.zeros(B.shape);
		   
		beta[:, -1] = np.ones(B.shape[0])
			
		for t in range(T - 1)[::-1]:
			beta[:, t] = np.dot(self.A, (B[:, t + 1] * beta[:, t + 1]))
			beta[:, t] /= np.sum(beta[:, t])
		return beta
	
	def _state_likelihood(self, obs):
		obs = np.atleast_2d(obs)
		B = np.zeros((self.n_states, obs.shape[1]))
		for s in range(self.n_states):
			#Needs scipy 0.14
			np.random.seed(self.random_state.randint(1))
			B[s, :] = st.multivariate_normal.pdf(
				obs.T, mean=self.mu[:, s].T, cov=self.covs[:, :, s].T)
			#This function can (and will!) return values >> 1
			#See the discussion here for the equivalent matlab function
			#https://groups.google.com/forum/#!topic/comp.soft-sys.matlab/YksWK0T74Ak
			#Key line: "Probabilities have to be less than 1,
			#Densities can be anything, even infinite (at individual points)."
			#This is evaluating the density at individual points...
		return B
	
	def _normalize(self, x):
		return (x + (x == 0)) / np.sum(x)
	
	def _stochasticize(self, x):
		return (x + (x == 0)) / np.sum(x, axis=1)#按行求和，然后行加起来等于1
	
	#使用这个函数可以在构造函数中使用更少的参数
	def _em_init(self, obs):
		if self.n_dims is None:
			self.n_dims = obs.shape[0]#在例子中为4，obs.shape=(4,40)
		if self.mu is None:
			#进行不放回的抽样
			subset = self.random_state.choice(np.arange(self.n_dims), size=self.n_states, replace=False)
			self.mu = obs[:, subset]
		if self.covs is None:
			self.covs = np.zeros((self.n_dims, self.n_dims, self.n_states))#4 * 4 * 2
			#首先求协方差，然后利用两次diag函数取得对角阵
			temp = np.diag(np.diag(np.cov(obs)))
		#	print temp.shape
		#	print self.covs
			self.covs += temp[:, :, None]
		return self
	
	def _em_step(self, obs): 
		obs = np.atleast_2d(obs)
		B = self._state_likelihood(obs)
		T = obs.shape[1]
		
		log_likelihood, alpha = self._forward(B)
		beta = self._backward(B)
		
		xi_sum = np.zeros((self.n_states, self.n_states))
		gamma = np.zeros((self.n_states, T))
		
		for t in range(T - 1):
			partial_sum = self.A * np.dot(alpha[:, t], (beta[:, t] * B[:, t + 1]).T)
			xi_sum += self._normalize(partial_sum)
			partial_g = alpha[:, t] * beta[:, t]
			gamma[:, t] = self._normalize(partial_g)
			  
		partial_g = alpha[:, -1] * beta[:, -1]
		gamma[:, -1] = self._normalize(partial_g)
		
		expected_prior = gamma[:, 0]
		expected_A = self._stochasticize(xi_sum)
		
		expected_mu = np.zeros((self.n_dims, self.n_states))
		expected_covs = np.zeros((self.n_dims, self.n_dims, self.n_states))
		
		gamma_state_sum = np.sum(gamma, axis=1)
		#Set zeros to 1 before dividing
		gamma_state_sum = gamma_state_sum + (gamma_state_sum == 0)
		
		for s in range(self.n_states):
			gamma_obs = obs * gamma[s, :]
			expected_mu[:, s] = np.sum(gamma_obs, axis=1) / gamma_state_sum[s]
			partial_covs = np.dot(gamma_obs, obs.T) / gamma_state_sum[s] - np.dot(expected_mu[:, s], expected_mu[:, s].T)
			#Symmetrize
			partial_covs = np.triu(partial_covs) + np.triu(partial_covs).T - np.diag(partial_covs)
		
		#Ensure positive semidefinite by adding diagonal loading
		expected_covs += .01 * np.eye(self.n_dims)[:, :, None]
		
		self.prior = expected_prior
		self.mu = expected_mu
		self.covs = expected_covs
		self.A = expected_A
		return log_likelihood
	
	def fit(self, obs, n_iter=15):
		#Support for 2D and 3D arrays
		#2D should be n_features, n_dims
		#3D should be n_examples, n_features, n_dims
		#For example, with 6 features per speech segment, 105 different words
		#this array should be size
		#(105, 6, X) where X is the number of frames with features extracted
		#For a single example file, the array should be size (6, X)
		print(obs.shape)
		if len(obs.shape) == 2:
			for i in range(n_iter):
				self._em_init(obs)#实际上只会执行一次
				log_likelihood = self._em_step(obs)
				print log_likelihood
		elif len(obs.shape) == 3:
			count = obs.shape[0]
			for n in range(count):
				for i in range(n_iter):
					self._em_init(obs[n, :, :])
					log_likelihood = self._em_step(obs[n, :, :])
		return self
	
	def transform(self, obs):
		#Support for 2D and 3D arrays
		#2D should be n_features, n_dims
		#3D should be n_examples, n_features, n_dims
		#For example, with 6 features per speech segment, 105 different words
		#this array should be size
		#(105, 6, X) where X is the number of frames with features extracted
		#For a single example file, the array should be size (6, X)
		if len(obs.shape) == 2:
			B = self._state_likelihood(obs)
			log_likelihood, _ = self._forward(B)
			return log_likelihood
		elif len(obs.shape) == 3:
			count = obs.shape[0]
			out = np.zeros((count,))
			for n in range(count):
				B = self._state_likelihood(obs[n, :, :])
				log_likelihood, _ = self._forward(B)
				out[n] = log_likelihood
			return out

if __name__ == "__main__":
	rstate = np.random.RandomState(0) #0表示局部的随机数种子，调用该方法可以获得一个对象
	#每次运行程序生成的随机数序列结果相同，从而可以复现
	#这里相当于保存了一个np.random的引用。

	t1 = np.ones((4, 40)) + .001 * rstate.rand(4, 40)#4行40列的均匀分布,[0,1),   type(t1) = ndarray
	t1 /= t1.sum(axis=0)#对每一列进行归一化。t1.astype(int),可以将元素类型转换为整型
	t2 = rstate.rand(*t1.shape)#shape为tuple类型，这样可以解包
	t2 /= t2.sum(axis=0)
	
	m1 = gmmhmm(2)
	m1.fit(t1)
	m2 = gmmhmm(2)
	m2.fit(t2)
	
	m1t1 = m1.transform(t1)
	m2t1 = m2.transform(t1)
	print("Likelihoods for test set 1")
	print("M1:", m1t1)
	print("M2:", m2t1)
	print("Prediction for test set 1")
	print("Model", np.argmax([m1t1, m2t1]) + 1)
	print()
	
	m1t2 = m1.transform(t2)
	m2t2 = m2.transform(t2)
	print("Likelihoods for test set 2")
	print("M1:", m1t2)
	print("M2:", m2t2)
	print("Prediction for test set 2")
	print("Model", np.argmax([m1t2, m2t2]) + 1)