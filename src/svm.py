import random
import numpy as np

class SVM():
	"""docstring for svm"""
	def __init__(self, x,y,c,gamma,T):
		self.x = np.array(map(lambda a: np.array(a),x))
		self.y = np.array(y)
		self.c = c
		self.gamma = float(gamma)
		self.T = int(T)
		self.w = None
		self.py = None

	def shuffel_date(self,x,y):
		x2 = []
		y2 =[]
		n = range(len(y))
		random.shuffle(n)
		for i in n:
			x2.append(x[i])
			y2.append(y[i])
		return x2,y2


	def learn_svm(self, gamma_func = None):
	

		w = np.array([0] * len(self.x[0]))
		t = 0

		for item in range(self.T):

			self.shuffel_date(self.x,self.y)
			for i in range(len(self.y)):
				t+=1
				if gamma_func == None:
				 	gamma= self.gamma_t(t)
				else:
					gamma= gamma_func(t)
				if self.y[i] * np.dot(self.x[i], w) <=1:
					w = (1- gamma) * w + gamma *self.c * self.y[i] * self.x[i]
				else:
					w = (1 - gamma) *w
			
		self.w = w			
		return w

	def gamma_t(self,t):
		return self.gamma/(1 + self.gamma *t / self.c)


	def predict(self,x2):
		if self.w == None:
			self.learn_svm()

		x2 = np.array(map(lambda a: np.array(a),x2))
		y = np.dot(x2, self.w)

		for i in range(len(y)):
			if y[i]>=0:
				y[i]=1
			else:
				y[i]=-1
		self.ty = y
		
		return y

		
	
	def accuracy(self,x2,y2):
		if self.py ==None:
			y = self.predict(x2)
		else:
			y = self.py
		ac = 0
		n = len(y2)
		for i in range(n):
			if y[i]==y2[i]:
				ac+=1

		return float(ac)/n

	def f_score(self,x2,y2):
		if self.py ==None:
			y = self.predict(x2)
		else:
			y = self.py
		n = len(y2)
		tp = 0
		fp = 0
		fn = 0
		for i in range(n):
			if y[i] == 1:
				if y[i] == y2[i]:
					tp += 1
				else:
					fp += 1
			else:
				if y[i] != y2[i]:
					fn +=1
		precision = float(tp)/(tp + fp) 
		recall = float(tp)/(tp + fn)
		f = 2* (precision * recall)/(precision + recall)
		return precision, recall, f




		
