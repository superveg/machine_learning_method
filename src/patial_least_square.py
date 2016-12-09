import random
import numpy as np
from numpy import linalg as la


class Partial_least_square():
	def __init__(self):
		self.y_is_vector = False
		self.epsilon = 10**-6
		self.Q = None
		self.beta = None
		self.P = None



	def fit(self,x,y,l = None):


		x = self.into_array(x)
		y = self.into_array(y)

		if l == None:
			l = min(x.shape[0],x.shape[1])

		tt = []
		uu = []
		pp = []
		qq = []
		for epoach in range(l):
			if self.y_is_vector:
				u = y
				t_converge = False
				t_pervious = None
				q = np.array(1)

				while not t_converge:
					p = np.dot(x.T, u)/float(la.norm(np.dot(x.T, u)))
					t = np.dot(x,p)
					if t_pervious != None:
						if la.norm(t-t_pervious)<self.epsilon:
							t_converge = True
					t_pervious = t

			else:
				j = random.randint(0,y.shape[1]-1)
				u = y.T[j]
				t_converge = False
				t_pervious = None

				while not t_converge:
					p = np.dot(x.T, u)/float(la.norm(np.dot(x.T, u)))
					t = np.dot(x,p)
					q = np.dot(y.T,t)/float(la.norm(np.dot(y.T,t)))
					u = np.dot(y,q)
					if t_pervious != None:
						if la.norm(t-t_pervious)<self.epsilon:
							t_converge = True
					t_pervious = t



			x = x - self.product_vector_to_matrix(t,p)
			y = y - self.product_vector_to_matrix(u,q)
			tt.append(t)
			uu.append(u)
			qq.append(q)
			pp.append(p)

		tt = np.array(tt)
		uu = np.array(uu)
		qq = np.array(qq)
		pp = np.array(pp) 
		beta = uu[0][0]/float(tt[0][0])
		self.beta = beta
		self.Q = qq
		self.P = pp
		return pp,qq,beta


	def predict(self,x):

		x = self.into_array(x)
		y = np.dot(x,self.P.T)
		y = np.dot(y,self.beta)
		y = np.dot(y,self.Q)
		return y



	def product_vector_to_matrix(self,x,y):

		a =[]
		for item in x:
			a.append(np.dot(item,y))

		return np.array(a)

	def into_array(self,y):

		if type(y[0]) == int:
			y = np.array(y)
			self.y_is_vector = True
		else:
			y = np.array(map(np.array, y))
		return y


	def various(self,y1,y2):

		y1 = self.into_array(y1)
		y2 = self.into_array(y2)


		if type(y1[0]) == float or type(y1[0]) == int:
			return np.dot(y1-y2,y1-y2)/float(len(y1))

		else:
			ac = 0
			for i in range(len(y1)):
				ac += np.dot(y1[i] - y2[i], y1[i] - y2[i])

		return ac/float(y1.shape[0]*y1.shape[1])



def main():
	None

if __name__ == "__main__":
	main()


