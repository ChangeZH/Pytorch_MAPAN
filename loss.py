import torch
from model import *
import torch.nn as nn
from torch.nn import functional as F

class Distance(nn.Module):
	"""docstring for Distance"""
	def __init__(self):
		super(Distance, self).__init__()

	def forward(self,x,y):
		return F.pairwise_distance(x,y,p=2)
		

class Identity_Loss(nn.Module):
	"""docstring for Cross Identity_Loss"""
	def __init__(self):
		super(Identity_Loss, self).__init__()
		self.loss=nn.CrossEntropyLoss()
		
	def forward(self,x,y):
		return self.loss(x,y)
		
class Triplet_Loss(nn.Module):
	"""docstring for Triplet_Loss"""
	def __init__(self,xi=1.2):
		super(Triplet_Loss, self).__init__()
		self.xi=xi

		self.D=Distance()
		
	def forward(self,v_f,i_f,y):
		for i in range(0,i_f.shape[0]):
			same=self.D(v_f,i_f[i,:])*(y==y[i])
			same_max=torch.max(same)
			diff=self.D(v_f,i_f[i,:])*(y!=y[i])
			diff_min=torch.min((y==y[i])*max(diff)+diff)
			l=same_max-diff_min+self.xi
			if i==0:
				loss_i=l*(l>0)
			else:
				loss_i=loss_i+l*(l>0)

		for i in range(0,v_f.shape[0]):
			same=self.D(i_f,v_f[i,:])*(y==y[i])
			same_max=torch.max(same)
			diff=self.D(i_f,v_f[i,:])*(y!=y[i])
			diff_min=torch.min((y==y[i])*max(diff)+diff)
			l=same_max-diff_min+self.xi
			if i==0:
				loss_v=l*(l>0)
			else:
				loss_v=loss_v+l*(l>0)
		return loss_i+loss_v

class All_Loss(nn.Module):
	"""docstring for All_Loss"""
	def __init__(self, beta=1, alp=0.05, xi=1.2):
		super(All_Loss, self).__init__()
		self.beta=beta
		self.alp=alp
		self.xi=xi
		self.Loss_i=Identity_Loss()
		self.Loss_t=Triplet_Loss(self.xi)

	def forward(self,x_v,x_i,v_f,i_f,y):
		li=(self.Loss_i(x_v,y)+self.Loss_i(x_i,y))/2
		lt=self.Loss_t(v_f,i_f,y)
		return self.beta*li+self.alp*lt


if __name__=='__main__':
	loss=All_Loss()
	V=torch.rand((4,3,288,144)).cuda()
	I=torch.rand((4,3,288,144)).cuda()
	out=torch.LongTensor([0,1,0,1]).cuda()
	model=MAPAN(N=2).cuda()
	x_v,x_i,v_f,i_f=model(V,I)
	l=loss(x_v,x_i,v_f,i_f,out)
	print(l)