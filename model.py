import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F

class Res_Block(nn.Module):
	"""docstring for Res_Block"""
	def __init__(self):
		super(Res_Block, self).__init__()
		self.res_block1=nn.Sequential(
			torchvision.models.resnet50(pretrained=True).conv1,
			torchvision.models.resnet50(pretrained=True).bn1,
			torchvision.models.resnet50(pretrained=True).relu,
			torchvision.models.resnet50(pretrained=True).maxpool)
		self.res_block2=torchvision.models.resnet50(pretrained=True).layer1
		self.res_block3=torchvision.models.resnet50(pretrained=True).layer2
		self.res_block4=torchvision.models.resnet50(pretrained=True).layer3
		self.res_block5=torchvision.models.resnet50(pretrained=True).layer4
		self.avgpool=torchvision.models.resnet50(pretrained=True).avgpool

	def forward(self,x):
		x=self.res_block1(x)
		x=self.res_block2(x)
		x=self.res_block3(x)
		x=self.res_block4(x)
		x5=self.res_block5(x)
		x=self.avgpool(x5)
		return x,x5

class Grid_Network(nn.Module):
	"""docstring for Grid_Network"""
	def __init__(self):
		super(Grid_Network, self).__init__()
		self.avgpool=nn.AdaptiveAvgPool2d(1)
		self.fc1=nn.Linear(2048,64)
		self.fc2=nn.Linear(64,6)

	def forward(self,x):
		batchsize=x.shape[0]
		x=self.avgpool(x)
		x=torch.reshape(x,(batchsize,-1))
		x=self.fc1(x)
		x=self.fc2(x)
		x=torch.reshape(x,(batchsize,2,3))
		return x

class Affine_Trans(nn.Module):
	"""docstring for Affine_Trans"""
	def __init__(self):
		super(Affine_Trans,self).__init__()

	def forward(self,x,A):
		[batchsize,C,W,H]=x.shape
		grid=F.affine_grid(A,x.size())
		out=F.grid_sample(x,grid,mode='bilinear',align_corners=False)
		return out

class Feature_Embed(nn.Module):
	"""docstring for Feature_Embed"""
	def __init__(self,N):
		super(Feature_Embed,self).__init__()
		self.N=N

		self.fc1_1=nn.Sequential(
			nn.Linear(2048,512),
			nn.BatchNorm1d(512))
		self.fc2_1=nn.Sequential(
			nn.ReLU(),
			nn.Linear(512,N),
			nn.Softmax(dim=1))

		self.fc1_2=nn.Sequential(
			nn.Linear(2048,512),
			nn.BatchNorm1d(512))
		self.fc2_2=nn.Sequential(
			nn.ReLU(),
			nn.Linear(512,N),
			nn.Softmax(dim=1))

	def forward(self,x1,x2):
		batchsize=x1.shape[0]
		x1=torch.reshape(x1,(batchsize,-1))
		x2=torch.reshape(x2,(batchsize,-1))

		x1=self.fc1_1(x1)
		feature1=x1
		x1=self.fc2_1(x1)

		x2=self.fc1_1(x2)
		feature2=x2
		x2=self.fc2_1(x2)

		return x1,x2,feature1,feature2

class MAPAN(nn.Module):
	"""docstring for MAPAN"""
	def __init__(self, N, lam=0.9):
		super(MAPAN, self).__init__()
		self.N=N
		self.lam=lam

		self.Res_Block_V=Res_Block()
		self.Res_Block_A=Res_Block()
		self.Res_Block_I=Res_Block()

		self.Grid_Network=Grid_Network()
		self.Affine_Trans=Affine_Trans()
		self.Feature_Embed=Feature_Embed(self.N)

	def forward(self,V,I):
		v,v5=self.Res_Block_V(V)
		i,_=self.Res_Block_I(I)

		A=self.Grid_Network(v5)
		a=self.Affine_Trans(V,A)

		a,_=self.Res_Block_A(a)
		v=self.lam*v+(1-self.lam)*a

		v,i,v_f,i_f=self.Feature_Embed(v,i)

		return v,i,v_f,i_f

if __name__=='__main__':
	V=torch.rand((2,3,288,144))
	I=torch.rand((2,3,288,144))
	model=MAPAN(N=4)
	a,b,c,d=model(V,I)