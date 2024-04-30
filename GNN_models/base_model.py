import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

from .gcn import GCN
from .gat import GAT 
from .gin import GIN  
from .sage import SAGE
from .pna import PNA

class Classifier(nn.Module):
	def __init__(self, nhid, nclass, dropout=0., with_bn=True, with_bias=True):
		super(Classifier, self).__init__()
		self.with_bn = with_bn
		self.layer1 = nn.Linear(nhid, nhid, bias=with_bias)
		self.layer2 = nn.Linear(nhid, nclass, bias=with_bias)
		if with_bn:
			self.bn1 = nn.BatchNorm1d(nhid)

		self.dropout = dropout

		self.initialize()

	def initialize(self):
		self.layer1.reset_parameters()
		self.layer2.reset_parameters()
		self.bn1.reset_parameters()

	def forward(self, x):
		x =self.layer1(x)
		x = self.bn1(x)
		x = F.relu(x)
		x = F.dropout(x, p=self.dropout, training=self.training)
		x = self.layer2(x)

		return F.log_softmax(x, dim=1)

class LabelEncoder(nn.Module):
	def __init__(self, nhid, nclass, nlayers=3, dropout=0.5, with_bn=True, with_bias=True):
		super(LabelEncoder, self).__init__()
		self.with_bn = with_bn
		self.layers = nn.ModuleList()
		self.dropout = dropout
		
		self.layers.append(nn.Linear(nclass, nhid, bias=with_bias))
		if with_bn:
			self.bns = nn.ModuleList()
			self.bns.append(nn.BatchNorm1d(nhid))

		for i in range(nlayers-2):
			self.layers.append(nn.Linear(nhid, nhid, bias=with_bias))
			self.bns.append(nn.BatchNorm1d(nhid))
		self.layers.append(nn.Linear(nhid, nhid, bias=with_bias))

		self.initialize()

	def initialize(self):
		for m in self.layers:
			m.reset_parameters()
		if self.with_bn:
			for m in self.bns:
				m.reset_parameters()

	def forward(self, y):
		# pdb.set_trace()
		for ii, layer in enumerate(self.layers):
			if ii == len(self.layers)-1:
				return layer(y)
			y = layer(y)
			if self.with_bn:
				y = self.bns[ii](y)

			y = F.relu(y)
			y = F.dropout(y, p=self.dropout, training=self.training)


class STnet(nn.Module):
	def __init__(self, nfeat, nhid, nclass, gnn, nlayers=2, gat_heads=4, dropout=0.5, with_bn=True, with_bias=True):
		super(STnet, self).__init__()

		self.nfeat = nfeat
		self.nhid = nhid
		self.nclass = nclass

		if gnn == "GCN":
			self.gnn_model = GCN(nfeat, nhid, nhid, nlayers, dropout, with_bias, with_bn)
		elif gnn == "GAT":
			self.gnn_model = GAT(nfeat, nhid, nhid, nlayers, gat_heads, 1, dropout, with_bn)
		elif gnn == "SAGE":
			self.gnn_model = SAGE(nfeat, nhid, nhid, nlayers, dropout, with_bias, with_bn)
		elif gnn == "GIN":
			self.gnn_model = GIN(nfeat, nhid, nhid, nlayers, dropout, with_bias, with_bn)
		elif gnn == 'PNA':
			self.gnn_model = PNA(nfeat, nhid, nhid, deg, nlayers, dropout, with_bias, with_bn)
		else:
			raise Exception("gnn mode error!")

		self.classifier = Classifier(nhid, nclass)

	def sum_pool(self, x, batch, dim):
		# pdb.set_trace()
		x = torch.cat([torch.sum(x[batch==i], dim=0).view(1, -1) for i in torch.unique(batch)])
		return x

	def avg_pool(self, x, batch, dim):
		# pdb.set_trace()
		x = torch.cat([torch.mean(x[batch==i], dim=0) for i in torch.unique(batch)])
		return x.view(-1, dim)

	def forward(self, x, edge_index, batch):
		# pdb.set_trace()
		x = x.view(batch.shape[0], -1)
		node_emb = self.gnn_model(x, edge_index)
		graph_rep = self.avg_pool(node_emb, batch, self.nhid)
		# graph_rep = self.sum_pool(node_emb, batch, self.nhid)

		graph_rep = graph_rep / torch.norm(graph_rep, p=2, dim=1, keepdim=True)

		pred = self.classifier(graph_rep)

		return pred, graph_rep


class linear_attention(nn.Module):
	def __init__(self, in_dim):
		super(linear_attention, self).__init__()
		self.layerQ = nn.Linear(in_dim, in_dim)
		self.layerK = nn.Linear(in_dim, in_dim)
		self.layerV = nn.Linear(in_dim, in_dim)
		self.initialize()

	def initialize(self):
		self.layerQ.reset_parameters()
		self.layerK.reset_parameters()
		self.layerV.reset_parameters()

	def forward(self, node_emb, label_emb, tau=0.5):
		# pdb.set_trace()
		Q = self.layerQ(label_emb)
		K = self.layerK(node_emb)
		V = self.layerV(node_emb)
		attention_score = torch.matmul(Q, K.transpose(-2, -1))
		attention_weight = F.softmax(attention_score * tau, dim=1)
		z = torch.matmul(attention_weight, V)
		return z


class Tenet(nn.Module):
	def __init__(self, nfeat, nhid, nclass, gnn, nlayers=2, gat_heads=4, dropout=0.5, tau=0.1, with_bn=True, with_bias=True):
		super(Tenet, self).__init__()
		self.nfeat = nfeat
		self.nhid = nhid
		self.nclass = nclass
		self.tau = tau

		if gnn == "GCN":
			self.gnn_model = GCN(nfeat, nhid, nhid, nlayers, dropout, with_bias, with_bn)
		elif gnn == "GAT":
			self.gnn_model = GAT(nfeat, nhid, nhid, nlayers, gat_heads, 1, dropout, with_bn)
		elif gnn == "SAGE":
			self.gnn_model = SAGE(nfeat, nhid, nhid, nlayers, dropout, with_bias, with_bn)
		elif gnn == "GIN":
			self.gnn_model = GIN(nfeat, nhid, nhid, nlayers, dropout, with_bias, with_bn)
		elif gnn == 'PNA':
			self.gnn_model = PNA(nfeat, nhid, nhid, deg, nlayers, dropout, with_bias, with_bn)
		else:
			raise Exception("gnn mode error!")

		self.label_encoder = LabelEncoder(nhid, nclass)
		self.classifier = Classifier(nhid, nclass)
		self.label_attentive = linear_attention(nhid)

	def avg_pool(self, x, batch, dim):
		# pdb.set_trace()
		x = torch.cat([torch.mean(x[batch==i], dim=0) for i in torch.unique(batch)])
		return x.view(-1, dim)

	def sum_pool(self, x, batch, dim):
		# pdb.set_trace()
		x = torch.cat([torch.sum(x[batch==i], dim=0).view(1, -1) for i in torch.unique(batch)])
		return x

	def forward(self, x, y, edge_index, batch):
		# pdb.set_trace()
		x = x.view(batch.shape[0], -1)
		fea = self.gnn_model(x, edge_index)

		y = F.one_hot(y, num_classes = self.nclass)*1.0
		label_emb = self.label_encoder(y)

		node_emb = self.label_attentive(fea, label_emb, self.tau)

		graph_rep = self.avg_pool(node_emb, batch, self.nhid)
		# graph_rep = self.sum_pool(node_emb, batch, self.nhid)

		graph_rep = graph_rep / torch.norm(graph_rep, p=2, dim=1, keepdim=True)

		pred = self.classifier(graph_rep)

		return pred, graph_rep