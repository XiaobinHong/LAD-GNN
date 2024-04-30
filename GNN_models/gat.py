import torch
import torch.nn as nn 
import torch.nn.functional as F 

from torch_geometric.nn import GATConv
from torch_sparse import coalesce, SparseTensor

class GAT(nn.Module):
	def __init__(self, nfeat, nhid, nclass, nlayers=2, nheads=4, output_heads=1, dropout=0.5, with_bn=True):
		super(GAT, self).__init__()

		self.with_bn = with_bn
		self.dropout = dropout
		self.name = "GAT"

		if with_bn:
			self.bns = nn.ModuleList()
			self.bns.append(nn.BatchNorm1d(nhid*nheads))

		self.gat_layers = nn.ModuleList()
		self.gat_layers.append(GATConv(nfeat, nhid, heads=nheads, dropout=dropout))

		for _ in range(nlayers-2):
			self.gat_layers.append(GATConv(nhid*nheads, nhid, heads=nheads, dropout=dropout))
			if with_bn:
				self.bns.append(nn.BatchNorm1d(nhid*nheads))

		self.gat_layers.append(GATConv(nhid*nheads, nclass, heads=output_heads, concat=False, dropout=dropout))

		self.initialize()

	def forward(self, x, edge_index, edge_weight=None):
		if edge_weight is not None:
			adj = SparseTensor.from_edge_index(edge_index, edge_weight, sparse_sizes=2 * x.shape[:1]).t()

		for ii, layer in enumerate(self.gat_layers[:-1]):
			if edge_weight is not None:
				x = layer(x, adj)
			else:
				x = layer(x, edge_index, edge_weight)
			if self.with_bn:
				x = self.bns[ii](x)
			x = F.relu(x)
			x = F.dropout(x, p=self.dropout, training = self.training)

		if edge_weight is not None:
			x = self.gat_layers[-1](x, adj)
		else:
			x = self.gat_layers[-1](x, edge_index, edge_weight)
		return x

	def initialize(self):
		for m in self.gat_layers:
			m.reset_parameters()
		if self.with_bn:
			for m in self.bns:
				m.reset_parameters()