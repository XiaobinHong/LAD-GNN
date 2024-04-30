import torch
import torch.nn as nn 
import torch.nn.functional as F 

from torch_geometric.nn import PNAConv
from torch_sparse import coalesce, SparseTensor

class PNA(nn.Module):
	def __init__(self, nfeat, nhid, nclass, deg, nlayers=2, dropout=0.5, with_bn=True, with_bias=True, save_mem=True):
		super(PNA, self).__init__()
		aggregators = ['mean', 'min', 'max', 'std']
		scalers = ['identity', 'amplification', 'attenuation']
		self.gcn_layers = nn.ModuleList()
		if with_bn:
			self.bns = nn.ModuleList()

		self.nlayers = nlayers
		if nlayers == 1:
			self.gcn_layers.append(PNAConv(in_channels=nfeat, out_channels=nclass,
						   aggregators=aggregators, scalers=scalers, deg=deg))
		else:
			self.gcn_layers.append(PNAConv(in_channels=nfeat, out_channels=nhid,
						   aggregators=aggregators, scalers=scalers, deg=deg))
			if with_bn:
				self.bns.append(nn.BatchNorm1d(nhid))

			for i in range(nlayers-1):
				self.gcn_layers.append(PNAConv(in_channels=nhid, out_channels=nclass,
						   aggregators=aggregators, scalers=scalers, deg=deg))

				if with_bn:
					self.bns.append(nn.BatchNorm1d(nhid))

		self.gcn_layers.append(PNAConv(in_channels=nhid, out_channels=nclass,
						   aggregators=aggregators, scalers=scalers, deg=deg))

		self.dropout = dropout
		self.with_bn = with_bn
		self.name = f"{nlayers}-layers PNA"

	def forward(self, x, edge_index, edge_weight=None):
		x, edge_index, edge_weight = self._ensure_contiguousness(x, edge_index, edge_weight)
		if edge_weight is not None:
			adj = SparseTensor.from_edge_index(edge_index, edge_weight, sparse_size=2*x.shape[:1]).t()

		for ii, layer in enumerate(self.gcn_layers):
			if edge_weight is not None:
				x = layer(x, adj)
			else:
				x = layer(x, edge_index)

			if ii != len(self.gcn_layers)-1:
				if self.with_bn:
					x = self.bns[ii](x)
				x = F.relu(x)
				x = F.dropout(x, p=self.dropout, training=self.training)

			x = self.gcn_layers[-1](x, edge_index)
			return x

	def initialize(self):
		for m in self.gcn_layers:
			m.reset_parameters()
		if self.with_bn:
			for bn in self.bns:
				bn.reset_parameters()

	def _ensure_contiguousness(self,
							   x,
							   edge_idx,
							   edge_weight):
		if not x.is_sparse:
			x = x.contiguous()
		if hasattr(edge_idx, 'contiguous'):
			edge_idx = edge_idx.contiguous()
		if edge_weight is not None:
			edge_weight = edge_weight.contiguous()
		return x, edge_idx, edge_weight